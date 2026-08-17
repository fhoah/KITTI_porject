import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "CNNSwinEncoder requires timm. Install it with `pip install timm`."
    ) from exc


DEBUG_FEATURE = True


class CNNSwinEncoder(nn.Module):
    """Hybrid local-CNN and Swin feature encoder for RAFT-Stereo."""

    def __init__(
        self,
        output_dim=256,
        norm_fn="instance",
        dropout=0.0,
        downsample=3,
        debug_feature=False,
    ):
        super().__init__()

        if downsample != 2:
            raise ValueError(
                "This encoder expects downsample=2 for H/4 features."
            )

        self.output_dim = output_dim
        self.downsample = downsample
        self.debug_feature = DEBUG_FEATURE or debug_feature
        self.debug_output_dir = Path(__file__).resolve().parents[1] / "debug"
        self._fusion_debugged = False
        self.debug_stats = {}

        # Local CNN branch: H x W -> H/4 x W/4
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(48, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(96, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(96, affine=True),
            nn.ReLU(inplace=True),
        )

        # Global Swin branch: use stage 0 to preserve spatial detail.
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(0,),
            strict_img_size=False,
        )

        self.swin_channels = self.swin.feature_info.channels()[0]

        # Fuse CNN local feature and Swin global feature.
        self.fusion = nn.Sequential(
            nn.Conv2d(
                96 + self.swin_channels,
                256,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else None
        )

        # ImageNet normalization for pretrained Swin.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self._initialize_new_layers()

    def _initialize_new_layers(self):
        for module in (self.cnn_branch, self.fusion):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )

    def _to_nchw(self, feature):
        if feature.ndim != 4:
            raise RuntimeError(
                f"Unexpected Swin feature rank: {feature.ndim}"
            )

        if feature.shape[1] == self.swin_channels:
            return feature

        if feature.shape[-1] == self.swin_channels:
            return feature.permute(0, 3, 1, 2).contiguous()

        raise RuntimeError(
            f"Unexpected Swin feature shape: {tuple(feature.shape)}"
        )

    def _prepare_swin_input(self, x):
        # RAFT input is [-1, 1]. Convert back to [0, 1].
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

        return (
            x - self.imagenet_mean.to(dtype=x.dtype)
        ) / self.imagenet_std.to(dtype=x.dtype)

    def save_debug_feature(self, label, feature):
        if not self.debug_feature:
            return

        if label == "Fusion feature" and self._fusion_debugged:
            return

        detached = feature.detach().float().cpu()
        finite = detached[torch.isfinite(detached)]
        if finite.numel() == 0:
            stats = {
                "shape": tuple(detached.shape),
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        else:
            stats = {
                "shape": tuple(detached.shape),
                "mean": finite.mean().item(),
                "std": finite.std(unbiased=False).item(),
                "min": finite.min().item(),
                "max": finite.max().item(),
            }
        self.debug_stats[label] = stats

        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.lower().replace(" ", "_")
        for channel_index in range(min(8, detached.shape[1])):
            channel = detached[0, channel_index].numpy()
            finite_mask = np.isfinite(channel)
            image = np.zeros(channel.shape, dtype=np.uint8)
            if finite_mask.any():
                finite_values = channel[finite_mask]
                value_min = finite_values.min()
                value_max = finite_values.max()
                if value_max > value_min:
                    normalized = (finite_values - value_min) / (value_max - value_min)
                    image[finite_mask] = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(
                self.debug_output_dir / f"{safe_label}_{channel_index}.png"
            )

        if label == "Fusion feature":
            self._fusion_debugged = True

    def forward(self, x, dual_inp=False):
        del dual_inp  # Kept only for BasicEncoder API compatibility.

        self._fusion_debugged = False

        is_list = isinstance(x, (tuple, list))

        if is_list:
            if len(x) == 0:
                raise ValueError("Input image list must not be empty.")

            batch_size = x[0].shape[0]

            if any(image.shape[0] != batch_size for image in x):
                raise ValueError(
                    "All input images must use the same batch size."
                )

            x = torch.cat(x, dim=0)

        original_h, original_w = x.shape[-2:]

        # Swin patch embedding requires dimensions divisible by 4.
        pad_h = (-original_h) % 4
        pad_w = (-original_w) % 4

        if pad_h or pad_w:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )

        cnn_feature = self.cnn_branch(x)
        if not self.training:
            print("CNN feature shape", tuple(cnn_feature.shape))
        self.save_debug_feature("CNN feature", cnn_feature)

        swin_input = self._prepare_swin_input(x)
        swin_feature = self.swin(swin_input)[0]
        swin_feature = self._to_nchw(swin_feature)

        if swin_feature.shape[-2:] != cnn_feature.shape[-2:]:
            raise RuntimeError(
                "CNN and Swin H/4 feature resolutions do not match: "
                f"{tuple(cnn_feature.shape[-2:])} vs "
                f"{tuple(swin_feature.shape[-2:])}"
            )

        if not self.training:
            print("Swin feature shape", tuple(swin_feature.shape))
        self.save_debug_feature("Swin feature", swin_feature)

        fused = torch.cat(
            [cnn_feature, swin_feature],
            dim=1,
        )

        output = self.fusion(fused)

        expected_h = (original_h + 3) // 4
        expected_w = (original_w + 3) // 4

        if output.shape[-2:] != (expected_h, expected_w):
            output = output[
                ...,
                :expected_h,
                :expected_w,
            ]

        if self.training and self.dropout is not None:
            output = self.dropout(output)

        if not torch.isfinite(output).all():
            raise RuntimeError(
                "CNNSwinEncoder produced NaN or Inf features."
            )

        fusion_stats = output.detach().float()
        fusion_stats.mean().item()
        fusion_stats.std(unbiased=False).item()
        fusion_stats.min().item()
        fusion_stats.max().item()

        if not self.training:
            print("Fusion feature shape", tuple(output.shape))
        self.save_debug_feature("Fusion feature", output)

        if is_list:
            output = output.split(batch_size, dim=0)

        return output
