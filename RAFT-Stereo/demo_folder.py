import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "core"))

import os
import glob
import time
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder


# =========================================
# GPU加速
# =========================================

torch.backends.cudnn.benchmark = True


# =========================================
# 路徑
# =========================================

DATASET_VAL_DIR = (SCRIPT_DIR / ".." / "KITTI_dataset" / "val").resolve()
LEFT_DIR = DATASET_VAL_DIR / "left"
RIGHT_DIR = DATASET_VAL_DIR / "right"
GT_DIR = DATASET_VAL_DIR / "disp"

OUTPUT_DIR = r"C:\Users\user\Desktop\KITTI_porject\RAFT_output"
VISUALIZATION_DIR = SCRIPT_DIR / "results_visualization"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

VISUALIZATION_DIR.mkdir(exist_ok=True, parents=True)


# =========================================
# Match validation samples by basename
# =========================================

left_by_name = {
    path.stem: path
    for path in LEFT_DIR.glob("*.png")
    if path.is_file()
}
right_by_name = {
    path.stem: path
    for path in RIGHT_DIR.glob("*.png")
    if path.is_file()
}
disp_by_name = {
    path.stem: path
    for path in GT_DIR.glob("*.npy")
    if path.is_file()
}

matched_names = sorted(
    set(left_by_name) & set(right_by_name) & set(disp_by_name)
)
matched_samples = [
    (left_by_name[name], right_by_name[name], disp_by_name[name])
    for name in matched_names
]

print(f"Resolved left path: {LEFT_DIR}")
print(f"Resolved right path: {RIGHT_DIR}")
print(f"Resolved disparity path: {GT_DIR}")
print(f"Matched sample count: {len(matched_samples)}")

if not matched_samples:
    print(
        "ERROR: No validation samples have matching left PNG, right PNG, "
        "and disparity NPY basenames. Exiting without creating CSV files."
    )
    raise SystemExit(0)


# =========================================
# RAFT參數
# =========================================

class Args:

    hidden_dims = [128] * 3
    corr_implementation = "alt"
    shared_backbone = False
    corr_levels = 4
    corr_radius = 4
    n_downsample = 2
    context_norm = "batch"
    slow_fast_gru = False
    mixed_precision = True
    n_gru_layers = 3


args = Args()


# =========================================
# 載入模型
# =========================================

print("Loading Model...")

model = torch.nn.DataParallel(
    RAFTStereo(args)
)
model.load_state_dict(
    torch.load(
        r"C:\Users\user\Desktop\KITTI_porject\RAFT-Stereo\checkpoints\10000_cnn_swin_h4_10k_5epoch.pth"
    )
)

model = model.module
model.cuda()
model.eval()

print("Model Loaded")


# =========================================
# 找圖
# =========================================

MAX_IMAGES = 10

matched_samples = matched_samples[:MAX_IMAGES]

print(
    f"Total Images: {len(matched_samples)}"
)


# =========================================
# 統計變數
# =========================================

results = []

pixel_results = []

mae_list = []


print(
    "Sign convention: RAFT-Stereo predicts negative horizontal flow for "
    "positive left-view disparity, so disparity = -flow_up; abs(flow_up) "
    "is not used because it would make wrong-sign flow appear valid."
)


# =========================================
# 開始
# =========================================

pbar = tqdm(
    enumerate(matched_samples),
    total=len(matched_samples),
    desc="RAFT Processing",
    unit="img"
)

for idx, (left_path, right_path, gt_path) in pbar:

    start = time.time()

    filename = left_path.name

    # =====================================
    # 讀圖
    # =====================================

    imgL = np.array(
        Image.open(left_path)
    )

    imgR = np.array(
        Image.open(right_path)
    )

    h, w = imgL.shape[:2]

    imgL_tensor = torch.from_numpy(
        imgL
    ).permute(
        2,
        0,
        1
    ).float()[None].cuda()

    imgR_tensor = torch.from_numpy(
        imgR
    ).permute(
        2,
        0,
        1
    ).float()[None].cuda()

    # =====================================
    # Padding
    # =====================================

    padder = InputPadder(
        imgL_tensor.shape,
        divis_by=32
    )

    imgL_tensor, imgR_tensor = padder.pad(
        imgL_tensor,
        imgR_tensor
    )

    # =====================================
    # 推論
    # =====================================

    with torch.no_grad():

        _, flow_up = model(
            imgL_tensor,
            imgR_tensor,
            iters=16,      # 速度優先
            test_mode=True
        )
    import matplotlib.pyplot as plt

    raw_flow = flow_up.squeeze().cpu().numpy()
    finite_raw_flow = raw_flow[np.isfinite(raw_flow)]
    if finite_raw_flow.size:
        print(
            "Raw horizontal flow stats "
            f"(before sign conversion): min={finite_raw_flow.min():.6f}, "
            f"max={finite_raw_flow.max():.6f}, "
            f"mean={finite_raw_flow.mean():.6f}"
        )
    else:
        print("Raw horizontal flow stats: no finite values")

    # StereoDataset trains RAFT-Stereo with horizontal flow = -disparity.
    # Negation preserves that convention; abs() would turn invalid positive
    # horizontal flow into apparently valid positive disparity.
    disparity = -raw_flow
    disparity = disparity[:h, :w]
    valid_disparity = np.isfinite(disparity) & (disparity > 0.0)

    finite_positive_disparity = disparity[valid_disparity]
    if finite_positive_disparity.size:
        print(
            "Predicted disparity stats "
            f"(after disparity=-flow_up): min={finite_positive_disparity.min():.6f}, "
            f"max={finite_positive_disparity.max():.6f}, "
            f"mean={finite_positive_disparity.mean():.6f}"
        )
    else:
        print("Predicted disparity stats: no finite positive values")

    disparity[~valid_disparity] = 0.0

    plt.figure(figsize=(12,4))
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()
    plt.title("Predicted Disparity")
    plt.savefig("disp_test.png", bbox_inches='tight')
    plt.close()
    print("已儲存 disp_test.png")

    # =====================================
    # Depth
    # =====================================

    focal = 721.5377
    baseline = 0.53715

    depth = np.zeros_like(
        disparity
    )

    depth[valid_disparity] = (

        focal * baseline

    ) / (

        disparity[valid_disparity]

    )

    depth = np.clip(
        depth,
        0,
        80
    )

    # =====================================
    # GT
    # =====================================

    gt_disparity = np.load(
        gt_path
    ).astype(np.float32)[:h, :w]

    valid_gt_disparity = np.isfinite(gt_disparity) & (gt_disparity > 0.0)
    gt_depth = np.zeros_like(gt_disparity, dtype=np.float32)
    gt_depth[valid_gt_disparity] = (
        focal * baseline
    ) / gt_disparity[valid_gt_disparity]

    eval_mask = (

        (gt_depth > 0)
        &
        (depth > 0)

    )

    if np.sum(eval_mask) == 0:
        continue

    # =====================================
    # Error
    # =====================================

    abs_error = np.abs(
        depth - gt_depth
    )

    mae = float(
        np.mean(
            abs_error[
                eval_mask
            ]
        )
    )

    mae_list.append(
        mae
    )

    # =====================================
    # Pixel CSV
    # =====================================

    valid_distance = gt_depth[
        eval_mask
    ].flatten()

    valid_error = abs_error[
        eval_mask
    ].flatten()

    for d, e in zip(
        valid_distance,
        valid_error
    ):

        pixel_results.append({

            "Image": filename,
            "Distance(m)": float(d),
            "Error(m)": float(e)

        })

    # =====================================
    # FPS
    # =====================================

    process_time = time.time() - start

    fps = 1 / max(
        process_time,
        1e-6
    )

    results.append({

        "Image": filename,
        "MAE": float(mae),
        "Time(sec)": float(process_time),
        "FPS": float(fps)

    })

    # =====================================
    # Five-panel visualization
    # =====================================

    # This map is visualization-only. The MAE and pixel statistics above keep
    # using the original, unclipped abs_error values on the unchanged eval_mask.
    error_map = np.zeros_like(abs_error, dtype=np.float32)
    error_map[eval_mask] = np.abs(depth[eval_mask] - gt_depth[eval_mask])
    masked_error_map = np.ma.masked_where(~eval_mask, error_map)
    error_cmap = plt.get_cmap("inferno").copy()
    error_cmap.set_bad(color="black")

    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    axes[0].imshow(imgL)
    axes[0].set_title("Left Image")

    disparity_plot = axes[1].imshow(disparity, cmap="jet")
    axes[1].set_title("RAFT Disparity")
    fig.colorbar(disparity_plot, ax=axes[1], label="Disparity (px)")

    predicted_depth_plot = axes[2].imshow(depth, cmap="viridis", vmin=0, vmax=80)
    axes[2].set_title("Predicted Depth")
    fig.colorbar(predicted_depth_plot, ax=axes[2], label="Depth (m)")

    gt_depth_plot = axes[3].imshow(gt_depth, cmap="viridis", vmin=0, vmax=80)
    axes[3].set_title("Ground Truth Depth")
    fig.colorbar(gt_depth_plot, ax=axes[3], label="Depth (m)")

    error_plot = axes[4].imshow(
        masked_error_map,
        cmap=error_cmap,
        vmin=0,
        vmax=2,
    )
    axes[4].set_title("Absolute Depth Error")
    fig.colorbar(error_plot, ax=axes[4], label="Error (m)")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{filename}\n"
        f"MAE={mae:.3f} m | FPS={fps:.2f} | Time={process_time:.3f} s"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(VISUALIZATION_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pbar.set_postfix(

        MAE=f"{mae:.3f}",
        FPS=f"{fps:.2f}"

    )


# =========================================
# 輸出結果
# =========================================

results_df = pd.DataFrame(
    results
)

required_result_columns = {"Image", "MAE", "Time(sec)", "FPS"}
missing_result_columns = required_result_columns.difference(results_df.columns)
if results_df.empty or missing_result_columns:
    print(
        "ERROR: Inference produced no complete result rows; "
        f"missing result columns: {sorted(missing_result_columns)}. "
        "Exiting without creating CSV files or computing a summary."
    )
    raise SystemExit(0)

results_csv = os.path.join(

    OUTPUT_DIR,

    "results.csv"

)

results_df.to_csv(

    results_csv,

    index=False,

    encoding="utf-8-sig"

)

print(
    f"\nresults.csv Saved"
)

# =========================================
# Pixel Results
# =========================================

pixel_df = pd.DataFrame(
    pixel_results
)

pixel_csv = os.path.join(

    OUTPUT_DIR,

    "pixel_results.csv"

)

pixel_df.to_csv(

    pixel_csv,

    index=False,

    encoding="utf-8-sig"

)

print(
    f"pixel_results.csv Saved"
)

# =========================================
# Summary
# =========================================

avg_mae = float(
    np.mean(mae_list)
)

avg_fps = float(
    np.mean(
        results_df["FPS"]
    )
)

avg_time = float(
    np.mean(
        results_df["Time(sec)"]
    )
)

print("\n========== Summary ==========")

print(
    f"Average MAE : {avg_mae:.3f} m"
)

print(
    f"Average FPS : {avg_fps:.2f}"
)

print(
    f"Average Time: {avg_time:.3f} sec"
)

print(
    f"Pixel Count : {len(pixel_results):,}"
)

print("============================")
