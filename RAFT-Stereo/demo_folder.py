import sys
sys.path.append("core")

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

LEFT_DIR = r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000\images_left"
RIGHT_DIR = r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000\images_right"
GT_DIR = r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000\depth_labels"

OUTPUT_DIR = r"C:\Users\user\Desktop\KITTI_porject\RAFT_output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
        r"C:\Users\user\Desktop\KITTI_porject\RAFT-Stereo\checkpoints\mykitti_fixed.pth"
    )
)

model = model.module
model.cuda()
model.eval()

print("Model Loaded")


# =========================================
# 找圖
# =========================================

left_images = sorted(
    glob.glob(
        os.path.join(
            LEFT_DIR,
            "*.png"
        )
    )
)

gt_files = sorted(
    glob.glob(
        os.path.join(
            GT_DIR,
            "*.npy"
        )
    )
)

MAX_IMAGES = 5

left_images = left_images[:MAX_IMAGES]
gt_files = gt_files[:MAX_IMAGES]

print(
    f"Total Images: {len(left_images)}"
)


# =========================================
# 統計變數
# =========================================

results = []

pixel_results = []

mae_list = []


# =========================================
# 開始
# =========================================

pbar = tqdm(
    enumerate(left_images),
    total=len(left_images),
    desc="RAFT Processing",
    unit="img"
)

for idx, left_path in pbar:

    start = time.time()

    filename = os.path.basename(
        left_path
    )

    right_path = os.path.join(
        RIGHT_DIR,
        filename
    )

    if not os.path.exists(
        right_path
    ):
        continue

    gt_path = gt_files[idx]

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

    disp = flow_up.squeeze().cpu().numpy()

    plt.figure(figsize=(12,4))
    plt.imshow(-disp, cmap='jet')
    plt.colorbar()
    plt.title("Predicted Disparity")
    plt.savefig("disp_test.png", bbox_inches='tight')
    plt.close()
    print("disp min =", disp.min())
    print("disp max =", disp.max())
    print("disp mean =", disp.mean())
    print("已儲存 disp_test.png")
    disparity = flow_up.squeeze().cpu().numpy()

    disparity = disparity[:h, :w]

    # =====================================
    # Depth
    # =====================================

    focal = 721.53
    baseline = 0.54

    depth = np.zeros_like(
        disparity
    )

    mask = disparity < 0

    depth[mask] = (

        focal * baseline

    ) / np.abs(

        disparity[mask]

    )

    depth = np.clip(
        depth,
        0,
        80
    )

    # =====================================
    # GT
    # =====================================

    gt_depth = np.load(
        gt_path
    )[:h, :w]

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