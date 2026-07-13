import os
import glob
import numpy as np
from tqdm import tqdm

# ==========================
# 路徑
# ==========================

DEPTH_DIR = r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000\depth_labels"

DISP_DIR = r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000\disp_labels"

os.makedirs(
    DISP_DIR,
    exist_ok=True
)

# ==========================
# KITTI 參數
# ==========================

FOCAL = 721.53
BASELINE = 0.54

# ==========================
# 找檔案
# ==========================

depth_files = sorted(

    glob.glob(
        os.path.join(
            DEPTH_DIR,
            "*.npy"
        )
    )

)

print(
    f"找到 {len(depth_files)} 個檔案"
)

# ==========================
# 開始轉換
# ==========================

for depth_file in tqdm(depth_files):

    depth = np.load(
        depth_file
    ).astype(np.float32)

    disparity = np.zeros_like(
        depth,
        dtype=np.float32
    )

    valid = depth > 0

    disparity[valid] = (

        FOCAL * BASELINE

    ) / depth[valid]

    save_path = os.path.join(

        DISP_DIR,

        os.path.basename(
            depth_file
        )

    )

    np.save(
        save_path,
        disparity
    )

print("\n轉換完成")