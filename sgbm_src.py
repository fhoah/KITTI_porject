#環境初始化：檢查並安裝必要函式庫
import subprocess
import sys


def install_packages():
    required = {"opencv-python", "numpy", "pandas", "matplotlib", "seaborn", "ultralytics"}
    # 這裡檢查已安裝的套件 (排除 opencv-python 的變體名稱)
    installed = {pkg.split('==')[0].lower() for pkg in
                 subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split()}

    for pkg in required:
        pkg_name = "cv2" if pkg == "opencv-python" else pkg
        try:
            __import__(pkg_name if pkg_name != "opencv-python" else "cv2")
        except ImportError:
            print(f"正在安裝 {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


install_packages()
#載入函式庫與基本路徑定義
import os
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 資料根目錄設定
BASE_RGB_DIR = "/content/drive/MyDrive/KITTI_Dataset/raw_data/raw_data_downloader"
GT_DIR = "/content/drive/MyDrive/KITTI_Dataset/processed_npy_stereo"

# 檢查資料夾是否存在
if not os.path.exists(BASE_RGB_DIR):
    print("⚠️ 警告：找不到影像根目錄，請檢查路徑設定。")
else:
    print("✅ 路徑配置完成。")
#相機參數與 SGBM 演算法設定
# 相機內參 (KITTI 預設)
focal_length = 721.53
baseline = 0.54

# SGBM 共通硬體參數 (調整此處可改變所有模式的基礎表現)
params = dict(
    minDisparity=0,
    numDisparities=16 * 6, # 視差搜尋範圍
    blockSize=5,           # 匹配塊大小
    P1=8 * 3 * 5 ** 2,     # 平滑度參數 1
    P2=32 * 3 * 5 ** 2,    # 平滑度參數 2
    disp12MaxDiff=1,
    uniquenessRatio=10
)


# 定義要進行比較的模式名稱與 OpenCV 代碼
modes = {
    "SGBM_5W": cv2.STEREO_SGBM_MODE_SGBM,
    "SGBM_3W": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "HH_8W": cv2.STEREO_SGBM_MODE_HH,
    "HH4_4W": cv2.STEREO_SGBM_MODE_HH4
}

print("✅ 演算法參數已設定。")


# 定義核心計算與誤差分析函數
def calculate_depth_and_mae(img_l, img_r, gt_depth, mode_val):
    """
    計算單張影像在指定模式下的 MAE 與執行時間
    """
    # 建立 SGBM 實例
    stereo = cv2.StereoSGBM_create(**params, mode=mode_val)

    # 計時並運算視差
    start_time = time.time()
    disp = stereo.compute(img_l, img_r).astype(np.float32) / 16.0
    elapsed_time = time.time() - start_time

    # 換算深度 Z = (f * B) / d
    depth = np.zeros_like(disp)
    mask = disp > 0
    depth[mask] = (focal_length * baseline) / disp[mask]

    # 計算 20m 內 MAE
    eval_mask = (gt_depth > 0) & (gt_depth <= 20) & (depth > 0)
    mae = np.mean(np.abs(depth[eval_mask] - gt_depth[eval_mask])) if np.any(eval_mask) else np.nan

    return mae, elapsed_time


# 執行 1000 組批次迴圈
detailed_results = []
total_found = 0
target_count = 1000

# 遍歷邏輯：日期 -> 序列 -> 影像
date_folders = sorted([d for d in os.listdir(BASE_RGB_DIR) if os.path.isdir(os.path.join(BASE_RGB_DIR, d))])

for date_dir in date_folders:
    if total_found >= target_count: break
    date_path = os.path.join(BASE_RGB_DIR, date_dir)
    drive_folders = sorted([d for d in os.listdir(date_path) if d.endswith('_sync')])

    for drive_dir in drive_folders:
        if total_found >= target_count: break
        l_img_dir = os.path.join(date_path, drive_dir, "image_02", "data")
        r_img_dir = os.path.join(date_path, drive_dir, "image_03", "data")

        if not os.path.exists(l_img_dir): continue
        img_files = sorted([f for f in os.listdir(l_img_dir) if f.endswith('.png')])

        for filename in img_files:
            if total_found >= target_count: break

            # 檔名對齊與檢查真值
            gt_name = f"{drive_dir}_image_02_{filename.replace('.png', '.npy')}"
            gt_path = os.path.join(GT_DIR, gt_name)
            if not os.path.exists(gt_path): continue

            # 讀取並分析
            img_l = cv2.imread(os.path.join(l_img_dir, filename), cv2.IMREAD_GRAYSCALE)
            img_r = cv2.imread(os.path.join(r_img_dir, filename), cv2.IMREAD_GRAYSCALE)
            gt_depth = np.load(gt_path)

            for m_name, m_val in modes.items():
                mae, sec = calculate_depth_and_mae(img_l, img_r, gt_depth, m_val)
                detailed_results.append({"Mode": m_name, "MAE": mae, "Time_Sec": sec})

            total_found += 1
            if total_found % 50 == 0: print(f"进度：{total_found}/{target_count}...")

print(f"🏁 數據處理完畢，共獲取 {total_found} 組樣本。")


#統計報告與結果視覺化
# 將資料轉為 DataFrame
df = pd.DataFrame(detailed_results)

if not df.empty:
    # 1. 數值統計表格
    report = df.groupby("Mode").mean(numeric_only=True)
    print("\n📝 深度研究報告摘要 (20m 內指標)：")
    print(report.to_string())

    # 2. 視覺化比較圖表
    plt.figure(figsize=(16, 6))

    # 左圖：精準度 (MAE) 箱型圖
    plt.subplot(1, 2, 1)
    sns.boxplot(x="Mode", y="MAE", data=df, palette="husl")
    plt.title("Error Comparison (MAE)", fontsize=14)
    plt.ylabel("Absolute Error (Meters)")
    plt.grid(axis='y', alpha=0.3)

    # 右圖：效率 (Time) 長條圖
    plt.subplot(1, 2, 2)
    sns.barplot(x="Mode", y="Time_Sec", data=df, palette="husl", errorbar=None)
    plt.title("Efficiency Comparison (Avg Time)", fontsize=14)
    plt.ylabel("Execution Time (Seconds)")
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 3. FPS 計算
    print("\n📊 處理能力分析 (FPS):")
    for mode in modes.keys():
        avg_t = df[df["Mode"] == mode]["Time_Sec"].mean()
        print(f"- {mode}: 每秒可處理 {1/avg_t:.2f} 幀 (FPS)")
else:
    print("❌ 無數據可供統計。")