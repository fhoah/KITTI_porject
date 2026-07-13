import os
import shutil
import random
import glob

# =========================================
# 路徑設定
# =========================================

ROOT = r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000"

LEFT = os.path.join(ROOT, "images_left")
RIGHT = os.path.join(ROOT, "images_right")
DISP = os.path.join(ROOT, "disp_labels")

OUT = os.path.join(ROOT, "dataset")

# =========================================
# 隨機種子
# =========================================

random.seed(42)

# =========================================
# 取得所有左圖
# =========================================

left_files = sorted(
    glob.glob(
        os.path.join(
            LEFT,
            "*.png"
        )
    )
)

print(f"總圖片數量: {len(left_files)}")

# =========================================
# 打亂
# =========================================

random.shuffle(left_files)

# =========================================
# 80% Train
# 20% Val
# =========================================

split_idx = int(
    len(left_files) * 0.8
)

train_files = left_files[:split_idx]
val_files = left_files[split_idx:]

# =========================================
# 建立資料夾
# =========================================

for mode in ["train", "val"]:

    for sub in ["left", "right", "disp"]:

        os.makedirs(
            os.path.join(
                OUT,
                mode,
                sub
            ),
            exist_ok=True
        )

# =========================================
# 複製函式
# =========================================

def copy_group(file_list, mode):

    success = 0
    fail = 0

    for left_path in file_list:

        name = os.path.basename(
            left_path
        )

        right_path = os.path.join(
            RIGHT,
            name
        )

        # 取得數字部分
        number = name.replace(
            ".png",
            ""
        )

        # 找對應 disparity
        disp_files = glob.glob(
            os.path.join(
                DISP,
                f"*{number}.npy"
            )
        )

        if len(disp_files) == 0:

            print(
                f"找不到對應 disparity: {number}"
            )

            fail += 1
            continue

        disp_path = disp_files[0]

        disp_name = os.path.basename(
            disp_path
        )

        # 檢查右圖存在
        if not os.path.exists(
            right_path
        ):

            print(
                f"找不到右圖: {name}"
            )

            fail += 1
            continue

        # 複製 Left
        shutil.copy2(

            left_path,

            os.path.join(
                OUT,
                mode,
                "left",
                name
            )

        )

        # 複製 Right
        shutil.copy2(

            right_path,

            os.path.join(
                OUT,
                mode,
                "right",
                name
            )

        )

        # 複製 Disp
        shutil.copy2(

            disp_path,

            os.path.join(
                OUT,
                mode,
                "disp",
                disp_name
            )

        )

        success += 1

    print(
        f"{mode} 完成 "
        f"(成功:{success}, 失敗:{fail})"
    )

# =========================================
# 開始
# =========================================

print("\n開始建立 Train Dataset...")
copy_group(
    train_files,
    "train"
)

print("\n開始建立 Validation Dataset...")
copy_group(
    val_files,
    "val"
)

# =========================================
# 統計
# =========================================

print("\n========================")
print(f"Train : {len(train_files)}")
print(f"Val   : {len(val_files)}")
print("========================")

print("\n輸出位置:")
print(OUT)