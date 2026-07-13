import numpy as np
import glob

files = sorted(glob.glob(
    r"C:\Users\user\Desktop\KITTI_porject\Train_Data_1000\disp_labels\*.npy"
))

disp = np.load(files[0])

print("檔案:", files[0])
print("shape =", disp.shape)
print("dtype =", disp.dtype)
print("min =", disp.min())
print("max =", disp.max())
print("mean =", disp.mean())