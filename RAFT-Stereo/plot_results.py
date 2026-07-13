import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================
# 路徑
# =========================================

DATA_DIR = r"C:\Users\user\Desktop\KITTI_porject\RAFT_output"

RESULTS_CSV = os.path.join(
    DATA_DIR,
    "results.csv"
)

PIXEL_CSV = os.path.join(
    DATA_DIR,
    "pixel_results.csv"
)


# =========================================
# 讀CSV
# =========================================

print("Loading CSV...")

results_df = pd.read_csv(
    RESULTS_CSV
)

pixel_df = pd.read_csv(
    PIXEL_CSV
)

print("CSV Loaded")


# =========================================
# MAE Summary
# =========================================

print("Generating MAE Summary...")

mae_values = results_df["MAE"].values

avg_mae = np.mean(
    mae_values
)

plt.figure(
    figsize=(12,6)
)

plt.plot(

    range(
        1,
        len(mae_values)+1
    ),

    mae_values,

    marker='o',
    linewidth=3

)

plt.axhline(

    y=float(avg_mae),

    color='red',
    linestyle='--',

    label=f'Average MAE = {avg_mae:.3f}'

)

plt.xlabel(
    "Image Number"
)

plt.ylabel(
    "MAE (m)"
)

plt.title(
    "MAE Trend Across Images"
)

plt.grid(alpha=0.3)

plt.legend()

plt.savefig(

    os.path.join(
        DATA_DIR,
        "MAE_summary.png"
    ),

    dpi=300,
    bbox_inches='tight'

)

plt.close()

print("MAE Summary Done")


# =========================================
# Pixel Scatter
# =========================================

print("Generating Pixel Scatter...")

distances = pixel_df[
    "Distance(m)"
].values

errors = pixel_df[
    "Error(m)"
].values

plt.figure(
    figsize=(12,6)
)

plt.scatter(

    distances,
    errors,

    s=0.01,
    alpha=0.03,

    c=errors,

    cmap="jet"

)

plt.colorbar(
    label="Pixel Error (m)"
)

plt.xlabel(
    "Ground Truth Distance (m)"
)

plt.ylabel(
    "Absolute Error (m)"
)

plt.title(
    f"Pixel-wise Error vs Distance\nTotal Pixels = {len(errors):,}"
)

plt.grid(alpha=0.3)

plt.savefig(

    os.path.join(
        DATA_DIR,
        "Pixel_Error_vs_Distance.png"
    ),

    dpi=300,
    bbox_inches='tight'

)

plt.close()

print("Pixel Scatter Done")


# =========================================
# Distance Bin Average Error
# =========================================

print("Generating Error vs Distance...")

bins = np.arange(
    0,
    85,
    5
)

centers = []
mean_errors = []

for i in range(
    len(bins)-1
):

    mask = (

        (distances >= bins[i])
        &
        (distances < bins[i+1])

    )

    if np.sum(mask) > 0:

        centers.append(

            (bins[i]+bins[i+1])/2

        )

        mean_errors.append(

            np.mean(
                errors[mask]
            )

        )

plt.figure(
    figsize=(12,6)
)

plt.plot(

    centers,
    mean_errors,

    marker='o',
    linewidth=3

)

plt.xlabel(
    "Distance (m)"
)

plt.ylabel(
    "Average Error (m)"
)

plt.title(
    "Average Error vs Distance"
)

plt.grid(alpha=0.3)

plt.savefig(

    os.path.join(
        DATA_DIR,
        "Error_vs_Distance.png"
    ),

    dpi=300,
    bbox_inches='tight'

)

plt.close()

print("Error vs Distance Done")


# =========================================
# Histogram
# =========================================

print("Generating Histogram...")

plt.figure(
    figsize=(12,6)
)

plt.hist(

    errors,

    bins=100

)

plt.xlabel(
    "Pixel Error (m)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "Pixel Error Distribution"
)

plt.grid(alpha=0.3)

plt.savefig(

    os.path.join(
        DATA_DIR,
        "Histogram_Error.png"
    ),

    dpi=300,
    bbox_inches='tight'

)

plt.close()

print("Histogram Done")


# =========================================
# Summary
# =========================================

print("\n========== Summary ==========")

print(
    f"Images        : {len(results_df)}"
)

print(
    f"Pixels        : {len(pixel_df):,}"
)

print(
    f"Average MAE   : {avg_mae:.3f} m"
)

print(
    f"Average Error : {errors.mean():.3f} m"
)

print(
    f"Median Error  : {np.median(errors):.3f} m"
)

print(
    f"Max Error     : {errors.max():.3f} m"
)

print("============================")

print("\nOutput Files")

print("MAE_summary.png")
print("Pixel_Error_vs_Distance.png")
print("Error_vs_Distance.png")
print("Histogram_Error.png")
# =========================================
# MAE vs Distance
# =========================================

print("Generating MAE vs Distance...")

distances = pixel_df[
    "Distance(m)"
].to_numpy(dtype=np.float64)

errors = pixel_df[
    "Error(m)"
].to_numpy(dtype=np.float64)

bins = np.arange(
    0,
    81,
    1
)

distance_centers = []
distance_mae = []

for i in range(len(bins)-1):

    mask = (

        (distances >= bins[i])
        &
        (distances < bins[i+1])

    )

    if np.sum(mask) > 0:

        distance_centers.append(

            (bins[i] + bins[i+1]) / 2

        )

        distance_mae.append(

            np.mean(
                errors[mask]
            )

        )

plt.figure(
    figsize=(12,6)
)

plt.plot(

    distance_centers,
    distance_mae,

    linewidth=2,

    label="RAFT Stereo"

)

plt.xlabel(
    "Ground Truth Distance (m)"
)

plt.ylabel(
    "MAE (m)"
)

plt.title(
    "RAFT Stereo MAE vs Distance"
)

plt.grid(alpha=0.3)

plt.legend()

plt.savefig(

    os.path.join(
        DATA_DIR,
        "MAE_vs_Distance.png"
    ),

    dpi=300,
    bbox_inches='tight'

)

plt.close()

print(
    "MAE_vs_Distance.png Done"
)