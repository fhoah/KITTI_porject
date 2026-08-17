import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "RAFT_output"))
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
PIXEL_CSV = os.path.join(DATA_DIR, "pixel_results.csv")

RESULTS_COLUMNS = {"Image", "MAE", "Time(sec)", "FPS"}
PIXEL_COLUMNS = {"Image", "Distance(m)", "Error(m)"}


def load_csv(path, required_columns):
    if not os.path.isfile(path):
        print(f"ERROR: CSV file does not exist: {path}")
        return None

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: Could not read CSV file {path}: {exc}")
        return None

    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        print(f"ERROR: {path} is missing required columns: {', '.join(missing)}")
        return None
    return dataframe


def filter_pixel_rows(pixel_df):
    filtered = pixel_df.copy()
    filtered["Distance(m)"] = pd.to_numeric(filtered["Distance(m)"], errors="coerce")
    filtered["Error(m)"] = pd.to_numeric(filtered["Error(m)"], errors="coerce")

    distance = filtered["Distance(m)"].to_numpy(dtype=np.float64)
    error = filtered["Error(m)"].to_numpy(dtype=np.float64)
    valid = (
        np.isfinite(distance)
        & np.isfinite(error)
        & (distance > 0.0)
        & (distance <= 80.0)
        & (error >= 0.0)
    )
    invalid_count = int((~valid).sum())
    print(f"Invalid pixel rows removed: {invalid_count:,}")
    return filtered.loc[valid].copy()


def calculate_distance_statistics(pixel_df, bin_width):
    distances = pixel_df["Distance(m)"].to_numpy(dtype=np.float64)
    errors = pixel_df["Error(m)"].to_numpy(dtype=np.float64)
    rows = []

    for start in range(0, 80, bin_width):
        end = start + bin_width
        if end == 80:
            mask = (distances >= start) & (distances <= end)
        else:
            mask = (distances >= start) & (distances < end)

        bin_errors = errors[mask]
        row = {
            "distance_start_m": start,
            "distance_end_m": end,
            "pixel_count": int(bin_errors.size),
            "mean_error_m": np.nan,
            "median_error_m": np.nan,
            "rmse_m": np.nan,
            "std_error_m": np.nan,
        }
        if bin_errors.size:
            row.update(
                mean_error_m=float(np.mean(bin_errors)),
                median_error_m=float(np.median(bin_errors)),
                rmse_m=float(np.sqrt(np.mean(np.square(bin_errors)))),
                std_error_m=float(np.std(bin_errors, ddof=0)),
            )
        rows.append(row)

    return pd.DataFrame(rows)


def save_figure(filename):
    path = os.path.join(DATA_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_mae_summary(results_df, average_image_mae):
    mae_values = pd.to_numeric(results_df["MAE"], errors="coerce").to_numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(mae_values) + 1), mae_values, marker="o", linewidth=3)
    plt.axhline(
        y=average_image_mae,
        color="red",
        linestyle="--",
        label=f"Average MAE = {average_image_mae:.3f}",
    )
    plt.xlabel("Image Number")
    plt.ylabel("MAE (m)")
    plt.title("MAE Trend Across Images")
    plt.grid(alpha=0.3)
    plt.legend()
    return save_figure("MAE_summary.png")


def plot_pixel_error(pixel_df):
    distances = pixel_df["Distance(m)"].to_numpy(dtype=np.float64)
    errors = pixel_df["Error(m)"].to_numpy(dtype=np.float64)
    plt.figure(figsize=(12, 6))
    points = plt.scatter(distances, errors, s=0.01, alpha=0.03, c=errors, cmap="jet")
    plt.colorbar(points, label="Pixel Error (m)")
    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("Absolute Error (m)")
    plt.title(f"Pixel-wise Error vs Distance\nTotal Pixels = {len(errors):,}")
    plt.grid(alpha=0.3)
    return save_figure("Pixel_Error_vs_Distance.png")


def plot_error_vs_distance(stats_5m):
    populated = stats_5m[stats_5m["pixel_count"] > 0]
    centers = (populated["distance_start_m"] + populated["distance_end_m"]) / 2
    plt.figure(figsize=(12, 6))
    plt.plot(centers, populated["mean_error_m"], marker="o", linewidth=3)
    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("Average Error (m)")
    plt.title("Average Error vs Distance")
    plt.grid(alpha=0.3)
    return save_figure("Error_vs_Distance.png")


def plot_histogram(pixel_df):
    plt.figure(figsize=(12, 6))
    plt.hist(pixel_df["Error(m)"].to_numpy(dtype=np.float64), bins=100)
    plt.xlabel("Pixel Error (m)")
    plt.ylabel("Count")
    plt.title("Pixel Error Distribution")
    plt.grid(alpha=0.3)
    return save_figure("Histogram_Error.png")


def plot_one_meter_mae(stats_1m):
    populated = stats_1m[stats_1m["pixel_count"] > 0]
    centers = (populated["distance_start_m"] + populated["distance_end_m"]) / 2
    plt.figure(figsize=(12, 6))
    plt.plot(centers, populated["mean_error_m"], linewidth=2, label="RAFT Stereo")
    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("Mean Absolute Error (m)")
    plt.title("RAFT Stereo MAE vs Distance")
    plt.grid(alpha=0.3)
    plt.legend()
    return save_figure("MAE_vs_Distance.png")


def plot_binned_metric(stats_5m, column, ylabel, title, filename):
    labels = [
        f"{int(start)}-{int(end)}"
        for start, end in zip(stats_5m["distance_start_m"], stats_5m["distance_end_m"])
    ]
    values = stats_5m[column].to_numpy(dtype=np.float64)
    x = np.arange(len(labels))
    plt.figure(figsize=(14, 6))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Ground Truth Distance Bin (m)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    return save_figure(filename)


def print_distance_summary(stats_5m):
    print("\n========== Error by Distance ==========\n")
    for row in stats_5m.itertuples(index=False):
        print(f"{row.distance_start_m}-{row.distance_end_m} m:")
        print(f"  Pixels       : {row.pixel_count:,}")
        if row.pixel_count:
            print(f"  MAE          : {row.mean_error_m:.3f} m")
            print(f"  Median Error : {row.median_error_m:.3f} m")
            print(f"  RMSE         : {row.rmse_m:.3f} m")
        else:
            print("  MAE          : N/A")
            print("  Median Error : N/A")
            print("  RMSE         : N/A")
        print()


def finite_mean(series):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def main():
    print("Loading CSV files...")
    results_df = load_csv(RESULTS_CSV, RESULTS_COLUMNS)
    pixel_df = load_csv(PIXEL_CSV, PIXEL_COLUMNS)
    if results_df is None or pixel_df is None:
        return 1

    pixel_df = filter_pixel_rows(pixel_df)
    if pixel_df.empty:
        print("ERROR: No valid pixel rows remain after filtering; no analysis was generated.")
        return 1

    stats_5m = calculate_distance_statistics(pixel_df, 5)
    stats_1m = calculate_distance_statistics(pixel_df, 1)
    stats_5m_path = os.path.join(DATA_DIR, "distance_error_5m.csv")
    stats_1m_path = os.path.join(DATA_DIR, "distance_error_1m.csv")
    stats_5m.to_csv(stats_5m_path, index=False)
    stats_1m.to_csv(stats_1m_path, index=False)

    errors = pixel_df["Error(m)"].to_numpy(dtype=np.float64)
    average_image_mae = finite_mean(results_df["MAE"])
    overall = pd.DataFrame(
        [{
            "image_count": int(results_df["Image"].nunique()),
            "pixel_count": int(len(pixel_df)),
            "average_image_mae_m": average_image_mae,
            "pixel_mae_m": float(np.mean(errors)),
            "pixel_median_error_m": float(np.median(errors)),
            "pixel_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "pixel_std_error_m": float(np.std(errors, ddof=0)),
            "average_fps": finite_mean(results_df["FPS"]),
            "average_time_sec": finite_mean(results_df["Time(sec)"]),
        }]
    )
    overall_path = os.path.join(DATA_DIR, "overall_metrics.csv")
    overall.to_csv(overall_path, index=False)

    generated_files = [
        plot_mae_summary(results_df, average_image_mae),
        plot_pixel_error(pixel_df),
        plot_error_vs_distance(stats_5m),
        plot_histogram(pixel_df),
        plot_one_meter_mae(stats_1m),
        plot_binned_metric(
            stats_5m, "mean_error_m", "MAE (m)",
            "MAE by Ground Truth Distance", "distance_error_5m.png",
        ),
        plot_binned_metric(
            stats_5m, "rmse_m", "RMSE (m)",
            "RMSE by Ground Truth Distance", "distance_rmse_5m.png",
        ),
        stats_5m_path,
        stats_1m_path,
        overall_path,
    ]

    print_distance_summary(stats_5m)
    print("========== Overall Summary ==========")
    for column, value in overall.iloc[0].items():
        print(f"{column}: {value}")

    print("\nGenerated files:")
    for path in generated_files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
