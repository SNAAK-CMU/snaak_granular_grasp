import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

data_dir = "/home/parth/snaak/snaak_data/data_parth/"


def save_index_to_csv(
    index: dict, out_path: str = "depth_vs_grapsedweigth.csv"
) -> None:
    """
    Convert the index dictionary into a pandas DataFrame and save to CSV.

    The resulting CSV contains columns: key, start_weight, final_weight,
    z_below_surface, weight_grasped.
    """
    rows = []
    for key, val in index.items():
        rows.append(
            {
                "key": key,
                "start_weight": val.get("start_weight"),
                "final_weight": val.get("final_weight"),
                "z_below_surface": val.get("z_below_surface"),
                "weight_grasped": val.get("weight_grasped"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved CSV: {out_path}")


def build_npz_index(base_dir: str) -> dict:
    """
    Scan all subdirectories under base_dir whose name starts with "10_05",
    read every .npz file, and build a dictionary:

    key: "<full_session_dir>_<npz_stem>"
    value: {
        "start_weight": float,
        "final_weight": float,
        "z_below_surface": float,
    }
    """
    index: dict[str, dict] = {}
    for root, dirs, files in os.walk(base_dir):
        # Only consider session folders starting with the substring "10_15"
        if not os.path.basename(root).startswith("10_15"):
            continue
        for fname in files:
            if not fname.endswith(".npz"):
                continue
            fpath = os.path.join(root, fname)
            try:
                data = np.load(fpath, allow_pickle=True)
                # Extract required fields; coerce to float when possible
                start_weight = data.get("start_weight")
                final_weight = data.get("final_weight")
                z_below_surface = data.get("z_below_surface")

                def to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                sw = to_float(start_weight)
                fw = to_float(final_weight)
                entry = {
                    "start_weight": sw,
                    "final_weight": fw,
                    "z_below_surface": to_float(z_below_surface),
                    "weight_grasped": sw - fw,
                }

                stem = os.path.splitext(fname)[0]
                padded = str(stem).zfill(3)
                subdir = os.path.basename(root)
                key = f"{subdir}_{padded}"
                index[key] = entry
            except Exception as e:
                # Skip problematic files but continue scanning
                print(f"Failed to load {fpath}: {e}")
    return index


def visualize_individual(grasp_depth, plot_start: bool = True) -> None:
    """
    Visualize the individual items in the index.
    """
    csv_path = "depth_vs_grapsedweigth.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}. Generate it first.")
        return

    df = pd.read_csv(csv_path)
    if "z_below_surface" not in df.columns:
        print("CSV missing required column 'z_below_surface'.")
        return

    depth_val = grasp_depth

    mask = np.isclose(df["z_below_surface"].astype(float), depth_val, atol=1e-9)
    dff = df[mask].copy()
    if dff.empty:
        print(f"No rows found for z_below_surface == {depth_val}")
        return

    # Extract sequence id from the tail of the key: '<full_session_dir>_<npz_stem>'
    def extract_seq(key: str) -> int:
        tail = os.path.basename(key).split("_")[-1]
        try:
            return int(tail)
        except Exception:
            # Fallback: try to parse last token after underscore in whole key
            try:
                return int(key.rsplit("_", 1)[-1])
            except Exception:
                return 0

    dff["seq"] = dff["key"].astype(str).apply(extract_seq)
    dff = dff.sort_values(["key", "seq"]).reset_index(drop=True)

    x = np.arange(1, len(dff) + 1)
    y_start = dff["start_weight"].astype(float).values
    y_grasped = dff["weight_grasped"].astype(float).values

    plt.figure(figsize=(8, 5))
    if plot_start:
        plt.plot(x, y_start, marker="o", label="start_weight")
    plt.plot(x, y_grasped, marker="s", label="weight_grasped")
    plt.xticks(x, [str(i) for i in dff["seq"].tolist()])
    plt.xlabel("Sequence (npz order)")
    plt.ylabel("Weight (g)")
    plt.title(f"Depth {depth_val}: Start vs Grasped")
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(bottom=0)
    ax.axhline(10, color="red", linestyle="--", linewidth=1)
    ax.grid(True, which="major", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_combined(depths: list[float]) -> None:
    """
    Visualize the combined data for multiple depths.
    """
    csv_path = "depth_vs_grapsedweigth.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}. Generate it first.")
        return

    df = pd.read_csv(csv_path)
    required_cols = {"z_below_surface", "weight_grasped"}
    if not required_cols.issubset(df.columns):
        print(f"CSV missing required columns: {required_cols}")
        return

    data = []
    labels = []
    for d in depths:
        mask = np.isclose(df["z_below_surface"].astype(float), float(d), atol=1e-9)
        vals = df.loc[mask, "weight_grasped"].dropna().astype(float).values
        if vals.size == 0:
            continue
        data.append(vals)
        labels.append(str(d))

    if not data:
        print("No matching data for provided depths.")
        return

    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    # Styling
    for box in bp["boxes"]:
        box.set(facecolor="#cfe2ff", edgecolor="#1f77b4")
    for whisker in bp["whiskers"]:
        whisker.set(color="#1f77b4")
    for cap in bp["caps"]:
        cap.set(color="#1f77b4")
    for median in bp["medians"]:
        median.set(color="#d62728", linewidth=2)

    plt.xlabel("Grasp depth (m)")
    plt.ylabel("Weight grasped (g)")
    plt.title("Weight grasped distribution by grasp depth")

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylim(bottom=0)
    ax.axhline(10, color="red", linestyle="--", linewidth=1)
    ax.grid(True, which="major", axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    index = build_npz_index(data_dir)
    print(f"Collected {len(index)} items from sessions starting with '10_05'")

    # Save to CSV
    save_index_to_csv(index)

    # visualize_individual(0.02, plot_start=False)
    # visualize_individual(0.03, plot_start=False)
    # visualize_individual(0.04, plot_start=False)
    # visualize_individual(0.05, plot_start=True)

    visualize_combined([0.02, 0.03, 0.04, 0.05])


if __name__ == "__main__":
    main()
