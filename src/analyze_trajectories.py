import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from src.io_validation import write_csv_atomic
else:
    from io_validation import write_csv_atomic

if __package__:
    from src.pipeline_utils import atomic_output_path
else:
    from pipeline_utils import atomic_output_path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "trajectories" / "tracks.csv"
OUT = ROOT / "outputs" / "trajectories" / "advanced"


def load_tracks(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Tracking CSV is empty: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Tracking CSV has no readable header: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed tracking CSV: {path}") from exc

    required = ["frame", "track_id", "center_x", "center_y"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for column in required:
        values = pd.to_numeric(df[column], errors="coerce")
        array = values.to_numpy(dtype=float, na_value=np.nan)

        if not np.isfinite(array).all():
            raise ValueError(
                f"Column '{column}' contains missing, non-numeric, "
                "or non-finite values."
            )

        df[column] = values

    for column in ("frame", "track_id"):
        values = df[column].to_numpy(dtype=float)

        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Column '{column}' must contain whole numbers.")

        df[column] = df[column].astype("int64")

    if (df["frame"] < 1).any():
        raise ValueError("Frame numbers must be >= 1.")

    if (df["track_id"] < 0).any():
        raise ValueError("Track IDs must be >= 0.")

    if df.duplicated(subset=["track_id", "frame"]).any():
        raise ValueError(
            "Duplicate observations found for the same track and frame."
        )

    df = df.rename(columns={"center_x": "x", "center_y": "y"})
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


def calculate_metrics(df):
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Trajectory DataFrame missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Cannot calculate metrics for an empty DataFrame.")

    data = df.copy()

    for column in ("frame", "track_id", "x", "y"):
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Column '{column}' contains non-finite or non-numeric values."
            )
        data[column] = values

    for column in ("frame", "track_id"):
        values = data[column].to_numpy(dtype=float)

        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Column '{column}' must contain whole numbers.")

        if ((values < -(2**63)) | (values >= 2**63)).any():
            raise ValueError(
                f"Column '{column}' exceeds the supported int64 range."
            )

    if (data["frame"] < 1).any():
        raise ValueError("Frame numbers must be >= 1.")

    if (data["track_id"] < 0).any():
        raise ValueError("Track IDs must be >= 0.")

    data["frame"] = data["frame"].astype("int64")
    data["track_id"] = data["track_id"].astype("int64")

    if data.duplicated(subset=["track_id", "frame"]).any():
        raise ValueError("Duplicate observations found for the same track and frame.")

    rows = []

    for track_id, track in data.groupby("track_id"):
        track = track.sort_values("frame")
        frames = track["frame"].to_numpy()
        x = track["x"].to_numpy(dtype=float)
        y = track["y"].to_numpy(dtype=float)

        frame_gaps = np.diff(frames)
        distances = np.hypot(np.diff(x), np.diff(y))

        # Average speed per observed interval accounts for missing frames.
        interval_speeds = distances / frame_gaps if len(frame_gaps) else np.array([])
        path_length = float(distances.sum())
        net_displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))

        rows.append({
            "track_id": int(track_id),
            "observations": int(len(track)),
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "span_frames": int(frames[-1] - frames[0] + 1),
            "fragmentation_gaps": int(np.sum(frame_gaps > 1)),
            "missing_frames": int(np.maximum(frame_gaps - 1, 0).sum()),
            "path_length_px": path_length,
            "net_displacement_px": net_displacement,
            "mean_speed_px_per_frame": (
                float(interval_speeds.mean()) if len(interval_speeds) else 0.0
            ),
            "max_speed_px_per_frame": (
                float(interval_speeds.max()) if len(interval_speeds) else 0.0
            ),
            "straightness": (
                net_displacement / path_length if path_length > 0 else 0.0
            ),
        })

    return pd.DataFrame(rows)


def _save_plot(output_path, plot_function):
    with atomic_output_path(output_path) as temporary_path:
        plot_function(temporary_path)


def plot_speed_distribution(metrics, output_path=None):
    output_path = output_path or OUT / "speed_distribution.png"

    def draw(path):
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            ax.hist(
                metrics["mean_speed_px_per_frame"],
                bins=30,
                edgecolor="black",
            )
            ax.set_xlabel("Mean speed (pixels/frame)")
            ax.set_ylabel("Number of tracks")
            ax.set_title("Track Speed Distribution")
            fig.tight_layout()
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)

    _save_plot(output_path, draw)


def plot_straightness(metrics, output_path=None):
    output_path = output_path or OUT / "trajectory_straightness.png"

    def draw(path):
        fig, ax = plt.subplots(figsize=(9, 5))
        try:
            ax.hist(metrics["straightness"], bins=25, edgecolor="black")
            ax.set_xlabel("Trajectory straightness (0 = winding, 1 = straight)")
            ax.set_ylabel("Number of tracks")
            ax.set_title("Trajectory Straightness Distribution")
            fig.tight_layout()
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)

    _save_plot(output_path, draw)


def plot_occupancy(df, output_path=None):
    output_path = output_path or OUT / "occupancy_map.png"

    def draw(path):
        fig, ax = plt.subplots(figsize=(9, 6))
        try:
            histogram = ax.hist2d(df["x"], df["y"], bins=40, cmap="hot")
            fig.colorbar(histogram[3], ax=ax, label="Observations")
            ax.set_xlabel("Image X (pixels)")
            ax.set_ylabel("Image Y (pixels)")
            ax.set_title("Object Movement Occupancy Map")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)

    _save_plot(output_path, draw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    args = parser.parse_args()

    df = load_tracks(args.input)
    metrics = calculate_metrics(df)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(metrics, OUT / "trajectory_metrics.csv")

    plot_speed_distribution(metrics)
    plot_straightness(metrics)
    plot_occupancy(df)

    print(f"Observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Track metrics saved: {OUT / 'trajectory_metrics.csv'}")
    print(f"Plots saved under: {OUT}")


if __name__ == "__main__":
    main()