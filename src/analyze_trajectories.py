import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "trajectories" / "tracks.csv"
OUT = ROOT / "outputs" / "trajectories" / "advanced"
OUT.mkdir(parents=True, exist_ok=True)



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
            raise ValueError(
                f"Column '{column}' must contain whole numbers."
            )

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
    rows = []

    for track_id, track in df.groupby("track_id"):
        track = track.sort_values("frame")
        frames = track["frame"].to_numpy()
        x = track["x"].to_numpy(dtype=float)
        y = track["y"].to_numpy(dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.hypot(dx, dy)

        frame_gaps = np.diff(frames)
        valid_steps = frame_gaps > 0
        speeds = distances[valid_steps] / frame_gaps[valid_steps]

        rows.append({
            "track_id": track_id,
            "observations": len(track),
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "span_frames": frames[-1] - frames[0] + 1,
            "fragmentation_gaps": int(np.sum(frame_gaps > 1)),
            "missing_frames": int(np.sum(np.maximum(frame_gaps - 1, 0))),
            "path_length_px": float(distances.sum()),
            "net_displacement_px": float(
                np.hypot(x[-1] - x[0], y[-1] - y[0])
            ),
            "mean_speed_px_per_frame": float(speeds.mean())
            if len(speeds) else 0.0,
            "max_speed_px_per_frame": float(speeds.max())
            if len(speeds) else 0.0,
            "straightness": float(
                np.hypot(x[-1] - x[0], y[-1] - y[0])
                / distances.sum()
            ) if distances.sum() > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def plot_speed_distribution(metrics):
    plt.figure(figsize=(9, 5))
    plt.hist(
        metrics["mean_speed_px_per_frame"],
        bins=30,
        edgecolor="black"
    )
    plt.xlabel("Mean speed (pixels/frame)")
    plt.ylabel("Number of tracks")
    plt.title("Pedestrian Track Speed Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "speed_distribution.png", dpi=160)
    plt.close()


def plot_straightness(metrics):
    plt.figure(figsize=(9, 5))
    plt.hist(metrics["straightness"], bins=25, edgecolor="black")
    plt.xlabel("Trajectory straightness (0 = winding, 1 = straight)")
    plt.ylabel("Number of tracks")
    plt.title("Trajectory Straightness Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "trajectory_straightness.png", dpi=160)
    plt.close()


def plot_occupancy(df):
    plt.figure(figsize=(9, 6))
    plt.hist2d(
        df["x"], df["y"],
        bins=40,
        cmap="hot"
    )
    plt.colorbar(label="Observations")
    plt.xlabel("Image X (pixels)")
    plt.ylabel("Image Y (pixels)")
    plt.title("Pedestrian Movement Occupancy Map")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / "occupancy_map.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Tracks file not found: {args.input}"
        )

    df = load_tracks(args.input)
    metrics = calculate_metrics(df)

    metrics.to_csv(OUT / "trajectory_metrics.csv", index=False)

    plot_speed_distribution(metrics)
    plot_straightness(metrics)
    plot_occupancy(df)

    print(f"Observations: {len(df)}")
    print(f"Unique tracks: {df['track_id'].nunique()}")
    print(f"Track metrics saved: {OUT / 'trajectory_metrics.csv'}")
    print(f"Plots saved under: {OUT}")


if __name__ == "__main__":
    main()