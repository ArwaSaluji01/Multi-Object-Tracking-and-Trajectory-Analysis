from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from src.io_validation import read_validated_csv
else:
    from io_validation import read_validated_csv

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "trajectories"
CSV_PATH = DATA_PATH / "tracks.csv"
SUMMARY_PATH = DATA_PATH / "track_summary.csv"
PLOT_PATH = ROOT / "outputs" / "trajectories" / "plots"


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    tracks = read_validated_csv(
        CSV_PATH,
        required_columns=[
            "frame", "track_id", "class_name", "center_x", "center_y"
        ],
        numeric_columns=["frame", "track_id", "center_x", "center_y"],
        integer_columns=("frame", "track_id"),
        unique_by=("track_id", "frame"),
        description="Tracking CSV",
    )

    summary = read_validated_csv(
        SUMMARY_PATH,
        required_columns=[
            "track_id", "duration_seconds", "path_length_pixels"
        ],
        numeric_columns=[
            "track_id", "duration_seconds", "path_length_pixels"
        ],
        integer_columns=("track_id",),
        unique_by=("track_id",),
        description="Track summary CSV",
    )

    # 1. Number of unique track IDs by class
    counts = (
        tracks.groupby("class_name")["track_id"]
        .nunique()
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    counts.plot(kind="barh")
    plt.xlabel("Unique track IDs")
    plt.ylabel("Object class")
    plt.title("Tracked IDs by Object Class")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "class_counts.png", dpi=160)
    plt.close()

    # 2. Track duration distribution
    plt.figure(figsize=(9, 5))
    plt.hist(summary["duration_seconds"], bins=30)
    plt.xlabel("Observed duration (seconds)")
    plt.ylabel("Number of tracks")
    plt.title("Track Duration Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "track_durations.png", dpi=160)
    plt.close()

    # 3. Path length distribution
    plt.figure(figsize=(9, 5))
    plt.hist(summary["path_length_pixels"], bins=30)
    plt.xlabel("Path length (pixels)")
    plt.ylabel("Number of tracks")
    plt.title("Trajectory Path Length Distribution")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "path_lengths.png", dpi=160)
    plt.close()

    # 4. Trajectories for the most frequently observed tracks
    top_ids = (
        tracks.groupby("track_id")["frame"]
        .count()
        .nlargest(15)
        .index
    )

    plt.figure(figsize=(10, 7))

    for track_id in top_ids:
        track = tracks[tracks["track_id"] == track_id].sort_values("frame")
        plt.plot(
            track["center_x"],
            track["center_y"],
            linewidth=1.2,
            label=f"ID {track_id}",
        )

    plt.gca().invert_yaxis()
    plt.xlabel("Image X (pixels)")
    plt.ylabel("Image Y (pixels)")
    plt.title("Trajectories of 15 Most Frequently Observed Tracks")
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "top_trajectories.png", dpi=160)
    plt.close()

    print(f"Plots saved to: {PLOT_PATH}")
    for path in sorted(PLOT_PATH.glob("*.png")):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
