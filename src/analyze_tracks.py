from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from io_validation import read_validated_csv, require_nonempty_file, write_csv_atomic


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "trajectories" / "tracks.csv"
OUTPUT_CSV = ROOT / "outputs" / "trajectories" / "track_summary.csv"
VIDEO_PATH = ROOT / "data" / "input" / "sample.mp4"

TRACK_COLUMNS = [
    "frame", "track_id", "class_name", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
]
NUMERIC_COLUMNS = [
    "frame", "track_id", "confidence",
    "x1", "y1", "x2", "y2", "center_x", "center_y",
]


def load_tracking_data(path):
    df = read_validated_csv(
        path,
        required_columns=TRACK_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        integer_columns=("frame", "track_id"),
        unique_by=("track_id", "frame"),
        description="Tracking CSV",
    )

    if (df["frame"] < 1).any():
        raise ValueError("Tracking CSV contains frame numbers below 1.")

    if (df["track_id"] < 0).any():
        raise ValueError("Tracking CSV contains negative track IDs.")

    if not df["class_name"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Tracking CSV contains missing or empty class names.")

    if not df["confidence"].between(0, 1).all():
        raise ValueError("Tracking confidence values must be between 0 and 1.")

    if (df["x2"] < df["x1"]).any() or (df["y2"] < df["y1"]).any():
        raise ValueError("Tracking CSV contains invalid bounding boxes.")

    return df


def get_video_fps(path):
    require_nonempty_file(path, "Source video")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source video: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS ({fps}) in {path}")

        return fps
    finally:
        cap.release()


def build_summary(df, fps):
    if df.empty:
        raise ValueError("Cannot summarize an empty tracking DataFrame.")

    try:
        fps = float(fps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"FPS must be a finite positive number; received {fps!r}."
        ) from exc

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"FPS must be a finite positive number; received {fps!r}."
        )

    required = {
        "track_id", "frame", "class_name", "confidence",
        "center_x", "center_y",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tracking DataFrame missing columns: {sorted(missing)}")

    summaries = []

    for track_id, track in df.groupby("track_id"):
        track = track.sort_values("frame")
        first = track.iloc[0]
        last = track.iloc[-1]

        dx = float(last["center_x"] - first["center_x"])
        dy = float(last["center_y"] - first["center_y"])
        displacement = float(np.hypot(dx, dy))

        elapsed_frames = int(last["frame"] - first["frame"])
        inclusive_frames = elapsed_frames + 1

        points = track[["center_x", "center_y"]].to_numpy(dtype=float)
        step_distances = np.hypot(
            np.diff(points[:, 0]),
            np.diff(points[:, 1]),
        )

        class_modes = track["class_name"].mode()
        if class_modes.empty:
            raise ValueError(f"Track {track_id} has no valid class label.")

        summaries.append({
            "track_id": int(track_id),
            "class_name": class_modes.iloc[0],
            "observed_frames": int(track["frame"].nunique()),
            "first_frame": int(first["frame"]),
            "last_frame": int(last["frame"]),
            # Elapsed time between the first and last observations.
            "duration_seconds": round(elapsed_frames / fps, 4),
            # Inclusive duration represented by the frame span.
            "span_seconds": round(inclusive_frames / fps, 4),
            "displacement_pixels": round(displacement, 2),
            "avg_displacement_pixels_per_frame": round(
                displacement / elapsed_frames if elapsed_frames else 0.0,
                2,
            ),
            "mean_confidence": round(float(track["confidence"].mean()), 4),
            # Observed point-to-point path; unobserved motion cannot be recovered.
            "path_length_pixels": round(float(step_distances.sum()), 2),
        })

    return pd.DataFrame(summaries)


def main():
    df = load_tracking_data(INPUT_CSV)
    fps = get_video_fps(VIDEO_PATH)
    summary = build_summary(df, fps)

    write_csv_atomic(summary, OUTPUT_CSV)

    print(f"Unique track IDs: {df['track_id'].nunique()}")
    print("\nObjects by class:")
    print(df.groupby("class_name")["track_id"].nunique())
    print(f"\nSummary saved to: {OUTPUT_CSV}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()