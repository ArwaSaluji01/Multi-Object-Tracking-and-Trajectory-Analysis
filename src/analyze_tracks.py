
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "trajectories" / "tracks.csv"
OUTPUT_CSV = ROOT / "outputs" / "trajectories" / "track_summary.csv"


def main():
    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        raise ValueError("Tracking CSV contains no detections.")

    summaries = []

    for track_id, track in df.groupby("track_id"):
        track = track.sort_values("frame")
        first = track.iloc[0]
        last = track.iloc[-1]

        dx = last["center_x"] - first["center_x"]
        dy = last["center_y"] - first["center_y"]
        displacement = (dx**2 + dy**2) ** 0.5

        frame_span = int(last["frame"] - first["frame"])
        duration_seconds = frame_span / 30.0  # Replace with video FPS

        summaries.append({
            "track_id": track_id,
            "class_name": track["class_name"].mode().iloc[0],
            "observed_frames": track["frame"].nunique(),
            "first_frame": int(first["frame"]),
            "last_frame": int(last["frame"]),
            "duration_seconds": round(duration_seconds, 2),
            "displacement_pixels": round(displacement, 2),
            "avg_displacement_pixels_per_frame": round(
                displacement / max(frame_span, 1), 2
            ),
            "mean_confidence": round(track["confidence"].mean(), 4),
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_CSV, index=False)

    print(f"Unique track IDs: {df['track_id'].nunique()}")
    print("\nObjects by class:")
    print(df.groupby("class_name")["track_id"].nunique())
    print(f"\nSummary saved to: {OUTPUT_CSV}")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
