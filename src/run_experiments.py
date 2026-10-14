
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "input" / "sample.mp4"
OUTPUT_DIR = ROOT / "outputs" / "experiments"

MODEL_NAME = "yolo11n.pt"

EXPERIMENTS = {
    "default": "bytetrack.yaml",
    "balanced": str(ROOT / "configs" / "bytetrack_balanced.yaml"),
    "persistent": str(ROOT / "configs" / "bytetrack_persistent.yaml"),
}


def run_experiment(name, tracker_config):
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    output_csv = OUTPUT_DIR / f"tracks_{name}.csv"
    frame_count = 0
    total_detections = 0
    track_frames = defaultdict(list)
    confidences = []
    start = time.perf_counter()

    try:
        with output_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                "frame", "track_id", "class_name",
                "confidence", "center_x", "center_y"
            ])

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1

                result = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_config,
                    verbose=False,
                )[0]

                boxes = result.boxes

                if boxes is None or boxes.id is None:
                    continue

                ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                coords = boxes.xyxy.cpu().numpy()

                for track_id, cls, score, box in zip(
                    ids, classes, scores, coords
                ):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    writer.writerow([
                        frame_count,
                        track_id,
                        model.names[cls],
                        round(float(score), 4),
                        cx,
                        cy,
                    ])

                    track_frames[track_id].append(frame_count)
                    confidences.append(float(score))
                    total_detections += 1

                if frame_count % 200 == 0:
                    print(f"{name}: processed {frame_count} frames")

    finally:
        cap.release()

    elapsed = time.perf_counter() - start

    observed_lengths = [
        len(frames) for frames in track_frames.values()
    ]

    # Count gaps between observed detections for each track.
    gap_events = sum(
        sum(
            current - previous > 1
            for previous, current in zip(frames, frames[1:])
        )
        for frames in track_frames.values()
    )

    return {
        "experiment": name,
        "frames_processed": frame_count,
        "processing_fps": round(frame_count / max(elapsed, 1e-9), 2),
        "total_detections": total_detections,
        "unique_track_ids": len(track_frames),
        "mean_observed_frames_per_track": round(
            sum(observed_lengths) / max(len(observed_lengths), 1), 2
        ),
        "short_tracks_le_10_frames": sum(
            length <= 10 for length in observed_lengths
        ),
        "detection_gap_events": gap_events,
        "mean_confidence": round(
            sum(confidences) / max(len(confidences), 1), 4
        ),
    }


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(VIDEO_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name, tracker_config in EXPERIMENTS.items():
        print(f"\nRunning experiment: {name}")
        results.append(run_experiment(name, tracker_config))

    summary = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nEXPERIMENT COMPARISON")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
