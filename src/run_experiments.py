import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

try:
    from src.pipeline_utils import atomic_output_path
except ModuleNotFoundError:
    from pipeline_utils import atomic_output_path


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "input" / "sample.mp4"
OUTPUT_DIR = ROOT / "outputs" / "experiments"

MODEL_NAME = "yolo11n.pt"

EXPERIMENTS = {
    "default": "bytetrack.yaml",
    "balanced": str(ROOT / "configs" / "bytetrack_balanced.yaml"),
    "persistent": str(ROOT / "configs" / "bytetrack_persistent.yaml"),
}

logger = logging.getLogger(__name__)


def run_experiment(name, tracker_config):
    logger.info(
        "Starting experiment '%s' with tracker '%s'",
        name,
        tracker_config,
    )

    output_csv = OUTPUT_DIR / f"tracks_{name}.csv"
    frame_count = 0
    total_detections = 0
    track_frames = defaultdict(list)
    confidences = []
    start = time.perf_counter()

    cap = None

    try:
        model = YOLO(MODEL_NAME)

        with atomic_output_path(output_csv) as temporary_csv:
            cap = cv2.VideoCapture(str(VIDEO_PATH))

            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

            with temporary_csv.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "frame", "track_id", "class_name",
                    "confidence", "center_x", "center_y",
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

                    if boxes is not None and boxes.id is not None:
                        ids = boxes.id.int().cpu().tolist()
                        classes = boxes.cls.int().cpu().tolist()
                        scores = boxes.conf.cpu().tolist()
                        coordinates = boxes.xyxy.cpu().numpy()

                        for track_id, cls, score, box in zip(
                            ids, classes, scores, coordinates
                        ):
                            x1, y1, x2, y2 = box
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            writer.writerow([
                                frame_count,
                                track_id,
                                model.names[cls],
                                round(float(score), 4),
                                center_x,
                                center_y,
                            ])

                            track_frames[track_id].append(frame_count)
                            confidences.append(float(score))
                            total_detections += 1

                    if frame_count % 200 == 0:
                        logger.info(
                            "%s: processed %d frames", name, frame_count
                        )

            if frame_count == 0:
                raise RuntimeError(
                    f"No frames could be read from {VIDEO_PATH}"
                )

    except Exception:
        logger.exception("Experiment '%s' failed.", name)
        raise

    finally:
        if cap is not None:
            cap.release()

    elapsed = time.perf_counter() - start
    observed_lengths = [
        len(frames) for frames in track_frames.values()
    ]

    gap_events = sum(
        sum(
            current - previous > 1
            for previous, current in zip(frames, frames[1:])
        )
        for frames in track_frames.values()
    )

    result = {
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

    logger.info(
        "Experiment '%s' completed: %d frames, %d detections",
        name,
        frame_count,
        total_detections,
    )
    return result


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name, tracker_config in EXPERIMENTS.items():
        results.append(run_experiment(name, tracker_config))

    summary = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "experiment_summary.csv"

    with atomic_output_path(summary_path) as temporary_summary:
        summary.to_csv(temporary_summary, index=False)

    logger.info("EXPERIMENT COMPARISON\n%s", summary.to_string(index=False))
    logger.info("Experiment summary saved: %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()