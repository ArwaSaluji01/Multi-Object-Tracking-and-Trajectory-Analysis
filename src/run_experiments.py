import csv
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml

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


def load_tracker_config(config_path):
    """Resolve and load a built-in or project-local tracker YAML."""
    resolved = Path(check_yaml(str(config_path))).resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Tracker configuration not found: {resolved}"
        )

    with resolved.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    if not isinstance(settings, dict):
        raise ValueError(f"Invalid tracker configuration: {resolved}")

    required = {
        "tracker_type",
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
    }
    missing = required - set(settings)

    if missing:
        raise ValueError(
            f"Tracker configuration {resolved} is missing: "
            f"{sorted(missing)}"
        )

    for key in (
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "match_thresh",
    ):
        value = settings[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(
                f"{key} must be between 0 and 1 in {resolved}"
            )

    if (
        not isinstance(settings["track_buffer"], int)
        or settings["track_buffer"] < 0
    ):
        raise ValueError(
            f"track_buffer must be a non-negative integer in {resolved}"
        )

    if settings["tracker_type"] != "bytetrack":
        raise ValueError(
            f"Expected ByteTrack, got {settings['tracker_type']!r}"
        )

    return resolved, settings


def calculate_track_statistics(track_frames, confidences):
    """Calculate descriptive statistics from observed tracker outputs."""
    observed_lengths = [len(frames) for frames in track_frames.values()]

    gap_events = sum(
        sum(
            current - previous > 1
            for previous, current in zip(frames, frames[1:])
        )
        for frames in track_frames.values()
    )

    return {
        "total_detections": sum(observed_lengths),
        "unique_track_ids": len(track_frames),
        "mean_observed_frames_per_track": round(
            sum(observed_lengths) / len(observed_lengths), 2
        ) if observed_lengths else 0.0,
        "short_tracks_le_10_frames": sum(
            length <= 10 for length in observed_lengths
        ),
        "detection_gap_events": gap_events,
        "mean_confidence": round(
            sum(confidences) / len(confidences), 4
        ) if confidences else 0.0,
    }


def run_experiment(name, tracker_config):
    if not VIDEO_PATH.is_file() or VIDEO_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Input video missing or empty: {VIDEO_PATH}"
        )

    resolved_config, settings = load_tracker_config(tracker_config)
    config_hash = hashlib.sha256(
        resolved_config.read_bytes()
    ).hexdigest()

    logger.info(
        "Starting '%s' | tracker=%s | settings=%s",
        name,
        resolved_config,
        settings,
    )

    output_csv = OUTPUT_DIR / f"tracks_{name}.csv"
    frame_count = 0
    track_frames = defaultdict(list)
    confidences = []
    total_start = time.perf_counter()
    cap = None

    try:
        # Model initialization is excluded from processing FPS.
        model_start = time.perf_counter()
        model = YOLO(MODEL_NAME)
        model_load_seconds = time.perf_counter() - model_start

        with atomic_output_path(output_csv) as temporary_csv:
            cap = cv2.VideoCapture(str(VIDEO_PATH))

            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

            processing_start = time.perf_counter()

            try:
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
                            tracker=str(resolved_config),
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

                        if frame_count % 200 == 0:
                            logger.info(
                                "%s: processed %d frames", name, frame_count
                            )

            finally:
                # Stop timing after processing and close the capture before
                # the atomic output context attempts to rename the CSV.
                processing_seconds = (
                    time.perf_counter() - processing_start
                )
                cap.release()
                cap = None

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

    end_to_end_seconds = time.perf_counter() - total_start
    stats = calculate_track_statistics(track_frames, confidences)

    result = {
        "experiment": name,
        "model": MODEL_NAME,
        "ultralytics_version": ultralytics.__version__,
        "input_video": str(VIDEO_PATH.relative_to(ROOT)),
        "input_size_bytes": VIDEO_PATH.stat().st_size,
        "tracker_config": str(resolved_config.relative_to(ROOT))
        if resolved_config.is_relative_to(ROOT)
        else str(resolved_config),
        "tracker_config_sha256": config_hash,
        "tracker_settings": json.dumps(settings, sort_keys=True),
        "frames_processed": frame_count,
        "model_load_seconds": round(model_load_seconds, 4),
        "processing_seconds": round(processing_seconds, 4),
        "processing_fps": round(
            frame_count / processing_seconds, 2
        ) if processing_seconds > 0 else 0.0,
        "end_to_end_seconds": round(end_to_end_seconds, 4),
        **stats,
    }

    logger.info(
        "Completed '%s': %d frames, %d detections, %.2f processing FPS",
        name,
        frame_count,
        stats["total_detections"],
        result["processing_fps"],
    )
    return result


def main():
    if not VIDEO_PATH.is_file() or VIDEO_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Input video missing or empty: {VIDEO_PATH}"
        )

    # Fail before running expensive inference if any configuration is invalid.
    loaded_configs = {
        name: load_tracker_config(path)
        for name, path in EXPERIMENTS.items()
    }

    fingerprints = {
        name: json.dumps(settings, sort_keys=True)
        for name, (_, settings) in loaded_configs.items()
    }

    if len(set(fingerprints.values())) != len(fingerprints):
        logger.warning(
            "Some experiments have identical tracker settings. "
            "Their comparison may not test different configurations."
        )

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
