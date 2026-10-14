import csv
import logging
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from src.pipeline_utils import atomic_output_path
except ModuleNotFoundError:
    from pipeline_utils import atomic_output_path


ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "data" / "input" / "sample.mp4"
OUTPUT_VIDEO = ROOT / "outputs" / "videos" / "tracked.mp4"
OUTPUT_CSV = ROOT / "outputs" / "trajectories" / "tracks.csv"

MODEL_NAME = "yolo11n.pt"
TRACKER = "bytetrack.yaml"
MAX_TRAJECTORY_POINTS = 50

logger = logging.getLogger(__name__)

def main():
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    logger.info("Loading model: %s", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    frame_number = 0
    start_time = time.perf_counter()

    try:
        with atomic_output_path(OUTPUT_VIDEO) as temporary_video, \
             atomic_output_path(OUTPUT_CSV) as temporary_csv:

            cap = None
            writer = None

            try:
                cap = cv2.VideoCapture(str(INPUT_VIDEO))

                if not cap.isOpened():
                    raise RuntimeError(
                        f"Could not open input video: {INPUT_VIDEO}"
                    )

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_fps = cap.get(cv2.CAP_PROP_FPS)

                if width <= 0 or height <= 0:
                    raise RuntimeError(
                        f"Invalid video dimensions: {width}x{height}"
                    )

                if not math.isfinite(source_fps) or source_fps <= 0:
                    logger.warning("Invalid source FPS; using 30 FPS.")
                    source_fps = 30.0

                writer = cv2.VideoWriter(
                    str(temporary_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not initialize writer: {temporary_video}"
                    )

                trajectories = defaultdict(
                    lambda: deque(maxlen=MAX_TRAJECTORY_POINTS)
                )

                with temporary_csv.open(
                    "w", newline="", encoding="utf-8"
                ) as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([
                        "frame", "track_id", "class_name", "confidence",
                        "x1", "y1", "x2", "y2", "center_x", "center_y",
                    ])

                    while True:
                        success, frame = cap.read()
                        if not success:
                            break

                        frame_number += 1

                        result = model.track(
                            frame,
                            persist=True,
                            tracker=TRACKER,
                            verbose=False,
                        )[0]

                        annotated = result.plot()
                        boxes = result.boxes

                        if boxes is not None and boxes.id is not None:
                            coordinates = boxes.xyxy.cpu().numpy()
                            ids = boxes.id.int().cpu().tolist()
                            classes = boxes.cls.int().cpu().tolist()
                            scores = boxes.conf.cpu().tolist()

                            for box, track_id, cls, score in zip(
                                coordinates, ids, classes, scores
                            ):
                                x1, y1, x2, y2 = map(int, box)
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2

                                trajectories[track_id].append((cx, cy))

                                csv_writer.writerow([
                                    frame_number, track_id, model.names[cls],
                                    round(float(score), 4),
                                    x1, y1, x2, y2, cx, cy,
                                ])

                        for points in trajectories.values():
                            for i in range(1, len(points)):
                                cv2.line(
                                    annotated,
                                    points[i - 1],
                                    points[i],
                                    (0, 255, 255),
                                    2,
                                )

                        elapsed = time.perf_counter() - start_time
                        fps = frame_number / max(elapsed, 1e-9)

                        cv2.putText(
                            annotated,
                            f"Frame: {frame_number} | FPS: {fps:.1f}",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        writer.write(annotated)

                        if frame_number % 100 == 0:
                            logger.info("Processed %d frames", frame_number)

                if frame_number == 0:
                    raise RuntimeError(
                        f"No frames could be read from {INPUT_VIDEO}"
                    )

            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()

    except Exception:
        logger.exception("Video tracking failed.")
        raise

    logger.info("Tracking completed: %d frames", frame_number)
    logger.info("Tracked video: %s", OUTPUT_VIDEO)
    logger.info("Tracking CSV: %s", OUTPUT_CSV)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()