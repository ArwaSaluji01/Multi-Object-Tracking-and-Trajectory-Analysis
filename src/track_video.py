
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "data" / "input" / "sample.mp4"
OUTPUT_VIDEO = ROOT / "outputs" / "videos" / "tracked.mp4"
OUTPUT_CSV = ROOT / "outputs" / "trajectories" / "tracks.csv"

MODEL_NAME = "yolo11n.pt"
TRACKER = "bytetrack.yaml"
MAX_TRAJECTORY_POINTS = 50


def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    trajectories = defaultdict(
        lambda: deque(maxlen=MAX_TRAJECTORY_POINTS)
    )
    frame_number = 0
    start_time = time.perf_counter()

    try:
        with OUTPUT_CSV.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame", "track_id", "class_name", "confidence",
                "x1", "y1", "x2", "y2", "center_x", "center_y"
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

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()

                    for box, track_id, cls, confidence in zip(
                        boxes, ids, classes, confidences
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        trajectories[track_id].append(
                            (center_x, center_y)
                        )

                        csv_writer.writerow([
                            frame_number,
                            track_id,
                            model.names[cls],
                            round(confidence, 4),
                            x1, y1, x2, y2,
                            center_x, center_y,
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
                processing_fps = frame_number / max(elapsed, 1e-9)

                cv2.putText(
                    annotated,
                    f"Frame: {frame_number} | FPS: {processing_fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                writer.write(annotated)

                if frame_number % 100 == 0:
                    print(f"Processed {frame_number} frames")

    finally:
        cap.release()
        writer.release()

    print(f"Video saved: {OUTPUT_VIDEO}")
    print(f"Tracking data saved: {OUTPUT_CSV}")
    print(f"Frames processed: {frame_number}")


if __name__ == "__main__":
    main()
