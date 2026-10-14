
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "MOT17-02-FRCNN"
SOURCE_DIR = ROOT / "data" / "MOT17" / "train" / SEQUENCE
TRACK_EVAL = ROOT / "external" / "TrackEval"

GT_ROOT = TRACK_EVAL / "data" / "gt" / "mot_challenge"
GT_SEQUENCE = GT_ROOT / "MOT17-train" / SEQUENCE

TRACKER_NAME = "YOLOByteTrack"
TRACKER_ROOT = (
    TRACK_EVAL / "data" / "trackers" / "mot_challenge"
    / "MOT17-train" / TRACKER_NAME / "data"
)

PREDICTIONS = TRACKER_ROOT / f"{SEQUENCE}.txt"
OUTPUT_DIR = ROOT / "outputs" / "benchmark"

MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"  # Safe default while your cooling fan is broken.


def prepare_evaluation_data():
    images_dir = SOURCE_DIR / "img1"
    gt_file = SOURCE_DIR / "gt" / "gt.txt"
    seqinfo = SOURCE_DIR / "seqinfo.ini"

    for path in (images_dir, gt_file, seqinfo):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file missing: {path}")

    GT_SEQUENCE.mkdir(parents=True, exist_ok=True)
    (GT_SEQUENCE / "gt").mkdir(exist_ok=True)
    (GT_ROOT / "seqmaps").mkdir(parents=True, exist_ok=True)

    shutil.copy2(gt_file, GT_SEQUENCE / "gt" / "gt.txt")
    shutil.copy2(seqinfo, GT_SEQUENCE / "seqinfo.ini")

    seqmap = GT_ROOT / "seqmaps" / "MOT17-train.txt"
    seqmap.write_text(
        f"name\n{SEQUENCE}\n",
        encoding="utf-8",
    )

    images = sorted(
        images_dir.glob("*.jpg"),
        key=lambda path: int(path.stem),
    )

    if not images:
        raise RuntimeError(f"No JPG frames found in {images_dir}")

    return images


def export_predictions(images):
    TRACKER_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
    total_detections = 0

    with PREDICTIONS.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        for frame_number, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path))

            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],  # COCO class 0 = person
                device=DEVICE,
                verbose=False,
            )[0]

            boxes = result.boxes

            if boxes is None or boxes.id is None:
                continue

            coordinates = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for box, track_id, confidence in zip(
                coordinates, track_ids, confidences
            ):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1

                writer.writerow([
                    frame_number,
                    track_id,
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(width), 2),
                    round(float(height), 2),
                    round(float(confidence), 4),
                    -1, -1, -1,
                ])

                total_detections += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{len(images)} frames")

    print(f"Prediction file: {PREDICTIONS}")
    print(f"Person detections with track IDs: {total_detections}")


def run_trackeval():
    script = TRACK_EVAL / "scripts" / "run_mot_challenge.py"

    if not script.exists():
        raise FileNotFoundError(
            "TrackEval not found. Clone it into external/TrackEval first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    command = [
        sys.executable,
        str(script),
        "--BENCHMARK", "MOT17",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", TRACKER_NAME,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1",
        "--GT_FOLDER", str(GT_ROOT),
        "--TRACKERS_FOLDER", str(
            TRACK_EVAL / "data" / "trackers" / "mot_challenge"
        ),
        "--OUTPUT_FOLDER", str(OUTPUT_DIR),
        "--SEQMAP_FILE", str(GT_ROOT / "seqmaps" / "MOT17-train.txt"),
        "--DO_PREPROC", "True",
    ]

    subprocess.run(command, cwd=TRACK_EVAL, check=True)


def main():
    images = prepare_evaluation_data()
    print(f"Sequence: {SEQUENCE}; frames: {len(images)}")

    export_predictions(images)
    run_trackeval()

    print(f"Evaluation complete. Results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
