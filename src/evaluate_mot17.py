import configparser
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from src.pipeline_utils import atomic_output_path
except ModuleNotFoundError:
    from pipeline_utils import atomic_output_path


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
DEVICE = "cpu"

logger = logging.getLogger(__name__)


def prepare_evaluation_data():
    images_dir = SOURCE_DIR / "img1"
    gt_file = SOURCE_DIR / "gt" / "gt.txt"
    seqinfo = SOURCE_DIR / "seqinfo.ini"

    for path in (images_dir, gt_file, seqinfo):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file missing: {path}"
            )

    config = configparser.ConfigParser()
    if not config.read(seqinfo, encoding="utf-8"):
        raise RuntimeError(f"Could not read sequence metadata: {seqinfo}")

    try:
        expected_frames = config.getint("Sequence", "seqLength")
    except (configparser.Error, ValueError) as exc:
        raise RuntimeError(
            f"Missing or invalid Sequence.seqLength in {seqinfo}"
        ) from exc

    images = sorted(
        images_dir.glob("*.jpg"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
    )

    if not images:
        raise RuntimeError(f"No JPG frames found in {images_dir}")

    frame_numbers = [
        int(path.stem) for path in images if path.stem.isdigit()
    ]

    if len(frame_numbers) != len(images):
        raise RuntimeError(
            f"Non-numeric frame filename found in {images_dir}"
        )

    expected_numbers = list(range(1, expected_frames + 1))

    if frame_numbers != expected_numbers:
        raise RuntimeError(
            f"Frame sequence mismatch: metadata expects frames "
            f"1–{expected_frames}, but found {len(frame_numbers)} JPGs "
            f"with contiguous numbering: "
            f"{frame_numbers == expected_numbers}"
        )

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

    logger.info(
        "Prepared sequence %s: %d frames", SEQUENCE, len(images)
    )
    return images


def export_predictions(images):
    TRACKER_ROOT.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)
    total_detections = 0

    try:
        with atomic_output_path(PREDICTIONS) as temporary_predictions:
            with temporary_predictions.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)

                for frame_number, image_path in enumerate(images, start=1):
                    frame = cv2.imread(str(image_path))

                    if frame is None:
                        raise RuntimeError(
                            f"Could not read frame {frame_number}: "
                            f"{image_path}"
                        )

                    result = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        classes=[0],
                        device=DEVICE,
                        verbose=False,
                    )[0]

                    boxes = result.boxes

                    if boxes is not None and boxes.id is not None:
                        coordinates = boxes.xyxy.cpu().numpy()
                        track_ids = boxes.id.int().cpu().tolist()
                        confidences = boxes.conf.cpu().tolist()

                        for box, track_id, confidence in zip(
                            coordinates, track_ids, confidences
                        ):
                            x1, y1, x2, y2 = box

                            writer.writerow([
                                frame_number,
                                track_id,
                                round(float(x1), 2),
                                round(float(y1), 2),
                                round(float(x2 - x1), 2),
                                round(float(y2 - y1), 2),
                                round(float(confidence), 4),
                                -1, -1, -1,
                            ])
                            total_detections += 1

                    if frame_number % 100 == 0:
                        logger.info(
                            "MOT17: processed %d/%d frames",
                            frame_number,
                            len(images),
                        )

    except Exception:
        logger.exception("Failed to export MOT17 predictions.")
        raise

    logger.info("Prediction file: %s", PREDICTIONS)
    logger.info("Person detections with track IDs: %d", total_detections)


def run_trackeval():
    script = TRACK_EVAL / "scripts" / "run_mot_challenge.py"

    if not script.is_file():
        raise FileNotFoundError(
            f"TrackEval script not found: {script}. "
            "Ensure external/TrackEval is available."
        )

    if not PREDICTIONS.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS}"
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

    logger.info("Running TrackEval for %s", SEQUENCE)

    try:
        subprocess.run(
            command,
            cwd=TRACK_EVAL,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(
            "TrackEval failed with exit code %s. Command: %s",
            exc.returncode,
            exc.cmd,
        )
        raise RuntimeError(
            "TrackEval evaluation failed; see the preceding error output."
        ) from exc

    logger.info("TrackEval results saved under: %s", OUTPUT_DIR)


def main():
    images = prepare_evaluation_data()
    logger.info("Sequence: %s; frames: %d", SEQUENCE, len(images))

    export_predictions(images)
    run_trackeval()

    logger.info("MOT17 evaluation completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()