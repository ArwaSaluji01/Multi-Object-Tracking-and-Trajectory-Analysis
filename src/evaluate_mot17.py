import configparser
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
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
MANIFEST_PATH = OUTPUT_DIR / f"{SEQUENCE}_manifest.json"

MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"
TRACKER_CONFIG = "bytetrack.yaml"

logger = logging.getLogger(__name__)


def _read_mot_rows(path, *, expected_columns=None, min_columns=7):
    """Read and validate the numeric structure of a MOT-format file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"MOT file not found: {path}")

    rows = []
    column_count = None

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row or all(not field.strip() for field in row):
                continue

            if column_count is None:
                column_count = len(row)
                if expected_columns is not None and column_count != expected_columns:
                    raise ValueError(
                        f"{path}:{line_number}: expected {expected_columns} "
                        f"columns, found {column_count}"
                    )
                if column_count < min_columns:
                    raise ValueError(
                        f"{path}:{line_number}: expected at least "
                        f"{min_columns} columns, found {column_count}"
                    )
            elif len(row) != column_count:
                raise ValueError(
                    f"{path}:{line_number}: inconsistent column count "
                    f"{len(row)}; expected {column_count}"
                )

            try:
                values = [float(field.strip()) for field in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric field"
                ) from exc

            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}:{line_number}: non-finite numeric value"
                )

            frame_value, id_value = values[0], values[1]
            if not frame_value.is_integer() or frame_value < 1:
                raise ValueError(
                    f"{path}:{line_number}: frame number must be a "
                    "positive integer"
                )
            if not id_value.is_integer():
                raise ValueError(
                    f"{path}:{line_number}: track/object ID must be an integer"
                )

            x, y, width, height = values[2:6]
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{path}:{line_number}: bounding-box width and height "
                    "must be positive"
                )

            rows.append(values)

    return rows


def validate_mot_file(
    path,
    *,
    expected_frames,
    ground_truth=False,
):
    """Validate frame ranges, bounding boxes and per-frame track IDs."""
    rows = _read_mot_rows(
        path,
        expected_columns=None if ground_truth else 10,
        min_columns=7,
    )

    if ground_truth and not rows:
        raise ValueError(f"Ground-truth file is empty: {path}")

    seen_frame_ids = set()

    for line_number, row in enumerate(rows, start=1):
        frame_number = int(row[0])
        track_id = int(row[1])

        if frame_number > expected_frames:
            raise ValueError(
                f"{path}:{line_number}: frame {frame_number} exceeds "
                f"sequence length {expected_frames}"
            )

        if not ground_truth:
            if track_id < 1:
                raise ValueError(
                    f"{path}:{line_number}: predicted track IDs must be positive"
                )

            confidence = row[6]
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"{path}:{line_number}: confidence must be in [0, 1]"
                )

            key = (frame_number, track_id)
            if key in seen_frame_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate predicted track ID "
                    f"{track_id} in frame {frame_number}"
                )
            seen_frame_ids.add(key)

    return {
        "rows": len(rows),
        "frames_with_rows": len({int(row[0]) for row in rows}),
    }


def prepare_evaluation_data():
    images_dir = SOURCE_DIR / "img1"
    gt_file = SOURCE_DIR / "gt" / "gt.txt"
    seqinfo = SOURCE_DIR / "seqinfo.ini"

    for path in (images_dir, gt_file, seqinfo):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path missing: {path}")

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {images_dir}")

    config = configparser.ConfigParser()
    if not config.read(seqinfo, encoding="utf-8"):
        raise RuntimeError(f"Could not read sequence metadata: {seqinfo}")

    try:
        expected_frames = config.getint("Sequence", "seqLength")
        width = config.getint("Sequence", "imWidth")
        height = config.getint("Sequence", "imHeight")
        config.getint("Sequence", "frameRate")
    except (configparser.Error, ValueError) as exc:
        raise RuntimeError(
            f"Missing or invalid sequence metadata in {seqinfo}"
        ) from exc

    if expected_frames < 1 or width < 1 or height < 1:
        raise RuntimeError(f"Invalid sequence dimensions or length: {seqinfo}")

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
        raise RuntimeError(f"Non-numeric frame filename found in {images_dir}")

    expected_numbers = list(range(1, expected_frames + 1))
    if frame_numbers != expected_numbers:
        raise RuntimeError(
            f"Frame sequence mismatch: metadata expects {expected_frames} "
            f"frames, but found {len(images)} JPGs with contiguous numbering: "
            f"{frame_numbers == expected_numbers}"
        )

    gt_stats = validate_mot_file(
        gt_file,
        expected_frames=expected_frames,
        ground_truth=True,
    )
    logger.info(
        "Validated ground truth: %d rows across %d frames",
        gt_stats["rows"],
        gt_stats["frames_with_rows"],
    )

    destination_gt = GT_SEQUENCE / "gt" / "gt.txt"
    destination_seqinfo = GT_SEQUENCE / "seqinfo.ini"
    seqmap = GT_ROOT / "seqmaps" / "MOT17-02-FRCNN-only.txt"

    (GT_SEQUENCE / "gt").mkdir(parents=True, exist_ok=True)
    (GT_ROOT / "seqmaps").mkdir(parents=True, exist_ok=True)

    shutil.copy2(gt_file, destination_gt)
    shutil.copy2(seqinfo, destination_seqinfo)

    with atomic_output_path(seqmap) as temporary_seqmap:
        temporary_seqmap.write_text(
            f"name\n{SEQUENCE}\n",
            encoding="utf-8",
        )

    logger.info(
        "Prepared sequence %s: %d frames, %dx%d",
        SEQUENCE,
        len(images),
        width,
        height,
    )
    return images


def export_predictions(images):
    if not images:
        raise ValueError("Cannot export predictions for an empty image list")

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
                            f"Could not read frame {frame_number}: {image_path}"
                        )

                    result = model.track(
                        frame,
                        persist=True,
                        tracker=TRACKER_CONFIG,
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
                            x1, y1, x2, y2 = map(float, box)

                            # TrackEval reads column 7 as tracker class in
                            # this checkout. Use pedestrian class ID 1;
                            # final three fields are auxiliary placeholders.
                            writer.writerow([
                                frame_number,
                                int(track_id),
                                round(x1, 2),
                                round(y1, 2),
                                round(x2 - x1, 2),
                                round(y2 - y1, 2),
                                round(float(confidence), 4),
                                1,
                                -1,
                                -1,
                            ])
                            total_detections += 1

                    if frame_number % 100 == 0:
                        logger.info(
                            "MOT17: processed %d/%d frames",
                            frame_number,
                            len(images),
                        )

        stats = validate_mot_file(
            PREDICTIONS,
            expected_frames=len(images),
            ground_truth=False,
        )
        if stats["rows"] != total_detections:
            raise RuntimeError(
                "Prediction count mismatch after export: "
                f"expected {total_detections}, validated {stats['rows']}"
            )

    except Exception:
        logger.exception("Failed to export or validate MOT17 predictions.")
        raise

    logger.info("Prediction file: %s", PREDICTIONS)
    logger.info("Person detections with track IDs: %d", total_detections)


def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def write_evaluation_manifest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sequence": SEQUENCE,
        "benchmark": "MOT17",
        "split": "train",
        "tracker_name": TRACKER_NAME,
        "model": MODEL_NAME,
        "device": DEVICE,
        "tracker_config": TRACKER_CONFIG,
        "class_filter": {"person_class_id": 0},
        "metrics": ["HOTA", "CLEAR", "Identity"],
        "prediction_file": str(PREDICTIONS.relative_to(ROOT)),
        "ground_truth_source": str(
            (SOURCE_DIR / "gt" / "gt.txt").relative_to(ROOT)
        ),
        "versions": {
            "python": sys.version.split()[0],
            "ultralytics": _package_version("ultralytics"),
            "opencv-python": _package_version("opencv-python"),
            "numpy": _package_version("numpy"),
        },
    }

    with atomic_output_path(MANIFEST_PATH) as temporary_manifest:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    logger.info("Evaluation manifest saved: %s", MANIFEST_PATH)


def run_trackeval():
    script = TRACK_EVAL / "scripts" / "run_mot_challenge.py"
    seqmap = GT_ROOT / "seqmaps" / "MOT17-02-FRCNN-only.txt"

    for path in (script, PREDICTIONS, GT_SEQUENCE / "gt" / "gt.txt",
                 GT_SEQUENCE / "seqinfo.ini", seqmap):
        if not path.is_file():
            raise FileNotFoundError(f"Required TrackEval input missing: {path}")

    config = configparser.ConfigParser()
    config.read(GT_SEQUENCE / "seqinfo.ini", encoding="utf-8")
    try:
        expected_frames = config.getint("Sequence", "seqLength")
    except (configparser.Error, ValueError) as exc:
        raise RuntimeError("Invalid prepared sequence metadata") from exc

    validate_mot_file(
        PREDICTIONS,
        expected_frames=expected_frames,
        ground_truth=False,
    )
    validate_mot_file(
        GT_SEQUENCE / "gt" / "gt.txt",
        expected_frames=expected_frames,
        ground_truth=True,
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
        "--SEQMAP_FILE", str(seqmap),
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
    write_evaluation_manifest()
    run_trackeval()

    logger.info("MOT17 evaluation completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
