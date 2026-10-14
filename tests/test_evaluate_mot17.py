
import csv
from pathlib import Path

import pytest

from src import evaluate_mot17 as evaluator


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)
    return path


def prediction(frame=1, track_id=7, x=10, y=20, width=30, height=40,
               confidence=0.9):
    return [
        frame, track_id, x, y, width, height, confidence, 1, -1, -1
    ]


def test_valid_predictions_pass_validation(tmp_path):
    path = write_rows(
        tmp_path / "predictions.txt",
        [prediction(1, 7), prediction(2, 7), prediction(2, 8)],
    )

    result = evaluator.validate_mot_file(
        path,
        expected_frames=2,
        ground_truth=False,
    )

    assert result == {"rows": 3, "frames_with_rows": 2}


@pytest.mark.parametrize(
    "row",
    [
        prediction(frame=0),
        prediction(frame=3),
        prediction(track_id=0),
        prediction(width=0),
        prediction(height=-1),
        prediction(confidence=1.2),
        prediction(x=float("nan")),
        prediction(x=float("inf")),
    ],
)
def test_invalid_prediction_rows_are_rejected(tmp_path, row):
    path = write_rows(tmp_path / "predictions.txt", [row])

    with pytest.raises(ValueError):
        evaluator.validate_mot_file(
            path,
            expected_frames=2,
            ground_truth=False,
        )


def test_duplicate_track_id_in_same_frame_is_rejected(tmp_path):
    path = write_rows(
        tmp_path / "predictions.txt",
        [prediction(1, 7), prediction(1, 7)],
    )

    with pytest.raises(ValueError, match="duplicate predicted track ID"):
        evaluator.validate_mot_file(
            path,
            expected_frames=2,
            ground_truth=False,
        )


def test_prediction_with_wrong_column_count_is_rejected(tmp_path):
    path = write_rows(
        tmp_path / "predictions.txt",
        [[1, 7, 10, 20, 30, 40, 0.9]],
    )

    with pytest.raises(ValueError, match="expected 10 columns"):
        evaluator.validate_mot_file(
            path,
            expected_frames=2,
            ground_truth=False,
        )


def test_inconsistent_column_count_is_rejected(tmp_path):
    path = write_rows(
        tmp_path / "predictions.txt",
        [prediction(), [2, 8, 10, 20, 30, 40, 0.9]],
    )

    with pytest.raises(ValueError, match="inconsistent column count"):
        evaluator.validate_mot_file(
            path,
            expected_frames=2,
            ground_truth=False,
        )


def test_valid_ground_truth_passes_validation(tmp_path):
    # MOT ground-truth row: frame, ID, x, y, w, h, mark, class, visibility.
    path = write_rows(
        tmp_path / "gt.txt",
        [
            [1, 1, 10, 20, 30, 40, 1, 1, 0.9],
            [2, 1, 11, 21, 30, 40, 1, 1, 0.9],
        ],
    )

    result = evaluator.validate_mot_file(
        path,
        expected_frames=2,
        ground_truth=True,
    )

    assert result == {"rows": 2, "frames_with_rows": 2}


def test_empty_ground_truth_is_rejected(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        evaluator.validate_mot_file(
            path,
            expected_frames=2,
            ground_truth=True,
        )


def test_prepare_evaluation_data_checks_frame_sequence(tmp_path, monkeypatch):
    source = tmp_path / "source"
    images_dir = source / "img1"
    gt_file = source / "gt" / "gt.txt"
    seqinfo = source / "seqinfo.ini"

    images_dir.mkdir(parents=True)
    (images_dir / "000001.jpg").write_bytes(b"placeholder")
    (images_dir / "000002.jpg").write_bytes(b"placeholder")

    write_rows(
        gt_file,
        [[1, 1, 10, 20, 30, 40, 1, 1, 0.9]],
    )
    seqinfo.write_text(
        "[Sequence]\n"
        "seqLength=2\n"
        "imWidth=640\n"
        "imHeight=480\n"
        "frameRate=30\n",
        encoding="utf-8",
    )

    track_eval = tmp_path / "external" / "TrackEval"
    gt_root = track_eval / "data" / "gt" / "mot_challenge"
    gt_sequence = gt_root / "MOT17-train" / evaluator.SEQUENCE

    monkeypatch.setattr(evaluator, "SOURCE_DIR", source)
    monkeypatch.setattr(evaluator, "TRACK_EVAL", track_eval)
    monkeypatch.setattr(evaluator, "GT_ROOT", gt_root)
    monkeypatch.setattr(evaluator, "GT_SEQUENCE", gt_sequence)

    images = evaluator.prepare_evaluation_data()

    assert [image.name for image in images] == [
        "000001.jpg",
        "000002.jpg",
    ]
    assert (gt_sequence / "gt" / "gt.txt").is_file()
    assert (gt_sequence / "seqinfo.ini").is_file()
    assert (
        gt_root / "seqmaps" / "MOT17-02-FRCNN-only.txt"
    ).read_text(encoding="utf-8") == f"name\n{evaluator.SEQUENCE}\n"


def test_prepare_evaluation_data_rejects_missing_frame(tmp_path, monkeypatch):
    source = tmp_path / "source"
    images_dir = source / "img1"
    images_dir.mkdir(parents=True)

    # Frame 2 is missing even though the metadata declares two frames.
    (images_dir / "000001.jpg").write_bytes(b"placeholder")
    write_rows(
        source / "gt" / "gt.txt",
        [[1, 1, 10, 20, 30, 40, 1, 1, 0.9]],
    )
    (source / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "seqLength=2\n"
        "imWidth=640\n"
        "imHeight=480\n"
        "frameRate=30\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(evaluator, "SOURCE_DIR", source)

    with pytest.raises(RuntimeError, match="Frame sequence mismatch"):
        evaluator.prepare_evaluation_data()


def test_missing_mot_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluator.validate_mot_file(
            tmp_path / "missing.txt",
            expected_frames=10,
            ground_truth=False,
        )
