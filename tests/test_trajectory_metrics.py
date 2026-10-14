import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_trajectories import calculate_metrics, load_tracks
from io_validation import read_validated_csv, write_csv_atomic
from analyze_tracks import build_summary


def test_load_tracks_sorts_observations(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame({
        "frame": [2, 1],
        "track_id": [7, 7],
        "center_x": [20, 10],
        "center_y": [20, 10],
    }).to_csv(path, index=False)

    df = load_tracks(path)
    assert df["frame"].tolist() == [1, 2]


def test_calculate_metrics():
    df = pd.DataFrame({
        "frame": [1, 2, 4],
        "track_id": [7, 7, 7],
        "x": [0, 3, 6],
        "y": [0, 4, 8],
    })

    metrics = calculate_metrics(df).iloc[0]

    assert metrics["observations"] == 3
    assert metrics["path_length_px"] == 10
    assert metrics["fragmentation_gaps"] == 1
    assert metrics["missing_frames"] == 1
    assert 0 <= metrics["straightness"] <= 1


def test_multiple_tracks():
    df = pd.DataFrame({
        "frame": [1, 2, 1],
        "track_id": [1, 1, 2],
        "x": [0, 3, 10],
        "y": [0, 4, 10],
    })

    metrics = calculate_metrics(df)

    assert len(metrics) == 2
    assert set(metrics["track_id"]) == {1, 2}


def test_load_tracks_rejects_missing_columns(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame({"frame": [1], "track_id": [1]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="Missing columns"):
        load_tracks(path)


def test_load_tracks_rejects_non_integer_frame(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame({
        "frame": [1.5],
        "track_id": [1],
        "center_x": [10],
        "center_y": [20],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="whole numbers"):
        load_tracks(path)


def test_load_tracks_rejects_duplicate_track_frame(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame({
        "frame": [1, 1],
        "track_id": [4, 4],
        "center_x": [10, 11],
        "center_y": [20, 21],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate observations"):
        load_tracks(path)


def test_load_tracks_rejects_non_finite_coordinates(tmp_path):
    path = tmp_path / "tracks.csv"
    pd.DataFrame({
        "frame": [1],
        "track_id": [1],
        "center_x": [float("inf")],
        "center_y": [20],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="non-finite"):
        load_tracks(path)


def test_empty_csv_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        read_validated_csv(
            path,
            required_columns=["frame"],
            numeric_columns=["frame"],
        )


def test_atomic_csv_write(tmp_path):
    path = tmp_path / "result.csv"
    path.write_text("old\nvalue\n", encoding="utf-8")

    write_csv_atomic(pd.DataFrame({"value": [42]}), path)

    assert pd.read_csv(path)["value"].tolist() == [42]
    assert not (tmp_path / ".result.tmp.csv").exists()

def test_speed_accounts_for_missing_frames():
    df = pd.DataFrame({
        "frame": [1, 4],
        "track_id": [7, 7],
        "x": [0, 6],
        "y": [0, 0],
    })

    metrics = calculate_metrics(df).iloc[0]

    assert metrics["path_length_px"] == 6
    assert metrics["net_displacement_px"] == 6
    assert metrics["mean_speed_px_per_frame"] == 2
    assert metrics["max_speed_px_per_frame"] == 2
    assert metrics["missing_frames"] == 2


def test_stationary_track_has_zero_speed_and_straightness():
    df = pd.DataFrame({
        "frame": [1, 2, 3],
        "track_id": [7, 7, 7],
        "x": [5, 5, 5],
        "y": [10, 10, 10],
    })

    metrics = calculate_metrics(df).iloc[0]

    assert metrics["path_length_px"] == 0
    assert metrics["net_displacement_px"] == 0
    assert metrics["mean_speed_px_per_frame"] == 0
    assert metrics["straightness"] == 0


def test_single_observation_track_has_zero_motion():
    df = pd.DataFrame({
        "frame": [10],
        "track_id": [3],
        "x": [12],
        "y": [25],
    })

    metrics = calculate_metrics(df).iloc[0]

    assert metrics["observations"] == 1
    assert metrics["span_frames"] == 1
    assert metrics["path_length_px"] == 0
    assert metrics["net_displacement_px"] == 0
    assert metrics["mean_speed_px_per_frame"] == 0
    assert metrics["straightness"] == 0


def test_calculate_metrics_rejects_empty_dataframe():
    df = pd.DataFrame(columns=["frame", "track_id", "x", "y"])

    with pytest.raises(ValueError, match="empty"):
        calculate_metrics(df)


def test_calculate_metrics_rejects_duplicate_observations():
    df = pd.DataFrame({
        "frame": [1, 1],
        "track_id": [7, 7],
        "x": [0, 1],
        "y": [0, 1],
    })

    with pytest.raises(ValueError, match="Duplicate observations"):
        calculate_metrics(df)


def test_calculate_metrics_rejects_non_finite_coordinates():
    df = pd.DataFrame({
        "frame": [1],
        "track_id": [7],
        "x": [float("inf")],
        "y": [0],
    })

    with pytest.raises(ValueError, match="non-finite"):
        calculate_metrics(df)


def test_track_summary_duration_uses_fps():
    df = pd.DataFrame({
        "frame": [10, 40],
        "track_id": [7, 7],
        "class_name": ["person", "person"],
        "confidence": [0.8, 0.9],
        "center_x": [0, 30],
        "center_y": [0, 40],
    })

    summary = build_summary(df, fps=30).iloc[0]

    assert summary["duration_seconds"] == 1.0
    assert summary["span_seconds"] == pytest.approx(31 / 30, abs=0.0001)
    assert summary["displacement_pixels"] == 50
    assert summary["avg_displacement_pixels_per_frame"] == pytest.approx(1.67)


def test_track_summary_rejects_invalid_fps():
    df = pd.DataFrame({
        "frame": [1],
        "track_id": [7],
        "class_name": ["person"],
        "confidence": [0.9],
        "center_x": [10],
        "center_y": [20],
    })

    with pytest.raises(ValueError, match="FPS"):
        build_summary(df, fps=0)


def test_track_summary_rejects_empty_dataframe():
    df = pd.DataFrame(columns=[
        "frame", "track_id", "class_name",
        "confidence", "center_x", "center_y",
    ])

    with pytest.raises(ValueError, match="empty"):
        build_summary(df, fps=30)