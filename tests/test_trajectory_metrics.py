import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_trajectories import calculate_metrics, load_tracks


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