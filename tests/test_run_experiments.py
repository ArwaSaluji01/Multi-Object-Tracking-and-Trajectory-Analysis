import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_experiments import (
    EXPERIMENTS,
    calculate_track_statistics,
    load_tracker_config,
)


def test_all_experiment_configs_load():
    loaded = {
        name: load_tracker_config(path)[1]
        for name, path in EXPERIMENTS.items()
    }

    assert set(loaded) == {"default", "balanced", "persistent"}
    assert all(config["tracker_type"] == "bytetrack"
               for config in loaded.values())


def test_experiment_configs_are_distinct():
    configs = [
        load_tracker_config(path)[1]
        for path in EXPERIMENTS.values()
    ]
    fingerprints = [repr(sorted(config.items())) for config in configs]

    assert len(set(fingerprints)) == len(fingerprints)


def test_persistent_config_has_longer_track_buffer():
    default = load_tracker_config(EXPERIMENTS["default"])[1]
    persistent = load_tracker_config(EXPERIMENTS["persistent"])[1]

    assert persistent["track_buffer"] > default["track_buffer"]


def test_balanced_config_uses_higher_track_creation_threshold():
    default = load_tracker_config(EXPERIMENTS["default"])[1]
    balanced = load_tracker_config(EXPERIMENTS["balanced"])[1]

    assert balanced["new_track_thresh"] > default["new_track_thresh"]
    assert balanced["track_high_thresh"] > default["track_high_thresh"]


def test_track_statistics_calculation():
    tracks = {
        1: [1, 2, 5],
        2: [1, 2],
        3: [4],
    }
    confidences = [0.8, 0.9, 0.7, 0.6, 0.5, 0.4]

    stats = calculate_track_statistics(tracks, confidences)

    assert stats["total_detections"] == 6
    assert stats["unique_track_ids"] == 3
    assert stats["mean_observed_frames_per_track"] == 2.0
    assert stats["short_tracks_le_10_frames"] == 3
    assert stats["detection_gap_events"] == 1
    assert stats["mean_confidence"] == pytest.approx(0.65)


def test_track_statistics_handle_no_detections():
    stats = calculate_track_statistics({}, [])

    assert stats["total_detections"] == 0
    assert stats["unique_track_ids"] == 0
    assert stats["mean_observed_frames_per_track"] == 0
    assert stats["detection_gap_events"] == 0
    assert stats["mean_confidence"] == 0
