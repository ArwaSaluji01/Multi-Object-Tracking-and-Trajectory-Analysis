from pathlib import Path

import pytest

from src.pipeline_utils import atomic_output_path


def test_atomic_output_publishes_completed_file(tmp_path):
    destination = tmp_path / "result.csv"

    with atomic_output_path(destination) as temporary:
        temporary.write_text("a,b\n1,2\n", encoding="utf-8")

    assert destination.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert not (tmp_path / "result.partial.csv").exists()


def test_atomic_output_removes_partial_file_on_failure(tmp_path):
    destination = tmp_path / "result.csv"

    with pytest.raises(RuntimeError, match="simulated failure"):
        with atomic_output_path(destination) as temporary:
            temporary.write_text("incomplete", encoding="utf-8")
            raise RuntimeError("simulated failure")

    assert not destination.exists()
    assert not (tmp_path / "result.partial.csv").exists()


def test_atomic_output_preserves_previous_output_on_failure(tmp_path):
    destination = tmp_path / "result.csv"
    destination.write_text("previous result", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated failure"):
        with atomic_output_path(destination) as temporary:
            temporary.write_text("incomplete", encoding="utf-8")
            raise RuntimeError("simulated failure")

    assert destination.read_text(encoding="utf-8") == "previous result"