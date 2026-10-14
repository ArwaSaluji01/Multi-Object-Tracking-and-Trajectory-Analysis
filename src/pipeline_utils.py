import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def atomic_output_path(final_path):
    """Write to a temporary file, then publish it after success."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = final_path.with_name(
        f"{final_path.stem}.partial{final_path.suffix}"
    )

    temporary_path.unlink(missing_ok=True)

    try:
        yield temporary_path

        if not temporary_path.is_file():
            raise RuntimeError(
                f"Expected output was not created: {temporary_path}"
            )

        os.replace(temporary_path, final_path)
        logger.info("Saved output: %s", final_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        logger.exception("Failed to create output: %s", final_path)
        raise