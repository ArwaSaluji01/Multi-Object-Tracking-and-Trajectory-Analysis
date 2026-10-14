from pathlib import Path

import numpy as np
import pandas as pd


def require_nonempty_file(path, description="Input file"):
    """Require an existing, non-empty regular file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {path}")

    return path


def validate_dataframe(
    df,
    *,
    source,
    required_columns,
    numeric_columns,
    integer_columns=(),
    unique_by=(),
):
    """Validate schema and numeric values without silently dropping rows."""
    if df.empty:
        raise ValueError(f"No data rows found in {source}")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    for column in numeric_columns:
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")
        values = converted.to_numpy(dtype=float, na_value=np.nan)

        invalid = ~np.isfinite(values)
        if invalid.any():
            rows = (np.flatnonzero(invalid)[:5] + 2).tolist()
            raise ValueError(
                f"{source}: column '{column}' contains missing, "
                f"non-numeric, or non-finite values near CSV rows {rows}"
            )

        result[column] = converted

    for column in integer_columns:
        values = result[column].to_numpy(dtype=float)

        if not np.equal(values, np.floor(values)).all():
            raise ValueError(
                f"{source}: column '{column}' must contain whole numbers"
            )

        result[column] = result[column].astype("int64")

    if unique_by:
        duplicates = result.duplicated(
            subset=list(unique_by), keep=False
        )
        if duplicates.any():
            sample = (
                result.loc[duplicates, list(unique_by)]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{source}: duplicate observations for keys "
                f"{list(unique_by)}; examples: {sample}"
            )

    return result


def read_validated_csv(
    path,
    *,
    required_columns,
    numeric_columns,
    integer_columns=(),
    unique_by=(),
    description="CSV file",
):
    """Read a CSV and validate its schema and values."""
    path = require_nonempty_file(path, description)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{description} has no readable header: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed CSV file: {path}") from exc

    return validate_dataframe(
        df,
        source=str(path),
        required_columns=required_columns,
        numeric_columns=numeric_columns,
        integer_columns=integer_columns,
        unique_by=unique_by,
    )


def write_csv_atomic(df, output_path):
    """Write a CSV to a temporary file, then replace the destination."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )

    try:
        df.to_csv(temporary_path, index=False)

        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise OSError(f"CSV output was not written correctly: {temporary_path}")

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
