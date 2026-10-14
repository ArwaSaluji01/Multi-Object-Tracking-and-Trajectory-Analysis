
import pandas as pd
import pytest

from src.io_validation import validate_dataframe


def validate_integer_value(value):
    return validate_dataframe(
        pd.DataFrame({"frame": [value]}),
        source="test data",
        required_columns=["frame"],
        numeric_columns=["frame"],
        integer_columns=["frame"],
    )


@pytest.mark.parametrize("value", [1e30, float(2**63)])
def test_integer_column_rejects_values_outside_int64_range(value):
    with pytest.raises(ValueError, match="int64 range"):
        validate_integer_value(value)


def test_integer_column_accepts_valid_whole_numbers():
    result = validate_integer_value(12)
    assert result.loc[0, "frame"] == 12
    assert str(result["frame"].dtype) == "int64"


def test_integer_column_rejects_fractional_values():
    with pytest.raises(ValueError, match="whole numbers"):
        validate_integer_value(1.5)
