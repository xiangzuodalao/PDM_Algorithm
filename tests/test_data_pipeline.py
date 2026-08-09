import pandas as pd
import pytest

from valeo_pdm.transformer.data import (
    clean_and_resample_timeseries,
    count_windows,
    load_timeseries_file,
    minimum_rows_for_usable_window_splits,
    require_usable_window_splits,
    validate_window_params,
    window_split_counts,
)


def test_clean_resample_reports_bad_rows_and_averages_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "CollectTime": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:30:00Z",
                "bad-time",
                "2026-01-01T01:00:00Z",
            ],
            "Values": [1, 3, 9, "bad-value"],
        }
    )

    result, report = clean_and_resample_timeseries(frame, "1h")

    assert result["value"].tolist() == [2.0]
    assert report["dropped_invalid_time_rows"] == 1
    assert report["dropped_invalid_value_rows"] == 1
    assert report["aggregated_rows"] == 1
    assert report["rows_after_resample"] == 1


def test_mixed_valid_time_formats_are_not_dropped() -> None:
    frame = pd.DataFrame(
        {
            "collect_time": ["2026-01-01", "2026-01-01T01:00:00Z"],
            "value": [1, 2],
        }
    )

    result, report = clean_and_resample_timeseries(frame, "1h")

    assert len(result) == 2
    assert report["dropped_invalid_time_rows"] == 0


def test_json_rows_wrapper_uses_shared_loader(tmp_path) -> None:
    source = tmp_path / "timeseries.json"
    source.write_text(
        '{"rows": [{"collect_time": "2026-01-01T00:00:00Z", "value": 1}]}',
        encoding="utf-8",
    )

    result, report = load_timeseries_file(source, "1h")

    assert result["value"].tolist() == [1.0]
    assert report["rows_after_resample"] == 1


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "1"])
def test_window_params_reject_non_positive_or_non_integer_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_window_params(value, 1, 1)


def test_label_must_not_exceed_sequence() -> None:
    with pytest.raises(ValueError, match="label_len"):
        validate_window_params(2, 3, 1)


def test_window_formula_and_real_split() -> None:
    assert count_windows(20, seq_len=4, pred_len=2, stride=2) == 8
    assert window_split_counts(20, 4, 2, 2, stride=1) == {
        "rows": 20,
        "windows": 15,
        "train_windows": 12,
        "val_windows": 1,
        "test_windows": 2,
        "stride": 1,
    }
    with pytest.raises(ValueError, match="窗口不足"):
        require_usable_window_splits(14, 4, 2, 2)

    assert require_usable_window_splits(15, 4, 2, 2)["windows"] == 10
    assert minimum_rows_for_usable_window_splits(4, 2, 2) == 15
    assert count_windows(30, seq_len=4, pred_len=2, stride=7) == 4
