from pathlib import Path

from valeo_pdm.paths import resolve_repo_path


def test_relative_data_path_is_resolved_from_repo_when_cwd_is_elsewhere(
    tmp_path: Path, monkeypatch
) -> None:
    expected_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    resolved = resolve_repo_path("tests/fixtures/onboard_scenario/csv_timeseries.csv")

    assert (
        resolved == expected_root / "tests" / "fixtures" / "onboard_scenario" / "csv_timeseries.csv"
    )
