"""Golden tests for the stdlib dataset post-transforms (standardize/demean)."""

from __future__ import annotations

import statistics

import pytest

from lazystats import ReturnDataset, demean, series_values, standardize


@pytest.fixture()
def dataset() -> ReturnDataset:
    return ReturnDataset(
        instruments=["ticker:SPY", "macro:FEDFUNDS"],
        rows=[
            {"date": "2024-01-01", "ticker:SPY": 0.01, "macro:FEDFUNDS": 5.25},
            {"date": "2024-01-02", "ticker:SPY": 0.02, "macro:FEDFUNDS": 5.50},
            {"date": "2024-01-03", "ticker:SPY": 0.03, "macro:FEDFUNDS": None},
            {"date": "2024-01-04", "ticker:SPY": None, "macro:FEDFUNDS": 5.00},
        ],
        metadata={"source": "market-data-hub"},
    )


def test_standardize_zscores_each_column(dataset: ReturnDataset) -> None:
    out = standardize(dataset)
    values = series_values(out)
    for instrument in out.instruments:
        sample = values[instrument]
        assert statistics.fmean(sample) == pytest.approx(0.0, abs=1e-9)
        assert statistics.stdev(sample) == pytest.approx(1.0)
    # golden: SPY sample is (0.01, 0.02, 0.03) -> z = (-1, 0, 1)
    assert values["ticker:SPY"] == pytest.approx([-1.0, 0.0, 1.0])


def test_standardize_preserves_missing_and_dates(dataset: ReturnDataset) -> None:
    out = standardize(dataset)
    assert [row["date"] for row in out.rows] == [row["date"] for row in dataset.rows]
    assert out.rows[2]["macro:FEDFUNDS"] is None
    assert out.rows[3]["ticker:SPY"] is None


def test_standardize_subset_leaves_others_untouched(dataset: ReturnDataset) -> None:
    out = standardize(dataset, instruments=["ticker:SPY"])
    assert out.rows[0]["macro:FEDFUNDS"] == pytest.approx(5.25)
    assert out.rows[0]["ticker:SPY"] == pytest.approx(-1.0)
    assert out.metadata["post_transforms"] == {"ticker:SPY": "standardize"}


def test_demean_golden(dataset: ReturnDataset) -> None:
    out = demean(dataset, instruments=["ticker:SPY"])
    assert series_values(out)["ticker:SPY"] == pytest.approx([-0.01, 0.0, 0.01])
    assert out.metadata["post_transforms"] == {"ticker:SPY": "demean"}


def test_original_dataset_is_not_mutated(dataset: ReturnDataset) -> None:
    standardize(dataset)
    assert dataset.rows[0]["ticker:SPY"] == pytest.approx(0.01)
    assert "post_transforms" not in dataset.metadata


def test_standardize_rejects_zero_variance() -> None:
    flat = ReturnDataset(
        instruments=["ticker:X"],
        rows=[{"date": f"2024-01-{d:02d}", "ticker:X": 0.5} for d in range(1, 4)],
    )
    with pytest.raises(ValueError, match="zero variance"):
        standardize(flat)


def test_standardize_rejects_too_few_observations() -> None:
    tiny = ReturnDataset(
        instruments=["ticker:X"], rows=[{"date": "2024-01-01", "ticker:X": 0.5}]
    )
    with pytest.raises(ValueError, match="at least 2 observations"):
        standardize(tiny)


def test_selection_validation(dataset: ReturnDataset) -> None:
    with pytest.raises(ValueError, match="unknown instruments"):
        standardize(dataset, instruments=["ticker:NOPE"])
    with pytest.raises(ValueError, match="empty"):
        standardize(dataset, instruments=[])
    with pytest.raises(ValueError, match="unique"):
        standardize(dataset, instruments=["ticker:SPY", "ticker:SPY"])
