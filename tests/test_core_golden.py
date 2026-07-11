"""Golden numeric tests — the same expected values as LazyTools'
test_statistical_analysis.py (plan v3.1 Fase 6 acceptance: identical numeric
output on the golden datasets)."""

from __future__ import annotations

import pytest

from lazystats import ReturnDataset, return_correlation, return_outliers, return_volatility


@pytest.fixture()
def dataset() -> ReturnDataset:
    """SPY/TLT, 4 rows, last SPY row missing — mirror of the LazyTools fixture."""
    return ReturnDataset(
        instruments=["ticker:SPY", "ticker:TLT"],
        rows=[
            {"date": "2024-01-01", "ticker:SPY": 0.01, "ticker:TLT": -0.02},
            {"date": "2024-01-02", "ticker:SPY": 0.02, "ticker:TLT": -0.01},
            {"date": "2024-01-03", "ticker:SPY": 0.03, "ticker:TLT": 0.0},
            {"date": "2024-01-04", "ticker:SPY": None, "ticker:TLT": 0.01},
        ],
        metadata={"source": "market-data-hub"},
    )


def test_volatility_golden_annualization(dataset: ReturnDataset) -> None:
    out = return_volatility(dataset, frequency="W")
    assert out["periods_per_year"] == 52
    spy = out["volatility"]["ticker:SPY"]
    assert spy["observations"] == 3
    assert spy["annualized_volatility"] == pytest.approx(0.0721110255)
    assert spy["period_volatility"] == pytest.approx(0.01)
    assert spy["mean_log_return"] == pytest.approx(0.02)


def test_volatility_single_observation_is_none() -> None:
    ds = ReturnDataset(instruments=["ticker:X"],
                       rows=[{"date": "2024-01-01", "ticker:X": 0.01}])
    vol = return_volatility(ds)["volatility"]["ticker:X"]
    assert vol["observations"] == 1
    assert vol["annualized_volatility"] is None


def test_correlation_golden_pairwise(dataset: ReturnDataset) -> None:
    out = return_correlation(dataset, frequency="W")
    assert out["pairwise_observations"]["ticker:SPY"]["ticker:TLT"] == 3
    assert out["correlation"]["ticker:SPY"]["ticker:TLT"] == pytest.approx(1.0)
    assert out["correlation"]["ticker:SPY"]["ticker:SPY"] == pytest.approx(1.0)


def test_correlation_min_periods_validation(dataset: ReturnDataset) -> None:
    with pytest.raises(ValueError, match="min_periods"):
        return_correlation(dataset, min_periods=1)
    out = return_correlation(dataset, min_periods=4)
    assert out["correlation"]["ticker:SPY"]["ticker:TLT"] is None


def test_outliers_golden_zscore() -> None:
    rows = [{"date": f"2024-01-{d:02d}", "ticker:SPY": 0.0} for d in range(1, 6)]
    rows.append({"date": "2024-01-07", "ticker:SPY": 0.10})
    ds = ReturnDataset(instruments=["ticker:SPY"], rows=rows)
    out = return_outliers(ds, threshold=2.0)
    assert out["total_outliers"] == 1
    hit = out["outliers"][0]
    assert hit["date"] == "2024-01-07"
    assert hit["log_return"] == pytest.approx(0.1)
    assert hit["z_score"] == pytest.approx(2.0412414523)
    assert hit["direction"] == "positive"


def test_outliers_threshold_validation() -> None:
    ds = ReturnDataset(instruments=["ticker:X"], rows=[])
    with pytest.raises(ValueError, match="threshold"):
        return_outliers(ds, threshold=0)


def test_outliers_returns_all_uncapped() -> None:
    """Core returns EVERYTHING; output caps are the bridge's concern."""
    rows = [{"date": f"2024-{1 + d // 28:02d}-{1 + d % 28:02d}", "ticker:X":
             (0.1 if d % 2 else -0.1)} for d in range(300)]
    ds = ReturnDataset(instruments=["ticker:X"], rows=rows)
    out = return_outliers(ds, threshold=0.0001)
    assert out["total_outliers"] == 300
    assert len(out["outliers"]) == 300


def test_non_numeric_return_raises(dataset: ReturnDataset) -> None:
    bad = ReturnDataset(instruments=["ticker:X"],
                        rows=[{"date": "2024-01-01", "ticker:X": "oops"}])
    with pytest.raises(ValueError, match="non-numeric"):
        return_volatility(bad)


def test_parity_with_lazytools_implementation() -> None:
    """Cross-repo drift guard: when lazytools is installed, the LazyTools tool
    (fake backend) and lazystats core must produce identical numbers."""
    lt_stats = pytest.importorskip("lazytools.statistical_analysis")
    import json

    rows = [
        {"date": "2024-01-01", "ticker:SPY": 0.011, "ticker:TLT": -0.021},
        {"date": "2024-01-02", "ticker:SPY": 0.025, "ticker:TLT": -0.012},
        {"date": "2024-01-03", "ticker:SPY": -0.034, "ticker:TLT": 0.003},
        {"date": "2024-01-04", "ticker:SPY": 0.007, "ticker:TLT": None},
    ]
    instruments = ["ticker:SPY", "ticker:TLT"]
    lt_dataset = lt_stats.ReturnDataset(
        instruments=instruments, rows=rows, metadata={"source": "market-data-hub"})

    class _Backend:
        def load_returns(self, q, *, start="", end="", frequency="D"):
            return lt_dataset

    provider = lt_stats.StatisticalAnalysisTools(_Backend())
    tools = {t.name: t for t in provider.as_tools()}
    lt = json.loads(tools["statistical_return_volatility"].run_sync(
        instruments="ticker:SPY,ticker:TLT", frequency="W"))

    ds = ReturnDataset(instruments=instruments, rows=rows)
    ls = return_volatility(ds, frequency="W")
    assert lt["payload"]["volatility"] == ls["volatility"]
