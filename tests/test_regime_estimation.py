"""The boundary between the hub's returns and the regime pipeline.

The tests that need the hub are marked and skipped where it is absent, in the
same way the rest of this suite treats the private package. The contract tests
below it need nothing and always run: they guard identities that already have
1,809 rows and 800,338 series points depending on them.
"""
from __future__ import annotations

import pytest

from lazystats.regimes.estimation import (
    PRODUCED_BY,
    PROVENANCE_SOURCE,
    SymbolReturns,
    is_production,
    symbol_returns,
)
from lazystats.regimes.series import series_key

PROD = "C:/data/market_data.duckdb"


class TestDepotIdentitiesAreFrozen:
    """Values the depot already contains. Changing one splits a series in two
    without failing: new rows land under a name nothing selects."""

    def test_the_producer_identity_is_the_one_in_the_depot(self):
        assert PRODUCED_BY == "scheduled:run_regime_daily"

    def test_the_producer_identity_names_the_job_not_the_module(self):
        """It has to survive the code moving between repositories, which is
        exactly what is happening to this module."""
        assert "lazystats" not in PRODUCED_BY
        assert "market_data_hub" not in PRODUCED_BY

    def test_provenance_names_where_the_code_now_lives(self):
        """Unlike the producer identity, this one should move with the code:
        provenance that names the wrong module is worse than a discontinuity."""
        assert PROVENANCE_SOURCE == "lazystats.regimes.estimation"


class TestProductionIsDeclaredNotGuessed:
    def test_the_same_database_is_production(self):
        assert is_production(PROD, PROD)

    def test_a_different_database_is_not(self):
        assert not is_production("C:/data/staging/market_data.duckdb", PROD)

    def test_a_deployment_cannot_declare_itself_production_by_accident(self):
        """The defect the original guarded against: comparing the resolved path
        with itself always agrees, and the namespace protecting production's
        vintages quietly disappears."""
        staging = "C:/data/staging/market_data.duckdb"
        assert not is_production(staging, PROD)
        assert series_key("GLD", market_db=staging, production_db=PROD) != "regime:GLD"


class TestReturnsBoundary:
    """Needs the private market-data-hub package, like the contract tests."""

    @staticmethod
    def _hub_or_skip():
        pytest.importorskip("market_data_hub")

    def test_the_symbol_that_comes_out_is_the_one_the_depot_keys_on(self):
        self._hub_or_skip()
        got = symbol_returns("ticker:GLD", start="2024-01-01", end="2024-03-31")
        assert got.symbol == "GLD"
        assert series_key(got.symbol, market_db=PROD, production_db=PROD) == "regime:GLD"

    def test_a_bare_symbol_and_a_canonical_id_agree_exactly(self):
        self._hub_or_skip()
        bare = symbol_returns("GLD", start="2024-01-01", end="2024-03-31")
        canonical = symbol_returns("ticker:GLD", start="2024-01-01", end="2024-03-31")
        assert bare == canonical

    def test_gaps_are_dropped_so_the_series_is_dense(self):
        self._hub_or_skip()
        got = symbol_returns("GLD", start="2024-01-01", end="2024-03-31")
        assert len(got.dates) == len(got.values)
        assert all(v is not None for v in got.values)
        assert list(got.dates) == sorted(got.dates)

    def test_the_values_match_the_hub_exactly(self):
        """The move must not change a single number: load_returns is a wrapper
        over the same extract_returns the previous pipeline called."""
        self._hub_or_skip()
        from market_data_hub.extract import extract_returns

        got = symbol_returns("GLD", start="2024-01-01", end="2024-03-31")
        frame, _ = extract_returns(["GLD"], start="2024-01-01", end="2024-03-31", frequency="D")
        column = frame.columns[0]
        expected = {str(i.date() if hasattr(i, "date") else i): float(v)
                    for i, v in frame[column].items()}

        assert len(got.dates) == len(expected)
        for date, value in zip(got.dates, got.values, strict=True):
            assert date in expected
            assert value == pytest.approx(expected[date], rel=1e-12, abs=1e-15)

    def test_an_unknown_symbol_refuses_rather_than_fitting_nothing(self):
        self._hub_or_skip()
        with pytest.raises(ValueError):
            symbol_returns("NOT_A_REAL_SYMBOL_XYZ", start="2024-01-01", end="2024-03-31")


class TestSymbolReturnsShape:
    def test_length_is_the_number_of_observations(self):
        r = SymbolReturns(symbol="GLD", dates=("2024-01-02",), values=(0.01,))
        assert len(r) == 1
