"""The two ways a regime series can silently lose its history.

Neither failure raises. A regime run whose key is wrong writes a perfectly valid
new series beside the old one and reports success, so these are the assertions
that have to hold rather than the exceptions that have to be caught.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lazystats.regimes.series import bare_symbol, series_key

PROD = Path("C:/data/market_data.duckdb")
STAGING = Path("C:/data/staging/market_data.duckdb")


class TestKeysStayBare:
    """``io.datahub`` labels instruments canonically; the depot does not."""

    def test_the_production_key_is_the_migrated_format(self):
        """405,313 rows were written as ``regime:<symbol>``. That is the format."""
        assert series_key("GLD", market_db=PROD, production_db=PROD) == "regime:GLD"

    def test_a_canonical_id_produces_the_same_key_as_a_bare_symbol(self):
        """The defect this file exists for: taking the instrument straight from
        a ReturnDataset would give ``regime:ticker:GLD``, orphaning the history
        without failing."""
        canonical = series_key("ticker:GLD", market_db=PROD, production_db=PROD)
        assert canonical == "regime:GLD"
        assert "ticker:" not in canonical

    @pytest.mark.parametrize("instrument", ["GLD", "ticker:GLD", "  ticker:GLD  "])
    def test_every_accepted_spelling_agrees(self, instrument):
        assert bare_symbol(instrument) == "GLD"

    def test_an_unknown_domain_is_refused_not_stripped(self):
        """Stripping it would build a key that looks right and names another
        instrument's history."""
        with pytest.raises(ValueError, match="unsupported instrument domain"):
            bare_symbol("isin:IE00B5BMR087")

    @pytest.mark.parametrize("bad", ["", "   ", "ticker:"])
    def test_empty_ids_are_refused(self, bad):
        with pytest.raises(ValueError):
            bare_symbol(bad)


class TestStagingCannotSupersedeProduction:
    """One depot, no per-database isolation. The key is the only separation."""

    def test_production_writes_the_unqualified_series(self):
        assert series_key("GLD", market_db=PROD, production_db=PROD) == "regime:GLD"

    def test_another_database_is_namespaced(self):
        key = series_key("GLD", market_db=STAGING, production_db=PROD)
        assert key != "regime:GLD"
        assert key.startswith("regime:GLD@")

    def test_two_non_production_databases_do_not_collide(self):
        other = Path("C:/data/other/market_data.duckdb")
        assert series_key("GLD", market_db=STAGING, production_db=PROD) != \
               series_key("GLD", market_db=other, production_db=PROD)

    def test_the_same_database_spelled_differently_is_still_production(self):
        """A staging deployment that passed its own path as production would
        bypass the namespace; the same path written two ways must not."""
        assert series_key("GLD", market_db="C:/data/./market_data.duckdb",
                          production_db=PROD) == "regime:GLD"


class TestVariantsDoNotUpsertEachOther:
    """A shorter-window fit is a different series, not a newer reading."""

    def test_a_variant_is_a_distinct_series(self):
        full = series_key("GLD", market_db=PROD, production_db=PROD)
        windowed = series_key("GLD", market_db=PROD, production_db=PROD, variant="8y")
        assert windowed != full
        assert windowed == "regime:GLD:8y"

    def test_two_variants_are_distinct_from_each_other(self):
        a = series_key("GLD", market_db=PROD, production_db=PROD, variant="3y")
        b = series_key("GLD", market_db=PROD, production_db=PROD, variant="10y")
        assert a != b

    def test_a_variant_on_a_non_production_database_keeps_both_qualifiers(self):
        key = series_key("GLD", market_db=STAGING, production_db=PROD, variant="8y")
        assert key.startswith("regime:GLD@")
        assert key.endswith(":8y")
