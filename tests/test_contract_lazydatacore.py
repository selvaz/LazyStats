"""Consumer-side contract test against market-data-hub's lazydatacore fixtures
(ecosystem stabilization plan, ECO-010 / Train B PR B3).

market-data-hub validates contracts/v1/*.json in its OWN test suite
(producer side); this file is the consumer side — it proves lazystats can
actually consume the fixture, not merely that the fixture parses.

Opt-in and skip-safe by design (matches lazystats' existing "market-data-hub
is optional, lazily imported" posture — see pyproject.toml's `contract`
extra):
  - skips if `market_data_hub` isn't installed (default test/dev extras
    don't pull it in; install the `contract` extra to run these for real).
  - skips if the fixture directory isn't available. Fixtures live in
    market-data-hub's repo root (contracts/v1/), NOT inside the installed
    package (only market_data_hub/ ships in the wheel) -- so a checkout is
    needed. Point MDH_CONTRACTS_DIR at one, e.g.:
        git clone --depth 1 https://github.com/selvaz/market-data-hub /tmp/mdh
        git -C /tmp/mdh checkout 0eb286e4e931c089ae9d4383f5f6f615d9b4d5e4
        MDH_CONTRACTS_DIR=/tmp/mdh/contracts/v1 pytest tests/test_contract_lazydatacore.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

market_data_hub = pytest.importorskip("market_data_hub")
lazydatacore = pytest.importorskip("market_data_hub.lazydatacore")

from lazystats.core.returns import (  # noqa: E402
    return_correlation,
    return_outliers,
    return_volatility,
)
from lazystats.models import ReturnDataset  # noqa: E402

_CONTRACTS_DIR = os.environ.get("MDH_CONTRACTS_DIR")
if not _CONTRACTS_DIR or not Path(_CONTRACTS_DIR).is_dir():
    pytest.skip(
        "MDH_CONTRACTS_DIR not set or not a directory -- point it at a "
        "market-data-hub checkout's contracts/v1/ (see module docstring)",
        allow_module_level=True,
    )

CONTRACTS_V1 = Path(_CONTRACTS_DIR)


def _load(name: str):
    return json.loads((CONTRACTS_V1 / name).read_text(encoding="utf-8"))


def test_analysis_result_fixture_is_schema_v1():
    raw = _load("analysis_result.json")
    res = lazydatacore.AnalysisResult.model_validate(raw)
    assert res.schema_version == lazydatacore.SCHEMA_VERSION == "1.0"
    assert res.producer == "market-data-hub"


def test_provenance_fixture_validates():
    raw = _load("provenance.json")
    prov = lazydatacore.Provenance.model_validate(raw)
    assert prov.source.source


def test_instrument_id_fixture_validates_every_entry():
    raw = _load("instrument_id.json")
    for canonical in raw:
        iid = lazydatacore.InstrumentId.model_validate(canonical)
        assert str(iid) == canonical


def _return_dataset_from_fixture() -> ReturnDataset:
    raw = _load("return_series.json")
    instruments = raw["instruments"]          # canonical ids, e.g. "ticker:AAPL"
    symbols = raw["symbols"]                   # bare column keys, e.g. "AAPL"
    # The hub's return frame is keyed by bare symbol; the consumer relabels
    # bare -> canonical when building a ReturnDataset. This mirrors exactly
    # what lazystats.io.datahub.load_returns does, so we exercise the real
    # consumption path rather than assuming the fixture is already canonical.
    bare_to_canonical = dict(zip(symbols, instruments, strict=True))
    rows = [
        {"date": row["date"], **{bare_to_canonical[sym]: row[sym] for sym in symbols}}
        for row in raw["rows"]
    ]
    return ReturnDataset(instruments=instruments, rows=rows, metadata=raw.get("metadata", {}))


def test_return_series_fixture_constructs_a_real_return_dataset():
    dataset = _return_dataset_from_fixture()
    assert dataset.instruments == ["ticker:AAPL", "ticker:MSFT"]
    assert len(dataset.rows) == 4
    # Column keys are canonical instrument ids, matching what
    # lazystats.io.datahub.load_returns actually produces (not bare symbols)
    # -- this is exactly the shape mismatch a prior version of this fixture
    # got wrong, caught only by a genuine consumer round-trip like this one.
    for row in dataset.rows:
        assert set(row) - {"date"} == set(dataset.instruments)


def test_return_series_fixture_is_consumable_by_core_statistics():
    """Not just shape-compatible: the fixture must survive lazystats' own
    statistics pipeline without error, on real (non-mocked) data."""
    dataset = _return_dataset_from_fixture()

    vol = return_volatility(dataset, frequency="W")
    assert set(vol["volatility"]) == set(dataset.instruments)
    for stats in vol["volatility"].values():
        assert stats["observations"] == 4
        assert stats["annualized_volatility"] is not None

    corr = return_correlation(dataset, frequency="W")
    assert corr["correlation"]["ticker:AAPL"]["ticker:AAPL"] == pytest.approx(1.0)

    outliers = return_outliers(dataset, frequency="W", threshold=2.0)
    assert isinstance(outliers, dict)
