# -*- coding: utf-8 -*-
"""Proof of adoption: LazyHMM emits its regime output as lazydatacore
``AnalysisResult`` envelopes, identified by canonical ``InstrumentId``s. Requires
the optional ``market-data-hub`` extra; skipped (like the other datahub tests'
spirit) when it is absent, since the contract models live there."""
from __future__ import annotations

import pytest

pytest.importorskip("market_data_hub.lazydatacore")

from market_data_hub.lazydatacore import (  # noqa: E402
    AnalysisResult,
    Domain,
    InstrumentId,
    ResultKind,
)

from lazystats.regimes.contract import to_analysis_results  # noqa: E402


def _fit_output() -> dict:
    """A minimal fit_regimes-shaped output with two series (panel model)."""
    return {
        "model": "panel",
        "criterion": "bic",
        "n_timesteps": 3,
        "series": {
            "SPY": {
                "S": 2,
                "labels": ["calm", "stress"],
                "states": [0, 0, 1],
                "high_vol_flag": [0, 0, 1],
                "prob_high_vol": [0.1, 0.2, 0.83],
                "bic": -123.4,
                "loglik": 56.7,
            },
            "ticker:TLT": {  # already-namespaced series name passes through
                "S": 2,
                "labels": ["calm", "stress"],
                "states": [1, 1, 0],
                "high_vol_flag": [1, 1, 0],
                "prob_high_vol": [0.7, 0.6, 0.05],
                "bic": -99.9,
                "loglik": 42.0,
            },
        },
    }


def test_emits_one_analysis_result_per_series() -> None:
    results = to_analysis_results(_fit_output(), tool_version="0.2.0")
    assert len(results) == 2
    assert all(isinstance(r, AnalysisResult) for r in results)
    assert all(r.kind is ResultKind.SIGNAL for r in results)
    assert all(r.produced_by == "lazyhmm.regime.v1" for r in results)


def test_identity_is_canonical_instrument_id() -> None:
    by_id = {str(r.instruments[0]): r for r in to_analysis_results(_fit_output())}
    # bare symbol -> ticker: ; already-namespaced name passes through
    assert "ticker:SPY" in by_id
    assert "ticker:TLT" in by_id
    assert by_id["ticker:SPY"].instruments[0].domain is Domain.TICKER


def test_payload_is_the_current_regime_signal() -> None:
    spy = next(
        r for r in to_analysis_results(_fit_output())
        if str(r.instruments[0]) == "ticker:SPY"
    )
    assert spy.payload["current_state"] == 1
    assert spy.payload["current_label"] == "stress"
    assert spy.payload["prob_high_vol"] == pytest.approx(0.83)
    assert spy.payload["high_vol"] is True
    assert spy.payload["model"] == "panel"
    assert spy.payload["n_timesteps"] == 3


def test_provenance_and_json_roundtrip() -> None:
    [r, _] = to_analysis_results(_fit_output(), source="lazyhmm", tool_version="0.2.0")
    assert r.provenance is not None
    assert r.provenance.source.source == "lazyhmm"
    assert r.provenance.tool_version == "0.2.0"
    # the whole envelope round-trips through JSON with the id as a plain string
    dumped = r.model_dump(mode="json")
    assert isinstance(dumped["instruments"][0], str)
    again = AnalysisResult.model_validate(dumped)
    assert again.instruments[0] == InstrumentId.parse(dumped["instruments"][0])


def test_accepts_compact_fit_regimes_public_return() -> None:
    # Mirrors the COMPACT shape fit_regimes() returns to callers: states /
    # prob_high_vol / high_vol_flag are stripped and current_* + prob_high_vol_now
    # are added (lazyhmm/tools.py). The current signal must still be populated.
    compact = {
        "result_key": "rk",
        "model": "panel",
        "n_timesteps": 3,
        "series": {
            "SPY": {
                "S": 2,
                "labels": ["calm", "stress"],
                "bic": -123.4,
                "loglik": 56.7,
                "current_state": 1,
                "current_label": "stress",
                "prob_high_vol_now": 0.83,
            },
        },
    }
    [r] = to_analysis_results(compact)
    assert r.payload["current_state"] == 1
    assert r.payload["current_label"] == "stress"
    assert r.payload["prob_high_vol"] == pytest.approx(0.83)
    assert r.payload["high_vol"] is True  # state 1 == S-1 (vol-ascending)
    assert str(r.instruments[0]) == "ticker:SPY"


def test_end_to_end_from_real_fit_regimes() -> None:
    # The natural usage — to_analysis_results(fit_regimes(...)) on the public
    # return — yields a populated signal, guarding against the compact-shape drift.
    import numpy as np

    from lazystats.regimes import fit_regimes

    rng = np.random.RandomState(0)
    calm = rng.normal(0.0, 1.0, 30)
    stress = rng.normal(0.0, 4.0, 30)
    data = np.concatenate([calm, stress]).reshape(-1, 1).tolist()

    out = fit_regimes(data, series_names=["SPY"], model="panel")
    [r] = to_analysis_results(out)
    assert r.payload["current_state"] is not None
    assert r.payload["current_label"] is not None
    assert r.payload["prob_high_vol"] is not None
    assert str(r.instruments[0]) == "ticker:SPY"


def test_empty_series_yields_no_results() -> None:
    assert to_analysis_results({"model": "panel", "series": {}}) == []
