# -*- coding: utf-8 -*-
"""The anomaly gate's configuration contract.

Mostly about the incoherent configurations, because those are the dangerous
ones. A wrong type fails loudly on first use; an inverted band does not fail
at all — it silently reverses what the gate considers unusual, and every
downstream step keeps working while investigating the wrong days.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lazystats.anomaly_gate_config import GateConfigError, load_gate_config

VALID = """
upstream_produced_by = "scheduled:etf_daily_stats"
vol_ratio_high = 1.5
vol_ratio_low = 0.6666666666666666
vol_ratio_delta_min = 0.15
corr_high = 0.7
corr_low = 0.15
corr_delta_min = 0.20
max_corr_shifts_per_day = 8
beta_benchmark = "ticker:SPY"
beta_z_threshold = 2.0
beta_z_delta_min = 1.0
dedup_lookback = 20
"""


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "gate.toml"
    p.write_text(body, encoding="utf-8")
    return p


class TestValid:
    def test_loads_every_field(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID))
        assert cfg.vol_ratio_high == 1.5
        assert cfg.max_corr_shifts_per_day == 8
        assert cfg.beta_benchmark == "ticker:SPY"
        assert cfg.upstream_produced_by == "scheduled:etf_daily_stats"

    def test_integers_are_accepted_where_a_number_is_wanted(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID.replace("2.0", "2")))
        assert cfg.beta_z_threshold == 2.0

    def test_the_config_is_immutable(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID))
        with pytest.raises(Exception):
            cfg.corr_high = 0.9  # type: ignore[misc]

    def test_provenance_covers_every_field(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID))
        prov = cfg.as_provenance()
        assert set(prov) == set(vars(cfg)), "a field absent from provenance is unrecorded"


class TestMissingOrWrongType:
    @pytest.mark.parametrize("key", [
        "upstream_produced_by", "vol_ratio_high", "vol_ratio_low", "vol_ratio_delta_min",
        "corr_high", "corr_low", "corr_delta_min", "max_corr_shifts_per_day",
        "beta_benchmark", "beta_z_threshold", "beta_z_delta_min", "dedup_lookback",
    ])
    def test_every_key_is_required(self, tmp_path, key):
        body = "\n".join(ln for ln in VALID.splitlines() if not ln.startswith(f"{key} ="))
        with pytest.raises(GateConfigError, match=key):
            load_gate_config(write(tmp_path, body))

    def test_a_boolean_is_not_a_number(self, tmp_path):
        """bool subclasses int in Python; `true` must not become 1.0."""
        with pytest.raises(GateConfigError):
            load_gate_config(write(tmp_path, VALID.replace("corr_high = 0.7", "corr_high = true")))

    def test_a_float_is_not_an_integer_count(self, tmp_path):
        with pytest.raises(GateConfigError, match="max_corr_shifts_per_day"):
            load_gate_config(write(tmp_path, VALID.replace(
                "max_corr_shifts_per_day = 8", "max_corr_shifts_per_day = 8.5")))

    def test_a_blank_benchmark_is_refused(self, tmp_path):
        with pytest.raises(GateConfigError, match="beta_benchmark"):
            load_gate_config(write(tmp_path, VALID.replace('"ticker:SPY"', '"  "')))

    def test_surrounding_whitespace_is_refused_not_trimmed(self, tmp_path):
        """Trimming would hide a typo that changes which series is read."""
        with pytest.raises(GateConfigError, match="whitespace"):
            load_gate_config(write(tmp_path, VALID.replace(
                '"scheduled:etf_daily_stats"', '" scheduled:etf_daily_stats"')))

    def test_a_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(GateConfigError, match="not found"):
            load_gate_config(tmp_path / "absent.toml")

    def test_malformed_toml_names_the_file(self, tmp_path):
        with pytest.raises(GateConfigError, match="not valid TOML"):
            load_gate_config(write(tmp_path, "vol_ratio_high = [unclosed"))


class TestIncoherentBands:
    """These are the configurations that would not fail on their own.

    An inverted band reverses the gate's meaning while every downstream step
    keeps running: the investigation would look healthy and cover the wrong
    days.
    """

    def test_an_inverted_volatility_band_is_refused(self, tmp_path):
        body = VALID.replace("vol_ratio_high = 1.5", "vol_ratio_high = 0.5")
        with pytest.raises(GateConfigError, match="must be below"):
            load_gate_config(write(tmp_path, body))

    def test_an_inverted_correlation_band_is_refused(self, tmp_path):
        body = VALID.replace("corr_low = 0.15", "corr_low = 0.9")
        with pytest.raises(GateConfigError, match="must be below"):
            load_gate_config(write(tmp_path, body))

    def test_equal_band_edges_are_refused(self, tmp_path):
        """A zero-width band admits nothing; the gate would find nothing, every
        day, and look like a quiet market."""
        body = VALID.replace("corr_low = 0.15", "corr_low = 0.7")
        with pytest.raises(GateConfigError, match="must be below"):
            load_gate_config(write(tmp_path, body))

    def test_a_correlation_band_outside_minus_one_to_one_is_refused(self, tmp_path):
        """Correlations cannot exceed 1: such a band can never trigger."""
        body = VALID.replace("corr_high = 0.7", "corr_high = 1.5")
        with pytest.raises(GateConfigError, match=r"\[-1, 1\]"):
            load_gate_config(write(tmp_path, body))

    def test_a_non_positive_volatility_ratio_is_refused(self, tmp_path):
        """A ratio of volatilities cannot be zero or negative."""
        body = VALID.replace("vol_ratio_low = 0.6666666666666666", "vol_ratio_low = 0")
        with pytest.raises(GateConfigError):
            load_gate_config(write(tmp_path, body))

    @pytest.mark.parametrize("key", ["vol_ratio_delta_min", "corr_delta_min",
                                     "beta_z_threshold", "beta_z_delta_min"])
    def test_a_non_positive_delta_is_refused(self, tmp_path, key):
        """A zero minimum delta removes the freshness requirement entirely,
        leaving the band alone to over-trigger."""
        import re
        body = re.sub(rf"^{key} = .*$", f"{key} = 0", VALID, flags=re.M)
        with pytest.raises(GateConfigError, match="must be positive"):
            load_gate_config(write(tmp_path, body))

    def test_a_non_positive_lookback_is_refused(self, tmp_path):
        with pytest.raises(GateConfigError, match="must be positive"):
            load_gate_config(write(tmp_path, VALID.replace("dedup_lookback = 20", "dedup_lookback = 0")))
