"""The anomaly gate's configuration contract.

Mostly about the incoherent configurations, because those are the dangerous
ones. A wrong type fails loudly on first use; an inverted band does not fail
at all — it silently reverses what the gate considers unusual, and every
downstream step keeps working while investigating the wrong days.
"""
from __future__ import annotations

import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lazystats.anomaly_gate_config import GateConfigError, load_gate_config

#: A coherent configuration with values chosen to be readable, not to
#: reflect anyone's judgement about a portfolio. No real universe, producer
#: or threshold appears in this repository's tests.
VALID = """
upstream_series_key = "example_daily_stats"
upstream_produced_by = "scheduled:example_daily_stats"
vol_ratio_high = 2.0
vol_ratio_low = 0.5
vol_ratio_delta_min = 0.25
corr_high = 0.8
corr_low = 0.1
corr_delta_min = 0.30
max_corr_shifts_per_day = 5
beta_benchmark = "ticker:EXAMPLE"
beta_z_threshold = 3.0
beta_z_delta_min = 1.5
dedup_lookback = 10
"""


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "gate.toml"
    p.write_text(body, encoding="utf-8")
    return p


def mutate(**changes: str) -> str:
    """VALID with some keys given different values.

    By key rather than by literal text: a test that searches for the old
    value stops testing anything the moment the fixture's numbers change,
    and does so silently — the replace simply finds nothing.
    """
    body = VALID
    for key, value in changes.items():
        body, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", body, flags=re.M)
        assert n == 1, f"{key} is not in the fixture"
    return body


class TestValid:
    def test_loads_every_field(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID))
        assert cfg.vol_ratio_high == 2.0
        assert cfg.max_corr_shifts_per_day == 5
        assert cfg.beta_benchmark == "ticker:EXAMPLE"
        assert cfg.upstream_series_key == "example_daily_stats"
        assert cfg.upstream_produced_by == "scheduled:example_daily_stats"

    def test_integers_are_accepted_where_a_number_is_wanted(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID.replace("3.0", "3")))
        assert cfg.beta_z_threshold == 3.0

    def test_the_config_is_immutable(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID))
        with pytest.raises(FrozenInstanceError):
            cfg.corr_high = 0.9  # type: ignore[misc]

    def test_provenance_covers_every_field(self, tmp_path):
        cfg = load_gate_config(write(tmp_path, VALID))
        prov = cfg.as_provenance()
        assert set(prov) == set(vars(cfg)), "a field absent from provenance is unrecorded"


class TestMissingOrWrongType:
    @pytest.mark.parametrize("key", [
        "upstream_series_key", "upstream_produced_by", "vol_ratio_high",
        "vol_ratio_low", "vol_ratio_delta_min",
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
            load_gate_config(write(tmp_path, mutate(corr_high="true")))

    def test_a_float_is_not_an_integer_count(self, tmp_path):
        with pytest.raises(GateConfigError, match="max_corr_shifts_per_day"):
            load_gate_config(write(tmp_path, mutate(max_corr_shifts_per_day="8.5")))

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_a_non_finite_threshold_is_refused(self, tmp_path, value):
        with pytest.raises(GateConfigError, match="finite"):
            load_gate_config(write(tmp_path, mutate(beta_z_threshold=value)))

    def test_a_blank_benchmark_is_refused(self, tmp_path):
        with pytest.raises(GateConfigError, match="beta_benchmark"):
            load_gate_config(write(tmp_path, mutate(beta_benchmark='"  "')))

    def test_surrounding_whitespace_is_refused_not_trimmed(self, tmp_path):
        """Trimming would hide a typo that changes which series is read."""
        with pytest.raises(GateConfigError, match="whitespace"):
            load_gate_config(write(tmp_path, mutate(
                upstream_produced_by='" scheduled:example_daily_stats"')))

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
        body = mutate(vol_ratio_high="0.4")
        with pytest.raises(GateConfigError, match="must be below"):
            load_gate_config(write(tmp_path, body))

    def test_an_inverted_correlation_band_is_refused(self, tmp_path):
        body = mutate(corr_low="0.9")
        with pytest.raises(GateConfigError, match="must be below"):
            load_gate_config(write(tmp_path, body))

    def test_equal_band_edges_are_refused(self, tmp_path):
        """A zero-width band admits nothing; the gate would find nothing, every
        day, and look like a quiet market."""
        body = mutate(corr_low="0.8")
        with pytest.raises(GateConfigError, match="must be below"):
            load_gate_config(write(tmp_path, body))

    def test_a_correlation_band_outside_minus_one_to_one_is_refused(self, tmp_path):
        """Correlations cannot exceed 1: such a band can never trigger."""
        body = mutate(corr_high="1.5")
        with pytest.raises(GateConfigError, match=r"\[-1, 1\]"):
            load_gate_config(write(tmp_path, body))

    def test_a_non_positive_volatility_ratio_is_refused(self, tmp_path):
        """A ratio of volatilities cannot be zero or negative."""
        body = mutate(vol_ratio_low="0")
        with pytest.raises(GateConfigError):
            load_gate_config(write(tmp_path, body))

    @pytest.mark.parametrize("key", ["vol_ratio_delta_min", "corr_delta_min",
                                     "beta_z_threshold", "beta_z_delta_min"])
    def test_a_non_positive_delta_is_refused(self, tmp_path, key):
        """A zero minimum delta removes the freshness requirement entirely,
        leaving the band alone to over-trigger."""
        body = mutate(**{key: "0"})
        with pytest.raises(GateConfigError, match="must be positive"):
            load_gate_config(write(tmp_path, body))

    def test_a_non_positive_lookback_is_refused(self, tmp_path):
        with pytest.raises(GateConfigError, match="must be positive"):
            load_gate_config(write(tmp_path, mutate(dedup_lookback="0")))
