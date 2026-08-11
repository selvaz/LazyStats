# -*- coding: utf-8 -*-
"""The ETF daily-stats configuration contract.

The point of these tests is not that TOML parses — it is that a
misconfigured scheduled run fails loudly, at load time, with a message
naming the file and the key, instead of quietly analysing the wrong
universe.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lazystats.etf_stats import ConfigError, load_config

VALID = """
instruments = ["SPY", "TLT"]
short_weeks = 13
long_weeks = 104
one_year_weeks = 52
daily_lookback_days = 400
outlier_window_days = 5
outlier_chart_days = 21
outlier_threshold = 2.0
series_key = "test_series"
return_horizons = [
    { label = "1M", days = 30 },
    { label = "YTD" },
]
"""


def write(tmp_path: Path, body: str, name: str = "cfg.toml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


class TestValidConfig:
    def test_loads_every_field(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))
        assert cfg.instruments == ("SPY", "TLT")
        assert cfg.short_weeks == 13
        assert cfg.long_weeks == 104
        assert cfg.outlier_threshold == 2.0
        assert cfg.series_key == "test_series"

    def test_ytd_horizon_carries_no_day_count(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))
        ytd = [h for h in cfg.return_horizons if h.label == "YTD"]
        assert len(ytd) == 1
        assert ytd[0].days_back is None, "YTD resolves against the prior year-end"

    def test_integer_threshold_is_accepted_as_float(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID.replace("2.0", "2")))
        assert cfg.outlier_threshold == 2.0

    def test_config_is_immutable(self, tmp_path):
        """A run must not be able to mutate its own preset halfway through."""
        cfg = load_config(write(tmp_path, VALID))
        with pytest.raises(Exception):
            cfg.short_weeks = 99  # type: ignore[misc]

    def test_provenance_block_matches_the_fields(self, tmp_path):
        prov = load_config(write(tmp_path, VALID)).as_provenance()
        assert prov["short_window_weeks"] == 13
        assert prov["outlier_threshold"] == 2.0
        assert prov["return_horizons"] == ["1M", "YTD"]


class TestRefusesToGuess:
    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.toml")

    def test_malformed_toml_names_the_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not valid TOML"):
            load_config(write(tmp_path, "instruments = [unclosed"))

    @pytest.mark.parametrize(
        "key",
        ["instruments", "short_weeks", "long_weeks", "one_year_weeks",
         "daily_lookback_days", "outlier_window_days", "outlier_chart_days",
         "series_key", "return_horizons"],
    )
    def test_every_required_key_is_required(self, tmp_path, key):
        body = "\n".join(ln for ln in VALID.splitlines() if not ln.startswith(f"{key} ="))
        if key == "return_horizons":
            body = VALID[: VALID.index("return_horizons")]
        with pytest.raises(ConfigError, match=key):
            load_config(write(tmp_path, body))

    def test_empty_instrument_list_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="nothing to analyse"):
            load_config(write(tmp_path, VALID.replace('["SPY", "TLT"]', "[]")))

    def test_duplicate_instruments_are_refused(self, tmp_path):
        """A duplicate would silently double-weight one instrument."""
        with pytest.raises(ConfigError, match="duplicates"):
            load_config(write(tmp_path, VALID.replace('["SPY", "TLT"]', '["SPY", "TLT", "SPY"]')))

    def test_short_window_must_be_shorter_than_long(self, tmp_path):
        """The report contrasts the two; inverted windows would invert its meaning."""
        with pytest.raises(ConfigError, match="shorter than"):
            load_config(write(tmp_path, VALID.replace("short_weeks = 13", "short_weeks = 200")))

    def test_negative_window_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="must be positive"):
            load_config(write(tmp_path, VALID.replace("short_weeks = 13", "short_weeks = -1")))

    def test_boolean_is_not_accepted_as_a_number(self, tmp_path):
        """bool subclasses int in Python; a threshold of `true` must not pass as 1."""
        with pytest.raises(ConfigError):
            load_config(write(tmp_path, VALID.replace("outlier_threshold = 2.0", "outlier_threshold = true")))

    def test_horizon_without_days_is_refused_unless_ytd(self, tmp_path):
        body = VALID.replace('{ label = "1M", days = 30 },', '{ label = "1M" },')
        with pytest.raises(ConfigError, match="positive integer"):
            load_config(write(tmp_path, body))

    def test_blank_series_key_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="series_key"):
            load_config(write(tmp_path, VALID.replace('"test_series"', '"   "')))


class TestRefusesToGuessHorizons:
    def test_duplicate_horizon_label_is_refused(self, tmp_path):
        """Two columns with one label would make the report ambiguous."""
        body = VALID.replace('{ label = "YTD" },', '{ label = "1M", days = 90 },')
        with pytest.raises(ConfigError, match="duplicate horizon label"):
            load_config(write(tmp_path, body))

    def test_non_string_horizon_label_is_refused(self, tmp_path):
        body = VALID.replace('{ label = "1M", days = 30 },', '{ label = 30, days = 30 },')
        with pytest.raises(ConfigError, match="non-empty string"):
            load_config(write(tmp_path, body))

    def test_blank_horizon_label_is_refused(self, tmp_path):
        body = VALID.replace('{ label = "1M", days = 30 },', '{ label = "  ", days = 30 },')
        with pytest.raises(ConfigError, match="non-empty string"):
            load_config(write(tmp_path, body))

    def test_ytd_with_days_is_refused_not_ignored(self, tmp_path):
        """Silently dropping it would make the report disagree with its config."""
        body = VALID.replace('{ label = "YTD" },', '{ label = "YTD", days = 200 },')
        with pytest.raises(ConfigError, match="must not carry 'days'"):
            load_config(write(tmp_path, body))

    def test_untrimmed_instrument_is_refused_not_normalised(self, tmp_path):
        """Stripping would turn a typo into a silent duplicate of another entry."""
        body = VALID.replace('["SPY", "TLT"]', '["SPY", " TLT"]')
        with pytest.raises(ConfigError, match="whitespace"):
            load_config(write(tmp_path, body))


class TestShippedExample:
    def test_the_example_config_actually_loads(self):
        """A broken example is worse than none: it is what a new user copies."""
        example = Path(__file__).resolve().parents[1] / "examples" / "etf_daily_stats.example.toml"
        cfg = load_config(example)
        assert cfg.instruments
        assert cfg.series_key != "etf_daily_stats", (
            "the example must not reuse the live series_key, or an example run "
            "would write into the real series"
        )
