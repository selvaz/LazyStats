"""The regime run's configuration contract.

The point is not that TOML parses. It is that a misconfigured scheduled run
fails loudly, at load time, instead of quietly fitting the wrong universe or
writing two windows into one series.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lazystats.regimes.config import ConfigError, load_config

VALID = """
instruments = ["SPY", "TLT"]
s_max = 3
n_starts = 20
random_state = 123
retro_days = 30

[windows.full]

[windows.8y]
lookback_years = 8

[comparisons.full_vs_8y]
baseline = "full"
candidate = "8y"
"""


def write(tmp_path: Path, body: str, name: str = "regimes.toml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


class TestValidConfig:
    def test_loads_every_field(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))
        assert cfg.instruments == ("SPY", "TLT")
        assert cfg.s_max == 3
        assert cfg.n_starts == 20
        assert cfg.random_state == 123
        assert cfg.retro_days == 30

    def test_the_unrestricted_window_carries_no_variant(self, tmp_path):
        """That is what keeps its estimates in the series the migrated history
        already lives in, rather than starting a parallel one."""
        cfg = load_config(write(tmp_path, VALID))
        assert cfg.window("full").variant is None

    def test_a_bounded_window_is_tagged_by_its_length(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))
        assert cfg.window("8y").variant == "8y"

    def test_comparisons_name_declared_windows(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID))
        assert len(cfg.comparisons) == 1
        assert (cfg.comparisons[0].baseline, cfg.comparisons[0].candidate) == ("full", "8y")


class TestWindowsAreGeneric:
    """The method has no opinion about eight years."""

    def test_any_two_bounded_windows_can_be_compared(self, tmp_path):
        body = """
instruments = ["SPY"]
s_max = 3
n_starts = 20
random_state = 123
retro_days = 30
[windows.3y]
lookback_years = 3
[windows.10y]
lookback_years = 10
[comparisons.short_vs_long]
baseline = "3y"
candidate = "10y"
"""
        cfg = load_config(write(tmp_path, body))
        assert {w.name for w in cfg.windows} == {"3y", "10y"}
        assert cfg.comparisons[0].baseline == "3y"

    def test_a_config_needs_no_comparison_at_all(self, tmp_path):
        """Fitting windows without contrasting them is a legitimate preset."""
        body = VALID[: VALID.index("[comparisons")]
        cfg = load_config(write(tmp_path, body))
        assert cfg.comparisons == ()


class TestRefusesToGuess:
    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.toml")

    def test_malformed_toml_names_the_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not valid TOML"):
            load_config(write(tmp_path, "instruments = [unclosed"))

    @pytest.mark.parametrize("key", ["instruments", "s_max", "n_starts",
                                     "random_state", "retro_days"])
    def test_every_required_key_is_required(self, tmp_path, key):
        body = "\n".join(ln for ln in VALID.splitlines() if not ln.startswith(f"{key} ="))
        with pytest.raises(ConfigError, match=key):
            load_config(write(tmp_path, body))

    def test_an_empty_universe_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="nothing to fit"):
            load_config(write(tmp_path, VALID.replace('["SPY", "TLT"]', "[]")))

    def test_duplicate_instruments_are_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="duplicates"):
            load_config(write(tmp_path, VALID.replace('["SPY", "TLT"]', '["SPY", "TLT", "SPY"]')))

    def test_canonical_ids_are_refused(self, tmp_path):
        """`regime:ticker:GLD` would be a new series beside the real one, with
        nothing failing."""
        with pytest.raises(ConfigError, match="bare symbols"):
            load_config(write(tmp_path, VALID.replace('"SPY"', '"ticker:SPY"')))

    def test_untrimmed_instruments_are_refused_not_normalised(self, tmp_path):
        with pytest.raises(ConfigError, match="whitespace"):
            load_config(write(tmp_path, VALID.replace('"TLT"', '" TLT"')))

    def test_a_non_positive_fitting_parameter_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="must be positive"):
            load_config(write(tmp_path, VALID.replace("s_max = 3", "s_max = 0")))

    def test_a_boolean_is_not_accepted_as_a_number(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(write(tmp_path, VALID.replace("n_starts = 20", "n_starts = true")))


class TestWindowsCannotCollide:
    """Two windows sharing a series would upsert into each other's history."""

    def test_two_unrestricted_windows_are_refused(self, tmp_path):
        body = VALID.replace("[windows.full]", "[windows.full]\n\n[windows.everything]")
        with pytest.raises(ConfigError, match="unqualified series"):
            load_config(write(tmp_path, body))

    def test_two_windows_of_the_same_length_are_refused(self, tmp_path):
        twin = "[windows.8y]\nlookback_years = 8\n\n[windows.eight]\nlookback_years = 8"
        body = VALID.replace("[windows.8y]\nlookback_years = 8", twin)
        with pytest.raises(ConfigError, match="collide"):
            load_config(write(tmp_path, body))

    def test_a_negative_lookback_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="positive integer"):
            load_config(write(tmp_path, VALID.replace("lookback_years = 8", "lookback_years = -8")))

    def test_no_windows_at_all_is_refused(self, tmp_path):
        body = VALID[: VALID.index("[windows.full]")]
        with pytest.raises(ConfigError, match="non-empty .windows."):
            load_config(write(tmp_path, body))


class TestComparisonsMustMakeSense:
    def test_an_unknown_window_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="not a declared window"):
            load_config(write(tmp_path, VALID.replace('candidate = "8y"', 'candidate = "12y"')))

    def test_comparing_a_window_with_itself_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="with itself"):
            load_config(write(tmp_path, VALID.replace('baseline = "full"', 'baseline = "8y"')))

    def test_a_missing_role_is_refused(self, tmp_path):
        body = VALID.replace('candidate = "8y"', "")
        with pytest.raises(ConfigError, match="candidate"):
            load_config(write(tmp_path, body))


class TestShippedExample:
    def test_the_example_config_actually_loads(self):
        """A broken example is worse than none: it is what a new user copies."""
        example = Path(__file__).resolve().parents[1] / "examples" / "regime_daily.example.toml"
        cfg = load_config(example)
        assert cfg.instruments
        assert cfg.window("full").variant is None
        assert cfg.comparisons


class TestDisplayNames:
    """Names are a preset, like the universe: the method has no opinion about
    what 'GLD' is called. They live here so the public package carries none."""

    def test_names_are_optional(self, tmp_path):
        assert load_config(write(tmp_path, VALID)).names == {}

    def test_a_name_is_read_for_a_declared_instrument(self, tmp_path):
        cfg = load_config(write(tmp_path, VALID + '\n[names]\nSPY = "S&P 500 ETF"\n'))
        assert cfg.names["SPY"] == "S&P 500 ETF"

    def test_a_name_for_an_unfitted_instrument_is_refused(self, tmp_path):
        """Otherwise the typo does nothing at all: the report shows the ticker
        and the preset looks applied."""
        with pytest.raises(ConfigError, match="not in 'instruments'"):
            load_config(write(tmp_path, VALID + '\n[names]\nNOPE = "Nothing"\n'))

    def test_an_empty_name_is_refused(self, tmp_path):
        with pytest.raises(ConfigError, match="non-empty string"):
            load_config(write(tmp_path, VALID + '\n[names]\nSPY = ""\n'))
