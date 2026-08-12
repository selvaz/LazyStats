"""Configuration contract for the ETF daily-stats run.

Separates the *method* — which lives in this public repository — from the
*preset*: which instruments to watch, over which windows, above which
outlier threshold. Those are project choices, not statistics, and they
belong to whoever runs the job.

There is deliberately **no default preset**. ``load_config`` requires an
explicit path, and ``run_daily_etf_stats.py`` requires ``--config``. A
default here would put one project's universe back inside a general-purpose
repository, which is exactly what the ecosystem cleanup is undoing — and a
job silently running against the wrong universe is worse than one that
refuses to start.

``examples/etf_daily_stats.example.toml`` shows the shape with a small,
illustrative universe. It is used only when passed explicitly.

TOML is read with the standard library's ``tomllib`` (Python 3.11+), so this
adds no dependency.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Horizon labels the report knows how to render. "YTD" is special-cased to
#: "since 31 December of the prior year" rather than a fixed day count, so it
#: carries no ``days`` value.
_YTD = "YTD"

#: Extra weeks fetched beyond the longest configured window. Weekly bars are
#: not guaranteed one per calendar week (holidays, partial weeks, listings that
#: miss a print), so asking for exactly N weeks of calendar history can return
#: fewer than N observations.
_WEEK_CUSHION = 8


@dataclass(frozen=True)
class ReturnHorizon:
    """One column of the returns table.

    Attributes:
        label: Display label, e.g. ``"3M"``.
        days_back: Calendar days back from as-of. ``None`` only for
            ``"YTD"``, which the runner resolves against the prior
            year-end.
    """

    label: str
    days_back: int | None


@dataclass(frozen=True)
class EtfStatsConfig:
    """Everything the run needs that is a project choice rather than method."""

    instruments: tuple[str, ...]
    short_weeks: int
    long_weeks: int
    one_year_weeks: int
    daily_lookback_days: int
    outlier_window_days: int
    outlier_chart_days: int
    outlier_threshold: float
    series_key: str
    produced_by: str
    return_horizons: tuple[ReturnHorizon, ...]

    @property
    def weekly_history_weeks(self) -> int:
        """Calendar weeks of weekly history a run has to fetch.

        Every weekly window — short, long and one-year — is sliced out of a
        single fetch, so this is the longest of them plus a cushion. Sizing it
        to ``long_weeks`` alone would silently shorten a ``one_year_weeks``
        baseline that exceeds it: the slice returns whatever it finds, so the
        run would report a volatility computed over fewer observations than
        ``as_provenance`` claims.
        """
        return max(self.long_weeks, self.one_year_weeks) + _WEEK_CUSHION

    def as_provenance(self) -> dict[str, object]:
        """The parameter block recorded alongside a result.

        Mirrors what the previous hardcoded version wrote, so a result
        produced from a config file stays comparable with one produced
        before the extraction.
        """
        return {
            "short_window_weeks": self.short_weeks,
            "long_window_weeks": self.long_weeks,
            "one_year_window_weeks": self.one_year_weeks,
            "daily_lookback_days": self.daily_lookback_days,
            "outlier_window_days": self.outlier_window_days,
            "outlier_chart_days": self.outlier_chart_days,
            "outlier_threshold": self.outlier_threshold,
            "return_horizons": [h.label for h in self.return_horizons],
        }


class ConfigError(ValueError):
    """The configuration file is missing, malformed or incomplete.

    Its own type so a caller can tell "you configured this wrong" apart
    from a failure inside the analysis.
    """


def _require(raw: dict[str, Any], key: str, kind: type, path: Path) -> Any:
    if key not in raw:
        raise ConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    # bool is a subclass of int; a threshold of `true` must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, kind):
        got = type(value).__name__
        raise ConfigError(f"{path.name}: '{key}' must be {kind.__name__}, got {got}")
    return value


def load_config(path: str | Path) -> EtfStatsConfig:
    """Load and validate a run configuration.

    Args:
        path: Explicit path to a TOML file. There is no default and no
            search: the caller states which preset it is running.

    Raises:
        ConfigError: The file is absent, unparseable, or a key is missing,
            of the wrong type, or out of range. Every message names the
            file and the key, because this is read by whoever is fixing a
            failed scheduled run.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p.name}: not valid TOML: {exc}") from exc

    instruments = _require(raw, "instruments", list, p)
    if not instruments:
        raise ConfigError(f"{p.name}: 'instruments' is empty; nothing to analyse")
    if not all(isinstance(s, str) and s.strip() for s in instruments):
        raise ConfigError(f"{p.name}: 'instruments' must be non-empty strings")
    # Refuse rather than strip: " SPY" and "SPY" would silently become the
    # same instrument after normalisation, hiding a typo in the preset.
    untrimmed = [s for s in instruments if s != s.strip()]
    if untrimmed:
        raise ConfigError(
            f"{p.name}: 'instruments' entries must not have leading or trailing "
            f"whitespace: {untrimmed}"
        )
    # Bare symbols only. ``io.datahub.load_returns`` also accepts the canonical
    # ``ticker:SPY`` form, but the rest of the run does not: the report's
    # instrument metadata is queried by bare symbol, and the renderer rebuilds
    # the canonical key itself, so a canonical value here would be looked up as
    # ``ticker:ticker:SPY`` and every cell would render blank while the analysis
    # reported success. Refuse rather than normalise, for the same reason as the
    # whitespace check above: one accepted spelling, stated in one place.
    qualified = [s for s in instruments if ":" in s]
    if qualified:
        raise ConfigError(
            f"{p.name}: 'instruments' must be bare symbols, not canonical ids: "
            f"{qualified} — write 'SPY', not 'ticker:SPY'"
        )
    if len(set(instruments)) != len(instruments):
        dupes = sorted({s for s in instruments if instruments.count(s) > 1})
        raise ConfigError(f"{p.name}: 'instruments' contains duplicates: {dupes}")

    windows = {
        name: _require(raw, name, int, p)
        for name in (
            "short_weeks",
            "long_weeks",
            "one_year_weeks",
            "daily_lookback_days",
            "outlier_window_days",
            "outlier_chart_days",
        )
    }
    for name, value in windows.items():
        if value <= 0:
            raise ConfigError(f"{p.name}: '{name}' must be positive, got {value}")
    if windows["short_weeks"] >= windows["long_weeks"]:
        raise ConfigError(
            f"{p.name}: 'short_weeks' ({windows['short_weeks']}) must be shorter than "
            f"'long_weeks' ({windows['long_weeks']}); the report contrasts the two"
        )

    threshold = raw.get("outlier_threshold")
    if isinstance(threshold, int) and not isinstance(threshold, bool):
        threshold = float(threshold)
    if not isinstance(threshold, float):
        raise ConfigError(f"{p.name}: 'outlier_threshold' must be a number")
    # `nan` and `inf` are valid TOML float literals and neither is <= 0, so the
    # positivity check alone lets them through — and `return_outliers` then
    # rejects them mid-run. Fail here, where the contract promises configuration
    # errors are raised.
    if not math.isfinite(threshold):
        raise ConfigError(f"{p.name}: 'outlier_threshold' must be finite, got {threshold}")
    if threshold <= 0:
        raise ConfigError(f"{p.name}: 'outlier_threshold' must be positive, got {threshold}")

    series_key = _require(raw, "series_key", str, p)
    if not series_key.strip():
        raise ConfigError(f"{p.name}: 'series_key' must not be blank")

    # Stated, not derived from series_key. Downstream jobs select rows by
    # this identity, and rows already written under it cannot be renamed —
    # so it has to be able to differ from the label, and a convention
    # applied silently here would be a rule nobody could see or override.
    produced_by = _require(raw, "produced_by", str, p)
    if not produced_by.strip():
        raise ConfigError(f"{p.name}: 'produced_by' must not be blank")

    horizons_raw = _require(raw, "return_horizons", list, p)
    if not horizons_raw:
        raise ConfigError(f"{p.name}: 'return_horizons' is empty")
    horizons = []
    seen_labels: set[str] = set()
    for entry in horizons_raw:
        if not isinstance(entry, dict) or "label" not in entry:
            raise ConfigError(
                f"{p.name}: each 'return_horizons' entry needs a 'label' "
                f"(and 'days' unless the label is '{_YTD}')"
            )
        label = entry["label"]
        if not isinstance(label, str) or not label.strip():
            raise ConfigError(
                f"{p.name}: horizon 'label' must be a non-empty string, got {label!r}"
            )
        if label in seen_labels:
            raise ConfigError(f"{p.name}: duplicate horizon label {label!r}")
        seen_labels.add(label)
        days = entry.get("days")
        if label == _YTD:
            # Refuse rather than ignore: a 'days' here means the author
            # expected it to be used, and silently dropping it would make
            # the report disagree with its own configuration.
            if days is not None:
                raise ConfigError(
                    f"{p.name}: horizon '{_YTD}' must not carry 'days' — it resolves "
                    f"against 31 December of the prior year, not a day count"
                )
        elif not isinstance(days, int) or isinstance(days, bool) or days <= 0:
            raise ConfigError(
                f"{p.name}: horizon '{label}' needs a positive integer 'days'"
            )
        horizons.append(ReturnHorizon(label=label, days_back=days))

    return EtfStatsConfig(
        instruments=tuple(instruments),
        short_weeks=windows["short_weeks"],
        long_weeks=windows["long_weeks"],
        one_year_weeks=windows["one_year_weeks"],
        daily_lookback_days=windows["daily_lookback_days"],
        outlier_window_days=windows["outlier_window_days"],
        outlier_chart_days=windows["outlier_chart_days"],
        outlier_threshold=threshold,
        series_key=series_key,
        produced_by=produced_by,
        return_horizons=tuple(horizons),
    )


__all__ = ["ConfigError", "EtfStatsConfig", "ReturnHorizon", "load_config"]
