"""Configuration contract for the weekly anomaly review.

The same separation as the daily gate: how a week's explanations are
gathered, checked and synthesised stays here; which series they live in, how
far back to look and how the universe is described belong to whoever runs
the job.

Two groups of fields are worth attention, and neither is a threshold.

The four **identities** — the daily explanations this reads, and the weekly
reviews it writes — couple this job to the ones around it. If the daily
explainer's identity changes, this review silently finds nothing to verify
and reports an empty week, which looks exactly like a quiet one. Keeping
them as explicit, validated fields makes the dependency visible instead of
leaving it in a module constant.

The **explainer** block describes an agent this module never constructs.
Which model reads the week, whether it may browse, and how long it may take
are operational choices; the caller builds the agent and passes it in. That
is what keeps a shadow comparison free of any model at all.

TOML via the standard library's ``tomllib``; no new dependency.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExplainerChoices:
    """How the reviewing agent is to be built — by its caller, not here."""

    model: str
    web: bool
    request_timeout_seconds: float


@dataclass(frozen=True)
class WeeklyReviewConfig:
    """Everything the review needs that is a deployment choice, not method."""

    #: Where the week's daily explanations are read from.
    daily_produced_by: str
    daily_series_key: str

    #: Where this review's own results are written, and read back from to
    #: find where the previous week ended.
    weekly_produced_by: str
    weekly_series_key: str

    #: The statistics job whose freshest snapshot the review checks against.
    upstream_produced_by: str
    upstream_series_key: str

    #: How far back to look when no previous review exists. The first run
    #: has no boundary to start from, and one has to be chosen.
    initial_lookback_days: int

    #: How many explanation rows to scan. A bound, not a window: the window
    #: is the date range, and this only stops an unbounded read.
    daily_scan_limit: int

    #: How the screened universe is described to the agent, and the
    #: threshold the upstream job used. Stated rather than recomputed, so
    #: the prompt cannot quietly disagree with what produced the numbers.
    universe_description: str
    outlier_threshold: float

    explainer: ExplainerChoices

    def as_provenance(self) -> dict[str, object]:
        """The parameter block recorded alongside a result."""
        return {
            "daily_produced_by": self.daily_produced_by,
            "daily_series_key": self.daily_series_key,
            "weekly_produced_by": self.weekly_produced_by,
            "weekly_series_key": self.weekly_series_key,
            "upstream_produced_by": self.upstream_produced_by,
            "upstream_series_key": self.upstream_series_key,
            "initial_lookback_days": self.initial_lookback_days,
            "daily_scan_limit": self.daily_scan_limit,
            "universe_description": self.universe_description,
            "outlier_threshold": self.outlier_threshold,
        }


class WeeklyConfigError(ValueError):
    """The configuration is missing, malformed or incoherent."""


def _text(raw: dict[str, Any], key: str, path: Path) -> str:
    if key not in raw:
        raise WeeklyConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise WeeklyConfigError(f"{path.name}: '{key}' must be a non-empty string")
    if value != value.strip():
        raise WeeklyConfigError(f"{path.name}: '{key}' must not have surrounding whitespace")
    return value


def _positive_int(raw: dict[str, Any], key: str, path: Path) -> int:
    if key not in raw:
        raise WeeklyConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WeeklyConfigError(
            f"{path.name}: '{key}' must be an integer, got {type(value).__name__}"
        )
    if value <= 0:
        raise WeeklyConfigError(f"{path.name}: '{key}' must be positive, got {value}")
    return value


def _number(raw: dict[str, Any], key: str, path: Path) -> float:
    if key not in raw:
        raise WeeklyConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    # bool subclasses int; a threshold of `true` must not become 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeeklyConfigError(
            f"{path.name}: '{key}' must be a number, got {type(value).__name__}"
        )
    return float(value)


def load_weekly_config(path: str | Path) -> WeeklyReviewConfig:
    """Load and validate a weekly review configuration.

    Raises:
        WeeklyConfigError: The file is absent or unparseable, a value is
            missing or of the wrong type, or two values contradict each
            other. Messages name the file and the key: they are read while
            diagnosing a scheduled run that failed.
    """
    p = Path(path)
    if not p.is_file():
        raise WeeklyConfigError(f"weekly review configuration not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise WeeklyConfigError(f"{p.name}: not valid TOML: {exc}") from exc

    explainer_raw = raw.get("explainer")
    if not isinstance(explainer_raw, dict):
        raise WeeklyConfigError(f"{p.name}: missing the [explainer] table")
    model = _text(explainer_raw, "model", p)
    web = explainer_raw.get("web")
    if not isinstance(web, bool):
        raise WeeklyConfigError(f"{p.name}: 'explainer.web' must be true or false")
    timeout = _number(explainer_raw, "request_timeout_seconds", p)
    if timeout <= 0:
        raise WeeklyConfigError(
            f"{p.name}: 'explainer.request_timeout_seconds' must be positive"
        )

    cfg = WeeklyReviewConfig(
        daily_produced_by=_text(raw, "daily_produced_by", p),
        daily_series_key=_text(raw, "daily_series_key", p),
        weekly_produced_by=_text(raw, "weekly_produced_by", p),
        weekly_series_key=_text(raw, "weekly_series_key", p),
        upstream_produced_by=_text(raw, "upstream_produced_by", p),
        upstream_series_key=_text(raw, "upstream_series_key", p),
        initial_lookback_days=_positive_int(raw, "initial_lookback_days", p),
        daily_scan_limit=_positive_int(raw, "daily_scan_limit", p),
        universe_description=_text(raw, "universe_description", p),
        outlier_threshold=_number(raw, "outlier_threshold", p),
        explainer=ExplainerChoices(model=model, web=web,
                                   request_timeout_seconds=timeout),
    )

    # Coherence. A weekly series that collided with the daily one would have
    # the review reading its own output as though it were an explanation to
    # verify, which produces a plausible-looking result about nothing.
    if cfg.weekly_produced_by == cfg.daily_produced_by:
        raise WeeklyConfigError(
            f"{p.name}: 'weekly_produced_by' and 'daily_produced_by' must differ; "
            f"otherwise the review reads its own output as input"
        )
    if cfg.weekly_series_key == cfg.daily_series_key:
        raise WeeklyConfigError(
            f"{p.name}: 'weekly_series_key' and 'daily_series_key' must differ"
        )
    if cfg.outlier_threshold <= 0:
        raise WeeklyConfigError(
            f"{p.name}: 'outlier_threshold' must be positive, got {cfg.outlier_threshold}"
        )
    return cfg


__all__ = [
    "ExplainerChoices",
    "WeeklyConfigError",
    "WeeklyReviewConfig",
    "load_weekly_config",
]
