"""Configuration contract for the daily anomaly gate.

Same separation as the ETF runner: the *method* — how a volatility shift or
a correlation break is detected — stays in this public repository, while the
bands, deltas and caps that decide what counts as unusual belong to whoever
runs the job. A threshold is a judgement about a portfolio, not a
statistical fact.

There is deliberately no default preset, and ``load_gate_config`` requires
an explicit path. A gate silently running at someone else's sensitivity
would either flood an investigation with noise or quietly miss the day that
mattered.

Two fields are not thresholds and deserve attention.
``upstream_series_key`` and ``upstream_produced_by`` name the series this
gate reads and the identity its rows carry. The two presets are therefore
**coupled**: if the producing job's identity changes, this one stops finding
its input and reports nothing, which is indistinguishable from a quiet day.
Keeping them as explicit, validated fields makes the dependency visible
instead of leaving it buried in a module constant — and, because they live
here rather than on the command line, no invocation can quietly point the
gate at a different series than the preset declares.

TOML via the standard library's ``tomllib``; no new dependency.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnomalyGateConfig:
    """What the gate treats as unusual, and where it reads from."""

    #: The series this gate reads, and the identity its rows carry. Both
    #: come from the preset: a CLI override would let one run disagree with
    #: the configuration it claims to be running.
    upstream_series_key: str
    upstream_produced_by: str

    # Volatility: the short/long ratio bands, plus the minimum fresh
    # day-over-day move required alongside being in the band. Band alone or
    # delta alone both over-trigger, which is why the pair exists.
    vol_ratio_high: float
    vol_ratio_low: float
    vol_ratio_delta_min: float

    # Correlation: the same band-plus-delta shape.
    corr_high: float
    corr_low: float
    corr_delta_min: float

    #: Hard cap on correlation items per day, worst movers first. A safety
    #: net against a data glitch inflating the investigation, independent of
    #: the thresholds above.
    max_corr_shifts_per_day: int

    #: Benchmark for the beta divergence check. Which index is the reference
    #: is a project choice, not a statistical fact; there is no universal one.
    beta_benchmark: str
    beta_z_threshold: float
    beta_z_delta_min: float

    #: How many past rows to scan when de-duplicating outliers.
    dedup_lookback: int

    def as_provenance(self) -> dict[str, object]:
        """The parameter block recorded alongside a result, so a run from
        configuration stays comparable with one from before the extraction."""
        return {
            "upstream_series_key": self.upstream_series_key,
            "upstream_produced_by": self.upstream_produced_by,
            "vol_ratio_high": self.vol_ratio_high,
            "vol_ratio_low": self.vol_ratio_low,
            "vol_ratio_delta_min": self.vol_ratio_delta_min,
            "corr_high": self.corr_high,
            "corr_low": self.corr_low,
            "corr_delta_min": self.corr_delta_min,
            "max_corr_shifts_per_day": self.max_corr_shifts_per_day,
            "beta_benchmark": self.beta_benchmark,
            "beta_z_threshold": self.beta_z_threshold,
            "beta_z_delta_min": self.beta_z_delta_min,
            "dedup_lookback": self.dedup_lookback,
        }


class GateConfigError(ValueError):
    """The gate configuration is missing, malformed or incoherent."""


def _number(raw: dict, key: str, path: Path) -> float:
    if key not in raw:
        raise GateConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    # bool subclasses int; a threshold of `true` must not become 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateConfigError(
            f"{path.name}: '{key}' must be a number, got {type(value).__name__}"
        )
    return float(value)


def _positive_int(raw: dict, key: str, path: Path) -> int:
    if key not in raw:
        raise GateConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateConfigError(
            f"{path.name}: '{key}' must be an integer, got {type(value).__name__}"
        )
    if value <= 0:
        raise GateConfigError(f"{path.name}: '{key}' must be positive, got {value}")
    return value


def _text(raw: dict, key: str, path: Path) -> str:
    if key not in raw:
        raise GateConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise GateConfigError(f"{path.name}: '{key}' must be a non-empty string")
    if value != value.strip():
        raise GateConfigError(f"{path.name}: '{key}' must not have surrounding whitespace")
    return value


def load_gate_config(path: str | Path) -> AnomalyGateConfig:
    """Load and validate a gate configuration.

    Raises:
        GateConfigError: The file is absent, unparseable, or a value is
            missing, of the wrong type, or incoherent with another. Messages
            name the file and the key: they are read while diagnosing a
            scheduled run that failed.
    """
    p = Path(path)
    if not p.is_file():
        raise GateConfigError(f"gate configuration not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise GateConfigError(f"{p.name}: not valid TOML: {exc}") from exc

    cfg = AnomalyGateConfig(
        upstream_series_key=_text(raw, "upstream_series_key", p),
        upstream_produced_by=_text(raw, "upstream_produced_by", p),
        vol_ratio_high=_number(raw, "vol_ratio_high", p),
        vol_ratio_low=_number(raw, "vol_ratio_low", p),
        vol_ratio_delta_min=_number(raw, "vol_ratio_delta_min", p),
        corr_high=_number(raw, "corr_high", p),
        corr_low=_number(raw, "corr_low", p),
        corr_delta_min=_number(raw, "corr_delta_min", p),
        max_corr_shifts_per_day=_positive_int(raw, "max_corr_shifts_per_day", p),
        beta_benchmark=_text(raw, "beta_benchmark", p),
        beta_z_threshold=_number(raw, "beta_z_threshold", p),
        beta_z_delta_min=_number(raw, "beta_z_delta_min", p),
        dedup_lookback=_positive_int(raw, "dedup_lookback", p),
    )

    # Coherence, not just types. An inverted band silently inverts the
    # gate's meaning: everything normal becomes unusual and vice versa,
    # and nothing downstream would report an error.
    if cfg.vol_ratio_low >= cfg.vol_ratio_high:
        raise GateConfigError(
            f"{p.name}: 'vol_ratio_low' ({cfg.vol_ratio_low}) must be below "
            f"'vol_ratio_high' ({cfg.vol_ratio_high}); they bound a band"
        )
    if cfg.corr_low >= cfg.corr_high:
        raise GateConfigError(
            f"{p.name}: 'corr_low' ({cfg.corr_low}) must be below "
            f"'corr_high' ({cfg.corr_high}); they bound a band"
        )
    if cfg.vol_ratio_low <= 0:
        raise GateConfigError(
            f"{p.name}: 'vol_ratio_low' must be positive, got {cfg.vol_ratio_low}"
        )
    for name in ("vol_ratio_delta_min", "corr_delta_min", "beta_z_threshold", "beta_z_delta_min"):
        if getattr(cfg, name) <= 0:
            raise GateConfigError(f"{p.name}: '{name}' must be positive, got {getattr(cfg, name)}")
    # Correlations live in [-1, 1]; a band outside it can never trigger, so
    # the gate would run every day and find nothing.
    if not (-1.0 <= cfg.corr_low <= 1.0) or not (-1.0 <= cfg.corr_high <= 1.0):
        raise GateConfigError(
            f"{p.name}: correlation bands must lie within [-1, 1]; "
            f"got low={cfg.corr_low}, high={cfg.corr_high}"
        )
    return cfg


__all__ = ["AnomalyGateConfig", "GateConfigError", "load_gate_config"]
