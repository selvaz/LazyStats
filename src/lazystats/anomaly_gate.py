# -*- coding: utf-8 -*-
"""Which movements in a daily statistics run deserve investigating.

A pure function over two consecutive payloads. It opens no database, reads
no environment and holds no module state: everything it needs — including
the thresholds — arrives as arguments, so the same inputs always produce the
same output and a shadow run can be compared exactly against a live one.

Four kinds of anomaly, each requiring **both** an unusual level and a fresh
move to get there. Either condition alone over-triggers: an instrument
parked in an elevated band would be reported every day forever, and a large
move within the normal band is just noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from lazystats.anomaly_gate_config import AnomalyGateConfig

#: Anomaly kinds this gate can emit.
ANOMALY_TYPES = ("return_outlier", "volatility_shift", "correlation_shift", "beta_divergence")


@dataclass(frozen=True)
class AnomalyItem:
    instrument: str
    anomaly_type: str
    date: str
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"instrument": self.instrument, "anomaly_type": self.anomaly_type,
                "date": self.date, "detail": dict(self.detail)}


@dataclass(frozen=True)
class InvestigationTarget:
    date: str
    items: tuple[AnomalyItem, ...]
    trigger_result_id: str

    def as_dict(self) -> dict[str, Any]:
        return {"date": self.date, "trigger_result_id": self.trigger_result_id,
                "items": [i.as_dict() for i in self.items]}


def is_weekend(iso_date: str) -> bool:
    """True for Saturday and Sunday.

    market-data-hub writes a placeholder row for non-trading calendar days
    with ``adj_close`` left unadjusted, which produces a spurious return
    between Friday's real close and the placeholder. ``v_returns`` applies
    no trading-calendar filter, so it surfaces as a z-score outlier on a day
    the market was shut. Filtering here is a downstream safety net,
    independent of whichever fix eventually lands upstream.
    """
    return date.fromisoformat(iso_date).weekday() >= 5


def _vol_band(ratio: float | None, cfg: AnomalyGateConfig) -> str | None:
    if ratio is None:
        return None
    if ratio >= cfg.vol_ratio_high:
        return "elevated"
    if ratio <= cfg.vol_ratio_low:
        return "compressed"
    return "normal"


def _corr_band(value: float | None, cfg: AnomalyGateConfig) -> str | None:
    if value is None:
        return None
    if value >= cfg.corr_high:
        return "high"
    if value <= cfg.corr_low:
        return "low"
    return "mid"


def _vol_ratios(payload: dict) -> dict[str, float | None]:
    """Short-window volatility over long-window, per instrument."""
    short = payload["volatility_short"]["volatility"]
    long_ = payload["volatility_long"]["volatility"]
    out: dict[str, float | None] = {}
    for key, s in short.items():
        l = long_.get(key)
        s_v = s.get("annualized_volatility") if s else None
        l_v = l.get("annualized_volatility") if l else None
        out[key] = (s_v / l_v) if (s_v is not None and l_v) else None
    return out


def _beta_z_scores(payload: dict, benchmark: str) -> dict[str, float | None]:
    """How many idiosyncratic sigmas each instrument's weekly return diverges
    from what its beta to ``benchmark`` predicts.

    ``beta = corr * vol_a / vol_benchmark`` — both terms are already in the
    payload, so no regression is fitted. Residual weekly volatility is
    ``vol_a * sqrt(1 - corr**2)``, the standard variance decomposition.
    ``None`` wherever an input is missing or the residual volatility is
    degenerate: dividing by it would manufacture an enormous z-score out of
    an instrument that simply tracks the benchmark.
    """
    vol = payload["volatility_short"]["volatility"]
    corr = payload["correlation_short"]["correlation"]
    returns_1w = payload["returns_table"]

    vol_bench = vol.get(benchmark, {}).get("period_volatility")
    bench_return = returns_1w.get(benchmark, {}).get("1W", {}).get("return")

    out: dict[str, float | None] = {}
    if not vol_bench or bench_return is None:
        return out
    for instrument, v in vol.items():
        if instrument == benchmark:
            continue
        vol_a = v.get("period_volatility")
        rho = corr.get(instrument, {}).get(benchmark)
        actual = returns_1w.get(instrument, {}).get("1W", {}).get("return")
        if vol_a is None or rho is None or actual is None:
            out[instrument] = None
            continue
        beta = rho * (vol_a / vol_bench)
        residual = actual - beta * bench_return
        residual_vol = vol_a * (max(0.0, 1 - rho ** 2)) ** 0.5
        out[instrument] = (residual / residual_vol) if residual_vol > 1e-12 else None
    return out


def evaluate_gate(
    *,
    current: dict,
    previous: dict,
    trigger_result_id: str,
    config: AnomalyGateConfig,
    already_investigated: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[InvestigationTarget, ...]:
    """Select the movements worth investigating, grouped by date.

    Args:
        current: The most recent daily statistics payload.
        previous: The payload before it. Required: every check needs a prior
            reading to measure a fresh move against.
        trigger_result_id: Identifies the run that produced ``current``.
        config: Thresholds. No defaults — see
            :mod:`lazystats.anomaly_gate_config`.
        already_investigated: ``(instrument, date)`` pairs covered by an
            earlier investigation, so a multi-day outlier window does not
            re-raise the same day. Supplied by the caller: reading it would
            mean touching a database, and this function does not.

    Returns:
        One target per date, ordered by date; empty when nothing qualifies.
    """
    as_of = current["as_of"]
    items: list[AnomalyItem] = []

    # -- return outliers -------------------------------------------------
    for o in current["outliers_last5"]["outliers"]:
        if is_weekend(o["date"]):
            continue
        if (o["instrument"], o["date"]) in already_investigated:
            continue
        items.append(AnomalyItem(
            instrument=o["instrument"], anomaly_type="return_outlier", date=o["date"],
            detail={"z_score": o["z_score"], "log_return": o["log_return"],
                    "direction": o["direction"]},
        ))

    # -- volatility: unusual band AND a fresh move ------------------------
    today_ratios = _vol_ratios(current)
    prior_ratios = _vol_ratios(previous)
    for instrument, ratio in today_ratios.items():
        band = _vol_band(ratio, config)
        prior = prior_ratios.get(instrument)
        if band is None or band == "normal" or prior is None:
            continue
        delta = abs(ratio - prior)
        if delta >= config.vol_ratio_delta_min:
            items.append(AnomalyItem(
                instrument=instrument, anomaly_type="volatility_shift", date=as_of,
                detail={"band": band, "ratio_short_over_long": ratio,
                        "ratio_prior": prior, "ratio_delta": delta},
            ))

    # -- correlation: unusual band AND a fresh move -----------------------
    today_corr = current["correlation_short"]["correlation"]
    prior_corr = previous["correlation_short"]["correlation"]
    candidates: list[AnomalyItem] = []
    seen: set[frozenset[str]] = set()
    for a, row in today_corr.items():
        for b, value in row.items():
            if a == b or value is None:
                continue
            pair = frozenset((a, b))
            if pair in seen:  # the matrix is symmetric; report each pair once
                continue
            seen.add(pair)
            band = _corr_band(value, config)
            prior = prior_corr.get(a, {}).get(b)
            if band is None or band == "mid" or prior is None:
                continue
            delta = abs(value - prior)
            if delta >= config.corr_delta_min:
                candidates.append(AnomalyItem(
                    instrument=f"{a.replace('ticker:', '')}/{b.replace('ticker:', '')}",
                    anomaly_type="correlation_shift", date=as_of,
                    detail={"band": band, "correlation_short": value,
                            "correlation_prior": prior, "correlation_delta": delta},
                ))
    # Worst movers first, then capped: a data glitch could otherwise produce
    # hundreds of pairs and inflate the investigation that follows.
    candidates.sort(key=lambda it: it.detail["correlation_delta"], reverse=True)
    items.extend(candidates[:config.max_corr_shifts_per_day])

    # -- beta divergence: unusual sigma AND a fresh move ------------------
    today_z = _beta_z_scores(current, config.beta_benchmark)
    prior_z = _beta_z_scores(previous, config.beta_benchmark)
    for instrument, z in today_z.items():
        prior = prior_z.get(instrument)
        if z is None or prior is None or abs(z) < config.beta_z_threshold:
            continue
        delta = abs(z - prior)
        if delta >= config.beta_z_delta_min:
            items.append(AnomalyItem(
                instrument=instrument, anomaly_type="beta_divergence", date=as_of,
                detail={"benchmark": config.beta_benchmark.replace("ticker:", ""),
                        "z_score": z, "z_score_prior": prior, "z_score_delta": delta},
            ))

    if not items:
        return ()

    by_date: dict[str, list[AnomalyItem]] = {}
    for item in items:
        by_date.setdefault(item.date, []).append(item)
    return tuple(
        InvestigationTarget(date=d, items=tuple(group), trigger_result_id=trigger_result_id)
        for d, group in sorted(by_date.items())
    )


__all__ = ["ANOMALY_TYPES", "AnomalyItem", "InvestigationTarget", "evaluate_gate", "is_weekend"]
