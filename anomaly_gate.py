"""Deterministic gate: decide which statistical anomalies (if any) are
worth an LLM investigation today.

Pure logic, no LLM, no network. Reads the two most recent
``etf_daily_stats`` rows from lazystats_depot (today's vs. the prior one --
whatever it was, gap or no gap) and flags:

  - return_outlier   -- any (instrument, date) in today's
                         ``outliers_last5.outliers`` not already covered by
                         a stored anomaly_explanation (dedup: an outlier
                         stays in that trailing 5-day window for several
                         days, and must only be investigated once).
  - volatility_shift -- a ticker's short/long volatility ratio is BOTH (a)
                         outside the "normal" band today (elevated or
                         compressed) and (b) moved there freshly -- changed
                         by at least VOL_RATIO_DELTA_MIN since the prior
                         row. (a) alone re-triggers every day of a
                         multi-week elevated-vol period; (b) alone fires on
                         ordinary noise even inside the normal band.
                         Together: only a fresh move into unusual territory.
  - correlation_shift -- a pair's short-window correlation is BOTH (a) in
                         an unusual band today (very high/co-moving or
                         very low/negative) and (b) moved there freshly --
                         at least CORR_DELTA_MIN since the prior row. Same
                         band+delta combination as volatility, for the same
                         reason: band alone re-triggers on a persisting
                         extreme (e.g. two funds that are just always
                         highly correlated); delta alone fires on ordinary
                         rolling-window noise even well inside typical
                         territory (observed: a pair moving 0.164->0.210,
                         both unremarkable levels, "crossed" a naive 0.2
                         line and would have triggered on delta alone).
  - beta_divergence   -- an instrument's actual trailing-1-week return
                         diverges from what its beta to SPY would predict,
                         BOTH (a) by an unusual amount today (|z| over
                         BETA_Z_THRESHOLD "idiosyncratic sigmas") and (b) a
                         fresh move to get there (|z| changed by at least
                         BETA_Z_DELTA_MIN since the prior row). Beta itself
                         needs no new data: for two series with volatilities
                         vol_A, vol_B and correlation rho, beta of A on B is
                         simply rho * (vol_A / vol_B) -- already-stored
                         correlation_short + volatility_short fields are
                         sufficient, no regression fit needed. The
                         "idiosyncratic sigma" denominator is the residual
                         (non-beta-explained) weekly vol implied by the same
                         two inputs: vol_A * sqrt(1 - rho**2).

``find_investigation_targets()`` groups flagged items by date -- several
tickers flagged for the same date is one macro event, one investigation,
not several -- and returns a list of targets (empty if nothing qualifies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import lazytools.registry as lazytools_registry
from lazystats.io.depot import ResultDepot

SERIES_PRODUCED_BY = "scheduled:etf_daily_stats"

# Volatility short/long ratio: "unusual" band boundaries, plus the minimum
# fresh day-over-day change required alongside being in that band (see
# module docstring -- band alone or delta alone both over-trigger).
VOL_RATIO_HIGH = 1.5
VOL_RATIO_LOW = 1 / 1.5  # ~0.67, symmetric in log-space with HIGH
VOL_RATIO_DELTA_MIN = 0.15

# Short-window correlation: "unusual" band boundaries, plus the minimum
# fresh day-over-day change required alongside being in that band (mirrors
# the volatility band+delta combination above).
CORR_HIGH = 0.7
CORR_LOW = 0.15  # at/below this, including negative, counts as "low"
CORR_DELTA_MIN = 0.20

# Hard cap on correlation_shift items per day, worst-movers-first -- a
# safety net against a pathological case (e.g. a data glitch) blowing up
# the investigation prompt/cost, independent of the threshold above.
MAX_CORR_SHIFTS_PER_DAY = 8

# Benchmark used for the beta_divergence check.
BETA_BENCHMARK = "ticker:SPY"
# |actual 1W return - beta-implied 1W return| expressed in units of
# idiosyncratic (non-beta-explained) weekly vol -- "how many sigma of
# unexplained move is this". Mirrors OUTLIER_THRESHOLD's z=2.0 convention
# from run_daily_etf_stats.py.
BETA_Z_THRESHOLD = 2.0
BETA_Z_DELTA_MIN = 1.0

# How many past investigation-group rows to scan for outlier dedup --
# generous relative to the 5-trading-day outliers_last5 window.
DEDUP_LOOKBACK = 20


@dataclass
class AnomalyItem:
    instrument: str
    anomaly_type: str  # "return_outlier" | "volatility_shift" | "correlation_shift"
    date: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationTarget:
    date: str
    items: list[AnomalyItem]
    trigger_result_id: str  # the etf_daily_stats row that produced this


def _is_weekend(iso_date: str) -> bool:
    """True for Saturday/Sunday.

    market-data-hub's live-price injection has a known data-quality bug
    (confirmed 2026-08-05): it writes a placeholder row for non-trading
    calendar days with ``adj_close`` left un-adjusted, producing a spurious
    day-over-day return between Friday's real close and the weekend
    placeholder. ``v_returns`` has no trading-calendar filter, so this
    surfaces as a z-score "outlier" on a day markets were closed. Filtering
    weekend dates here is a downstream safety net independent of whichever
    fix (if any) eventually lands upstream in market-data-hub.
    """
    return date.fromisoformat(iso_date).weekday() >= 5


def _vol_band(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    if ratio >= VOL_RATIO_HIGH:
        return "elevated"
    if ratio <= VOL_RATIO_LOW:
        return "compressed"
    return "normal"


def _vol_ratios(row_payload: dict) -> dict[str, float | None]:
    short = row_payload["volatility_short"]["volatility"]
    long_ = row_payload["volatility_long"]["volatility"]
    out: dict[str, float | None] = {}
    for key, s in short.items():
        l = long_.get(key)
        s_v = s.get("annualized_volatility") if s else None
        l_v = l.get("annualized_volatility") if l else None
        out[key] = (s_v / l_v) if (s_v is not None and l_v) else None
    return out


def _corr_band(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= CORR_HIGH:
        return "high"
    if value <= CORR_LOW:
        return "low"
    return "mid"


def _beta_z_scores(row_payload: dict, benchmark: str = BETA_BENCHMARK) -> dict[str, float | None]:
    """Per instrument: how many idiosyncratic-sigma its trailing-1W actual
    return diverges from what its beta to ``benchmark`` predicts.

    beta = corr(A, benchmark) * vol_A / vol_benchmark -- both terms already
    in volatility_short/correlation_short, no regression fit needed.
    Residual (non-beta-explained) weekly vol = vol_A * sqrt(1 - corr**2),
    the standard CAPM variance decomposition. Returns ``None`` for an
    instrument where any input is missing/degenerate (e.g. the benchmark
    itself, or a zero residual vol).
    """
    vol = row_payload["volatility_short"]["volatility"]
    corr = row_payload["correlation_short"]["correlation"]
    returns_1w = row_payload["returns_table"]

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
        actual_return = returns_1w.get(instrument, {}).get("1W", {}).get("return")
        if vol_a is None or rho is None or actual_return is None:
            out[instrument] = None
            continue
        beta = rho * (vol_a / vol_bench)
        expected_return = beta * bench_return
        residual = actual_return - expected_return
        residual_vol = vol_a * (max(0.0, 1 - rho**2)) ** 0.5
        out[instrument] = (residual / residual_vol) if residual_vol > 1e-12 else None
    return out


def _already_investigated(depot: ResultDepot) -> set[tuple[str, str]]:
    """(instrument, date) pairs already covered by a stored
    return_outlier anomaly_explanation, scanning the most recent
    investigation-group rows."""
    covered: set[tuple[str, str]] = set()
    for entry in depot.list(produced_by="lazystats.anomaly_explainer", cadence="stable", limit=DEDUP_LOOKBACK):
        row = depot.load(entry["result_id"])
        if not row:
            continue
        for item in row["payload"].get("items", []):
            if item.get("anomaly_type") == "return_outlier":
                # Stored items carry the LLM's output instrument, which has
                # the "ticker:" prefix stripped (see anomaly_explainer's
                # EXPLAIN_PROMPT); the gate's own outliers_last5 items keep
                # it. Without re-adding it here the two never match and
                # dedup silently never fires (found live 2026-08-05: the
                # same already-explained outlier kept re-triggering).
                instrument = item["instrument"]
                if not instrument.startswith("ticker:"):
                    instrument = f"ticker:{instrument}"
                covered.add((instrument, item["date"]))
    return covered


def find_investigation_targets(
    *, depot_path: str | None = None, explanations_depot_path: str | None = None
) -> list[InvestigationTarget]:
    """Compare the two most recent ``etf_daily_stats`` rows and return
    grouped-by-date investigation targets (empty list if nothing qualifies).
    """
    depot_path = depot_path or lazytools_registry.resolve_db("lazystats_depot")
    explanations_depot_path = explanations_depot_path or lazytools_registry.resolve_db("anomaly_explanations")

    depot = ResultDepot(depot_path)
    explanations_depot = ResultDepot(explanations_depot_path) if explanations_depot_path else None
    try:
        recent = depot.list(produced_by=SERIES_PRODUCED_BY, cadence="stable", limit=2)
        if len(recent) < 2:
            return []  # need a prior row to compare against
        today_row = depot.load(recent[0]["result_id"])
        prior_row = depot.load(recent[1]["result_id"])
        if today_row is None or prior_row is None:
            return []
        today, prior = today_row["payload"], prior_row["payload"]

        already = _already_investigated(explanations_depot) if explanations_depot else set()

        items: list[AnomalyItem] = []

        # -- return outliers --
        as_of = today["as_of"]
        for o in today["outliers_last5"]["outliers"]:
            if _is_weekend(o["date"]):
                continue
            key = (o["instrument"], o["date"])
            if key in already:
                continue
            items.append(
                AnomalyItem(
                    instrument=o["instrument"],
                    anomaly_type="return_outlier",
                    date=o["date"],
                    detail={
                        "z_score": o["z_score"],
                        "log_return": o["log_return"],
                        "direction": o["direction"],
                    },
                )
            )

        # -- volatility: unusual band AND a fresh move to get there --
        today_ratios = _vol_ratios(today)
        prior_ratios = _vol_ratios(prior)
        for instrument, today_ratio in today_ratios.items():
            today_band = _vol_band(today_ratio)
            prior_ratio = prior_ratios.get(instrument)
            if today_band is None or today_band == "normal" or prior_ratio is None:
                continue
            delta = abs(today_ratio - prior_ratio)
            if delta >= VOL_RATIO_DELTA_MIN:
                items.append(
                    AnomalyItem(
                        instrument=instrument,
                        anomaly_type="volatility_shift",
                        date=as_of,
                        detail={
                            "band": today_band,
                            "ratio_short_over_long": today_ratio,
                            "ratio_prior": prior_ratio,
                            "ratio_delta": delta,
                        },
                    )
                )

        # -- correlation: unusual band AND a fresh move to get there --
        today_corr = today["correlation_short"]["correlation"]
        prior_corr = prior["correlation_short"]["correlation"]
        corr_candidates: list[AnomalyItem] = []
        seen_pairs: set[frozenset[str]] = set()
        for a, row_a in today_corr.items():
            for b, value in row_a.items():
                if a == b or value is None:
                    continue
                pair = frozenset((a, b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                today_band = _corr_band(value)
                prior_value = prior_corr.get(a, {}).get(b)
                if today_band is None or today_band == "mid" or prior_value is None:
                    continue
                delta = abs(value - prior_value)
                if delta >= CORR_DELTA_MIN:
                    corr_candidates.append(
                        AnomalyItem(
                            instrument=f"{a.replace('ticker:', '')}/{b.replace('ticker:', '')}",
                            anomaly_type="correlation_shift",
                            date=as_of,
                            detail={
                                "band": today_band,
                                "correlation_short": value,
                                "correlation_prior": prior_value,
                                "correlation_delta": delta,
                            },
                        )
                    )
        corr_candidates.sort(key=lambda it: it.detail["correlation_delta"], reverse=True)
        items.extend(corr_candidates[:MAX_CORR_SHIFTS_PER_DAY])

        # -- beta divergence: unusual idiosyncratic-sigma AND a fresh move --
        today_beta_z = _beta_z_scores(today)
        prior_beta_z = _beta_z_scores(prior)
        for instrument, z in today_beta_z.items():
            prior_z = prior_beta_z.get(instrument)
            if z is None or prior_z is None or abs(z) < BETA_Z_THRESHOLD:
                continue
            delta = abs(z - prior_z)
            if delta >= BETA_Z_DELTA_MIN:
                items.append(
                    AnomalyItem(
                        instrument=instrument,
                        anomaly_type="beta_divergence",
                        date=as_of,
                        detail={
                            "benchmark": BETA_BENCHMARK.replace("ticker:", ""),
                            "z_score": z,
                            "z_score_prior": prior_z,
                            "z_score_delta": delta,
                        },
                    )
                )

        if not items:
            return []

        by_date: dict[str, list[AnomalyItem]] = {}
        for item in items:
            by_date.setdefault(item.date, []).append(item)

        return [
            InvestigationTarget(date=d, items=group, trigger_result_id=today_row["result_id"])
            for d, group in sorted(by_date.items())
        ]
    finally:
        depot.close()
        if explanations_depot:
            explanations_depot.close()
