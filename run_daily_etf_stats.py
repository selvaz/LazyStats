#!/usr/bin/env python
"""Daily ETF returns/volatility/correlation/outliers job.

A deterministic ``lazybridge.Plan`` pipeline -- no LLM in the loop.  Every
step is a plain Python callable; the Plan engine threads the running bundle
between steps as JSON text (``from_prev``), which is why each step accepts
and returns a JSON-serialisable dict.

Steps:
    fetch              -> load daily + weekly returns for INSTRUMENTS, plus
                           each instrument's classification (asset_class,
                           area) from market-data-hub's etf_classification --
                           embedded in the bundle (and the saved payload) so
                           the report can always be re-rendered from the
                           saved JSON alone, with no live DB access.
    volatility_short    } return_volatility on the last SHORT_WEEKS of the
    volatility_long     } weekly series vs. the last LONG_WEEKS
    volatility_1y       -> return_volatility on the last ONE_YEAR_WEEKS (52)
                           of the weekly series -- the baseline annualized
                           vol used by returns_table's vol_multiple, kept
                           distinct from short/long since it must line up
                           with the "1Y" horizon it's compared against
    correlation_short    } return_correlation, same short/long weekly split
    correlation_long     }
    outliers_last5      -> return_outliers on the daily series: the recent
                           individual events (last OUTLIER_WINDOW_DAYS trading
                           days) plus a per-day positive/negative count over
                           the last OUTLIER_CHART_DAYS trading days, for the
                           report's frequency chart
    returns_table       -> cumulative simple return per instrument over
                           RETURN_HORIZONS (1W/1M/3M/6M/YTD/1Y), from the
                           daily series, plus vol_multiple = return /
                           (volatility_1y * sqrt(horizon_days / 365)) --
                           how many "sigma" of the 1Y-annualized volatility
                           that horizon's return represents
    save_artifact       -> bundle everything into one ResultDepot row
                           (kind="report", cadence="stable",
                           series_key=SERIES_KEY); returns the canonical
                           saved row (same shape ResultDepot.load() returns)
    render_report       -> lazystats_report.render_html(row) -> an HTML
                           file on disk
    send_telegram       -> send that HTML file via Telegram, if configured

Requires ``LAZYSTATS_RESULT_DEPOT_DB`` (see LazyTools' ``KNOWN_DBS``) and
the ``contract`` extra (market-data-hub) to be installed. ``TELEGRAM_BOT_TOKEN``
/ ``TELEGRAM_CHAT_ID`` are optional -- the send step logs and skips (does not
fail the job) when they're unset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, timedelta

from lazybridge import Agent, Plan, Step
from market_data_hub.db.connection import get_conn

import lazytools.registry as lazytools_registry
from lazystats.core.returns import return_correlation, return_outliers, return_volatility
from lazystats.io.datahub import load_returns
from lazystats.io.depot import ResultDepot
from lazystats.models import ReturnDataset
from etf_stats_config import ConfigError, EtfStatsConfig, load_config
from etf_stats_report import render_html

# The preset -- which instruments, over which windows, above which outlier
# threshold -- is NOT here. It is a project choice, not a statistical method,
# and it now lives in the caller's own configuration file, passed with
# --config. See etf_stats_config.py for the contract and
# examples/etf_daily_stats.example.toml for the shape.
#
# CONFIG is populated once by main() before the plan runs. Deliberately no
# default: a run against an unstated universe is worse than one that refuses
# to start.
CONFIG: EtfStatsConfig | None = None


def cfg() -> EtfStatsConfig:
    """The active configuration, or a clear error if the run was not configured."""
    if CONFIG is None:
        raise RuntimeError(
            "no configuration loaded: run_daily_etf_stats.py requires --config "
            "<path-to.toml> (there is no default preset)"
        )
    return CONFIG


REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _ds_to_dict(ds: ReturnDataset) -> dict:
    return {"instruments": ds.instruments, "rows": ds.rows, "metadata": ds.metadata}


def _ds_from_dict(d: dict) -> ReturnDataset:
    return ReturnDataset(instruments=d["instruments"], rows=d["rows"], metadata=d["metadata"])


def _slice_last(ds: ReturnDataset, n_weeks: int) -> ReturnDataset:
    rows = ds.rows[-n_weeks:] if n_weeks < len(ds.rows) else ds.rows
    return ReturnDataset(instruments=ds.instruments, rows=rows, metadata=ds.metadata)


def _display_name(raw_name: str | None, ticker: str) -> str:
    """market-data-hub's instruments.name is stored as
    "CATEGORY | AREA | Proper Name" (the classification is already
    embedded, redundant with our own asset_class/area fields) -- keep
    only the last segment. Falls back to the ticker itself if no name
    is on file."""
    if not raw_name:
        return ticker
    return raw_name.rsplit("|", 1)[-1].strip() or ticker


def _instrument_meta(tickers: list[str]) -> list[dict]:
    """Each ticker's display name (market-data-hub's instruments table) and
    (asset_class, area) classification (etf_classification table) --
    embedded in the bundle/saved payload so ``etf_stats_report.render_html``
    never needs live DB access to reconstruct the report, only the saved
    JSON."""
    con = get_conn(read_only=True)
    try:
        placeholders = ",".join("?" for _ in tickers)
        cls_rows = con.execute(
            f"SELECT symbol, asset_class, area FROM etf_classification WHERE symbol IN ({placeholders})",
            tickers,
        ).fetchall()
        name_rows = con.execute(
            f"""
            SELECT l.symbol, i.name FROM listings l
            JOIN instruments i ON i.instrument_id = l.instrument_id
            WHERE l.symbol IN ({placeholders})
            """,
            tickers,
        ).fetchall()
    finally:
        con.close()
    classified = {symbol: {"asset_class": asset_class, "area": area} for symbol, asset_class, area in cls_rows}
    names = {symbol: name for symbol, name in name_rows}
    return [
        {
            "ticker": t,
            "name": _display_name(names.get(t), t),
            "asset_class": classified.get(t, {}).get("asset_class") or "UNKNOWN",
            "area": classified.get(t, {}).get("area") or "",
        }
        for t in tickers
    ]


def fetch(arg: str) -> dict:
    params = json.loads(arg)
    as_of = params["as_of"]
    long_start = (date.fromisoformat(as_of) - timedelta(weeks=cfg().long_weeks + 8)).isoformat()
    daily_start = (date.fromisoformat(as_of) - timedelta(days=cfg().daily_lookback_days)).isoformat()

    weekly = load_returns(list(cfg().instruments), start=long_start, end=as_of, frequency="W")
    daily = load_returns(list(cfg().instruments), start=daily_start, end=as_of, frequency="D")

    return {
        "as_of": as_of,
        "instruments": weekly.instruments,
        "instrument_meta": _instrument_meta(list(cfg().instruments)),
        "weekly": _ds_to_dict(weekly),
        "daily": _ds_to_dict(daily),
    }


def volatility_short(arg: str) -> dict:
    bundle = json.loads(arg)
    weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg().short_weeks)
    result = return_volatility(weekly, frequency="W")
    result["window_weeks"] = len(weekly.rows)
    bundle["volatility_short"] = result
    return bundle


def volatility_long(arg: str) -> dict:
    bundle = json.loads(arg)
    weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg().long_weeks)
    result = return_volatility(weekly, frequency="W")
    result["window_weeks"] = len(weekly.rows)
    bundle["volatility_long"] = result
    return bundle


def volatility_1y(arg: str) -> dict:
    bundle = json.loads(arg)
    weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg().one_year_weeks)
    result = return_volatility(weekly, frequency="W")
    result["window_weeks"] = len(weekly.rows)
    bundle["volatility_1y"] = result
    return bundle


def correlation_short(arg: str) -> dict:
    bundle = json.loads(arg)
    weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg().short_weeks)
    result = return_correlation(weekly, frequency="W")
    result["window_weeks"] = len(weekly.rows)
    bundle["correlation_short"] = result
    return bundle


def correlation_long(arg: str) -> dict:
    bundle = json.loads(arg)
    weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg().long_weeks)
    result = return_correlation(weekly, frequency="W")
    result["window_weeks"] = len(weekly.rows)
    bundle["correlation_long"] = result
    return bundle


def outliers_last5(arg: str) -> dict:
    bundle = json.loads(arg)
    daily = _ds_from_dict(bundle["daily"])
    full = return_outliers(daily, frequency="D", threshold=cfg().outlier_threshold)

    # return_outliers is a whole-sample z-score (no lookback window of its
    # own) -- trim its result to whichever trading-day window each part of
    # the report needs: the event list stays a tight last-week view, the
    # frequency chart looks back a full trading month.
    trading_days = sorted({row["date"] for row in daily.rows})
    recent_days = trading_days[-cfg().outlier_window_days:]
    chart_days = trading_days[-cfg().outlier_chart_days:]

    recent_set = set(recent_days)
    recent_outliers = [o for o in full["outliers"] if o["date"] in recent_set]

    chart_set = set(chart_days)
    daily_counts = {d: {"positive": 0, "negative": 0} for d in chart_days}
    for o in full["outliers"]:
        if o["date"] in chart_set:
            daily_counts[o["date"]][o["direction"]] += 1

    bundle["outliers_last5"] = {
        **full,
        "outliers": recent_outliers,
        "total_outliers": len(recent_outliers),
        "window_trading_days": recent_days,
    }
    bundle["outlier_daily_counts"] = {
        "window_trading_days": chart_days,
        "counts": daily_counts,
    }
    return bundle


def _cumulative_return(rows: list[dict], instrument: str, as_of: date, since: date) -> float | None:
    """exp(sum of log returns in (since, as_of]) - 1 -- None if no
    observations fall in the window (e.g. a horizon longer than the
    fetched daily history)."""
    total = 0.0
    n = 0
    for row in rows:
        d = date.fromisoformat(row["date"])
        if since < d <= as_of:
            value = row.get(instrument)
            if value is not None:
                total += value
                n += 1
    return math.exp(total) - 1 if n else None


def returns_table(arg: str) -> dict:
    """Per instrument, per RETURN_HORIZONS label: the cumulative return
    plus ``vol_multiple`` = return / (volatility_1y * sqrt(horizon_days
    / 365)) -- how many "sigma" of the 1Y-annualized volatility, scaled
    down to that horizon via the square-root-of-time rule, that horizon's
    actual return represents. Both values are saved explicitly (not left
    for the report to derive) since this payload is meant to be read
    directly by an LLM later, not just rendered."""
    bundle = json.loads(arg)
    daily = _ds_from_dict(bundle["daily"])
    as_of = date.fromisoformat(bundle["as_of"])
    vol_1y = bundle["volatility_1y"]["volatility"]

    table: dict[str, dict[str, dict[str, float | None]]] = {t: {} for t in daily.instruments}
    for label, days_back in ((h.label, h.days_back) for h in cfg().return_horizons):
        since = date(as_of.year - 1, 12, 31) if label == "YTD" else as_of - timedelta(days=days_back)
        horizon_days = (as_of - since).days
        for instrument in daily.instruments:
            ret = _cumulative_return(daily.rows, instrument, as_of, since)
            annualized_vol = vol_1y.get(instrument, {}).get("annualized_volatility")
            multiple = None
            if ret is not None and annualized_vol:
                horizon_vol = annualized_vol * math.sqrt(horizon_days / 365.0)
                multiple = ret / horizon_vol if horizon_vol > 0 else None
            table[instrument][label] = {"return": ret, "vol_multiple": multiple}

    bundle["returns_table"] = table
    return bundle


def save_artifact(arg: str) -> dict:
    """Persist the bundle and return the canonical saved row -- the exact
    shape ``ResultDepot.load()`` returns, and the only input
    ``etf_stats_report.render_html`` needs. Re-loading (rather than just
    hand-assembling the dict) guarantees the render step -- and any later,
    independent re-render from the depot -- see identically-shaped data."""
    bundle = json.loads(arg)
    depot_path = lazytools_registry.resolve_db("lazystats_depot")
    depot = ResultDepot(depot_path)
    try:
        result_id = depot.save(
            kind="report",
            produced_by="scheduled:etf_daily_stats",
            instruments=bundle["instruments"],
            payload={
                "as_of": bundle["as_of"],
                "instrument_meta": bundle["instrument_meta"],
                "volatility_short": bundle["volatility_short"],
                "volatility_long": bundle["volatility_long"],
                "volatility_1y": bundle["volatility_1y"],
                "correlation_short": bundle["correlation_short"],
                "correlation_long": bundle["correlation_long"],
                "outliers_last5": bundle["outliers_last5"],
                "outlier_daily_counts": bundle["outlier_daily_counts"],
                "returns_table": bundle["returns_table"],
            },
            provenance={
                "source": "lazystats.io.datahub.load_returns -> market-data-hub",
                "instruments": bundle["instruments"],
                **cfg().as_provenance(),
                "vol_multiple_formula": "return / (volatility_1y.annualized_volatility * sqrt(horizon_days / 365))",
                "as_of": bundle["as_of"],
            },
            cadence="stable",
            series_key=cfg().series_key,
        )
        row = depot.load(result_id)
    finally:
        depot.close()
    return row


def render_report(arg: str) -> dict:
    row = json.loads(arg)
    html = render_html(row)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f"etf_daily_stats_{row['payload']['as_of']}_{row['result_id']}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return {"result_id": row["result_id"], "html_path": out_path}


def send_telegram(arg: str) -> str:
    info = json.loads(arg)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return (
            f"Saved result_id={info['result_id']}; report at {info['html_path']} "
            "(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID unset, send skipped)"
        )
    from lazytools.connectors.telegram import TelegramClient

    client = TelegramClient.from_token(token)
    with open(info["html_path"], "rb") as f:
        client.send_document(
            chat_id=chat_id,
            document=f.read(),
            filename=os.path.basename(info["html_path"]),
            caption=f"ETF daily stats — {info['result_id']}",
        )
    return f"Saved result_id={info['result_id']}; report sent to Telegram ({info['html_path']})"


def build_plan() -> Plan:
    return Plan(
        Step(fetch, name="fetch"),
        Step(volatility_short, name="volatility_short"),
        Step(volatility_long, name="volatility_long"),
        Step(volatility_1y, name="volatility_1y"),
        Step(correlation_short, name="correlation_short"),
        Step(correlation_long, name="correlation_long"),
        Step(outliers_last5, name="outliers_last5"),
        Step(returns_table, name="returns_table"),
        Step(save_artifact, name="save_artifact"),
        Step(render_report, name="render_report"),
        Step(send_telegram, name="send_telegram"),
    )


def main() -> int:
    global CONFIG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the run configuration (TOML): instruments, windows, "
             "outlier threshold, return horizons. Required -- there is no "
             "default preset. See examples/etf_daily_stats.example.toml.",
    )
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Override the as-of date (YYYY-MM-DD); default: today")
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Write the rendered report here instead of ./reports -- lets a "
             "shadow run keep its output away from the live one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and render, but do not persist to the result depot or "
             "send anything. For comparing a candidate configuration against "
             "the live run without touching either.",
    )
    args = parser.parse_args()

    try:
        CONFIG = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    agent = Agent(engine=build_plan(), name="etf_daily_stats")
    env = agent(json.dumps({"as_of": args.as_of}))
    if env.error is not None:
        print(f"FAILED: {env.error.message}", file=sys.stderr)
        return 1
    print(env.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
