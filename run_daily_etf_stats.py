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

from lazystats.core.returns import return_correlation, return_outliers, return_volatility
from lazystats.io.datahub import load_returns
from lazystats.models import ReturnDataset
from lazystats.etf_stats import ConfigError, EtfStatsConfig, load_config
from etf_stats_report import render_html

# The preset -- which instruments, over which windows, above which outlier
# threshold -- is NOT here. It is a project choice, not a statistical method,
# and it comes from the caller's own configuration file via --config. See
# lazystats.etf_stats for the contract and
# examples/etf_daily_stats.example.toml for the shape.


#: Where a LIVE run writes its rendered report. A shadow run never uses
#: this: --dry-run requires an explicit --output-dir and refuses this path.
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


def _make_analysis_steps(cfg: EtfStatsConfig) -> list:
    """The pure analysis steps, bound to one configuration.

    Closures rather than module globals: the configuration a run used is
    fixed when the plan is built, so two plans in one process cannot read
    each other's preset, and no step can be invoked without one.

    Nothing here persists, sends or writes to disk — that is what makes the
    same steps safe to reuse for both the live and the shadow plan.
    """

    def fetch(arg: str) -> dict:
        params = json.loads(arg)
        as_of = params["as_of"]
        long_start = (date.fromisoformat(as_of) - timedelta(weeks=cfg.long_weeks + 8)).isoformat()
        daily_start = (date.fromisoformat(as_of) - timedelta(days=cfg.daily_lookback_days)).isoformat()

        weekly = load_returns(list(cfg.instruments), start=long_start, end=as_of, frequency="W")
        daily = load_returns(list(cfg.instruments), start=daily_start, end=as_of, frequency="D")

        return {
            "as_of": as_of,
            "instruments": weekly.instruments,
            "instrument_meta": _instrument_meta(list(cfg.instruments)),
            "weekly": _ds_to_dict(weekly),
            "daily": _ds_to_dict(daily),
        }


    def volatility_short(arg: str) -> dict:
        bundle = json.loads(arg)
        weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg.short_weeks)
        result = return_volatility(weekly, frequency="W")
        result["window_weeks"] = len(weekly.rows)
        bundle["volatility_short"] = result
        return bundle


    def volatility_long(arg: str) -> dict:
        bundle = json.loads(arg)
        weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg.long_weeks)
        result = return_volatility(weekly, frequency="W")
        result["window_weeks"] = len(weekly.rows)
        bundle["volatility_long"] = result
        return bundle


    def volatility_1y(arg: str) -> dict:
        bundle = json.loads(arg)
        weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg.one_year_weeks)
        result = return_volatility(weekly, frequency="W")
        result["window_weeks"] = len(weekly.rows)
        bundle["volatility_1y"] = result
        return bundle


    def correlation_short(arg: str) -> dict:
        bundle = json.loads(arg)
        weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg.short_weeks)
        result = return_correlation(weekly, frequency="W")
        result["window_weeks"] = len(weekly.rows)
        bundle["correlation_short"] = result
        return bundle


    def correlation_long(arg: str) -> dict:
        bundle = json.loads(arg)
        weekly = _slice_last(_ds_from_dict(bundle["weekly"]), cfg.long_weeks)
        result = return_correlation(weekly, frequency="W")
        result["window_weeks"] = len(weekly.rows)
        bundle["correlation_long"] = result
        return bundle


    def outliers_last5(arg: str) -> dict:
        bundle = json.loads(arg)
        daily = _ds_from_dict(bundle["daily"])
        full = return_outliers(daily, frequency="D", threshold=cfg.outlier_threshold)

        # return_outliers is a whole-sample z-score (no lookback window of its
        # own) -- trim its result to whichever trading-day window each part of
        # the report needs: the event list stays a tight last-week view, the
        # frequency chart looks back a full trading month.
        trading_days = sorted({row["date"] for row in daily.rows})
        recent_days = trading_days[-cfg.outlier_window_days:]
        chart_days = trading_days[-cfg.outlier_chart_days:]

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
        for label, days_back in ((h.label, h.days_back) for h in cfg.return_horizons):
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

    return [
        Step(fetch, name="fetch"),
        Step(volatility_short, name="volatility_short"),
        Step(volatility_long, name="volatility_long"),
        Step(volatility_1y, name="volatility_1y"),
        Step(correlation_short, name="correlation_short"),
        Step(correlation_long, name="correlation_long"),
        Step(outliers_last5, name="outliers_last5"),
        Step(returns_table, name="returns_table"),
    ]


def _canonical_row(bundle: dict, cfg: EtfStatsConfig, result_id: str, created_at: str) -> dict:
    """The row shape ``ResultDepot.load()`` returns — built without a depot.

    The live path persists and re-loads (so the renderer sees exactly what a
    later independent re-read would see); the shadow path builds the same
    shape here. Keeping one constructor means the two paths cannot drift
    into rendering different structures.
    """
    return {
        "result_id": result_id,
        "kind": "report",
        "produced_by": cfg.produced_by,
        "instruments": bundle["instruments"],
        "payload": {
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
        "provenance": {
            "source": "lazystats.io.datahub.load_returns -> market-data-hub",
            "instruments": bundle["instruments"],
            **cfg.as_provenance(),
            "vol_multiple_formula": "return / (volatility_1y.annualized_volatility * sqrt(horizon_days / 365))",
            "as_of": bundle["as_of"],
        },
        "created_at": created_at,
        "cadence": "stable",
        "series_key": cfg.series_key,
    }


def _make_save_artifact(cfg: EtfStatsConfig):
    """LIVE ONLY. Persists to the result depot and re-loads the saved row."""

    def save_artifact(arg: str) -> dict:
        # Imported here, not at module level: loading this runner or building
        # a shadow plan must not pull the result depot or the DB registry
        # into the process. A shadow run should not even be able to reach
        # them by accident, and an import is a reachable path.
        import lazytools.registry as lazytools_registry
        from lazystats.io.depot import ResultDepot

        bundle = json.loads(arg)
        row_in = _canonical_row(bundle, cfg, result_id="", created_at="")
        depot_path = lazytools_registry.resolve_db("lazystats_depot")
        depot = ResultDepot(depot_path)
        try:
            result_id = depot.save(
                kind=row_in["kind"],
                produced_by=row_in["produced_by"],
                instruments=row_in["instruments"],
                payload=row_in["payload"],
                provenance=row_in["provenance"],
                cadence="stable",
                series_key=cfg.series_key,
            )
            row = depot.load(result_id)
        finally:
            depot.close()
        return row

    return save_artifact


def _make_write_shadow_outputs(cfg: EtfStatsConfig, output_dir: str):
    """SHADOW ONLY. Writes the canonical payload and its HTML to an explicit
    directory. Touches no database and sends nothing: this closure captures
    neither a depot nor a Telegram client, so there is no code path from
    here to production state."""

    def write_shadow_outputs(arg: str) -> dict:
        bundle = json.loads(arg)
        as_of = bundle["as_of"]
        result_id = f"shadow_{as_of}"
        row = _canonical_row(bundle, cfg, result_id=result_id, created_at=SHADOW_CREATED_AT)

        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"shadow_payload_{as_of}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=1, sort_keys=True)

        html_path = os.path.join(output_dir, f"shadow_{cfg.series_key}_{as_of}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(render_html(row))

        return {"result_id": result_id, "json_path": json_path, "html_path": html_path}

    return write_shadow_outputs


def _make_render_report(cfg: EtfStatsConfig):
    """LIVE ONLY. Writes into the production reports directory.

    A closure over the config for the same reason as the other stages: the
    report filename carries the series name, and a downstream consumer
    locates yesterday's report by that name rather than by timestamp.
    """

    def render_report(arg: str) -> dict:
        row = json.loads(arg)
        html = render_html(row)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out_path = os.path.join(
            REPORTS_DIR,
            f"{cfg.series_key}_{row['payload']['as_of']}_{row['result_id']}.html",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return {"result_id": row["result_id"], "html_path": out_path}

    return render_report


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


def _is_inside_reports_dir(candidate: str) -> bool:
    """Whether ``candidate`` is the production reports directory or under it.

    Compares canonical paths, not strings. ``abspath`` equality is not enough
    on Windows: a different case spells the same directory, and a junction or
    symlink reaches the same tree under another name. ``realpath`` resolves
    links, ``normcase`` folds case, and the ``commonpath`` check catches
    subdirectories — a shadow run writing into ``reports/shadow/`` would still
    be writing into production output.
    """
    reports = os.path.normcase(os.path.realpath(REPORTS_DIR))
    target = os.path.normcase(os.path.realpath(candidate))
    if target == reports:
        return True
    try:
        return os.path.commonpath([target, reports]) == reports
    except ValueError:
        # Different drives on Windows: commonpath raises rather than
        # returning something meaningless. Different drive means not inside.
        return False


def build_live_plan(cfg: EtfStatsConfig) -> Plan:
    """The production plan: analyse, persist to the depot, render into the
    production reports directory, notify."""
    return Plan(
        *_make_analysis_steps(cfg),
        Step(_make_save_artifact(cfg), name="save_artifact"),
        Step(_make_render_report(cfg), name="render_report"),
        Step(send_telegram, name="send_telegram"),
    )


def build_shadow_plan(cfg: EtfStatsConfig, output_dir: str) -> Plan:
    """The shadow plan: the same analysis, then write the canonical payload
    and its HTML into an explicit directory.

    It does not contain ``save_artifact``, ``render_report`` or
    ``send_telegram`` — not disabled versions of them, absent. A shadow run
    therefore has no code path to the result depot, to the production
    reports directory, or to Telegram, and that is checked by inspecting the
    plan's step names rather than by trusting a flag.
    """
    return Plan(
        *_make_analysis_steps(cfg),
        Step(_make_write_shadow_outputs(cfg, output_dir), name="write_shadow_outputs"),
    )


#: Fixed timestamp in shadow rows: a shadow payload is compared byte-for-byte
#: against another run's, and a wall-clock field would differ every time for
#: reasons that have nothing to do with the analysis.
SHADOW_CREATED_AT = "1970-01-01T00:00:00+00:00"

#: Step names that must never appear in a shadow plan.
_LIVE_ONLY_STEPS = frozenset({"save_artifact", "render_report", "send_telegram"})


def main() -> int:
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
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        # A shadow run must state where its output goes. Defaulting would
        # let it land in the production reports directory, which is the one
        # thing --dry-run exists to prevent.
        if not args.output_dir:
            print("CONFIG ERROR: --dry-run requires --output-dir", file=sys.stderr)
            return 2
        if _is_inside_reports_dir(args.output_dir):
            print(
                f"CONFIG ERROR: --output-dir must not be the production reports "
                f"directory, nor inside it ({REPORTS_DIR})",
                file=sys.stderr,
            )
            return 2
        out = os.path.abspath(args.output_dir)
        plan = build_shadow_plan(cfg, out)
        name = f"{cfg.series_key}_shadow"
    else:
        if args.output_dir:
            print(
                "CONFIG ERROR: --output-dir applies to --dry-run only; a live "
                "run writes to the production reports directory",
                file=sys.stderr,
            )
            return 2
        plan = build_live_plan(cfg)
        name = cfg.series_key

    agent = Agent(engine=plan, name=name)
    env = agent(json.dumps({"as_of": args.as_of}))
    if env.error is not None:
        print(f"FAILED: {env.error.message}", file=sys.stderr)
        return 1
    print(env.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
