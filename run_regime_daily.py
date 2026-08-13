#!/usr/bin/env python
"""Daily regime estimation, as a deterministic lazybridge Plan.

No LLM in the loop. Every step is a plain Python callable and the Plan engine
threads the running bundle between them as JSON text, which is why each step
accepts and returns a JSON-serialisable dict.

**The bundle carries identifiers, never series.** A step hands on the symbols,
the window and the depot keys — not the returns. With a hundred symbols and a
decade of daily history, threading the data itself would put megabytes through
every step boundary, and each step would be holding a copy of something that
belongs in one place. The prices are loaded inside the step that fits them and
are gone by the time it returns.

That is also what makes these usable as agent tools later. Each operation is an
ordinary typed function in ``lazystats.regimes``: ``Tool.wrap(fit_symbol)`` is
the whole implementation, with no wrapper class re-declaring its parameters.

The preset — which instruments, over which windows, with which fitting
parameters — comes from ``--config``. There is no default: a scheduled job
fitting an unstated universe is worse than one that refuses to start.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from lazybridge import Agent, Plan, Step

from lazystats.io.depot import ResultDepot
from lazystats.regimes.config import ConfigError, RegimeConfig, load_config
from lazystats.regimes.daily_payload import REPORT_KIND, REPORT_SERIES_KEY
from lazystats.regimes.daily_payload import build_payload as build_daily_payload
from lazystats.regimes.estimation import (
    PERIODS_PER_YEAR,
    PRODUCED_BY,
    PROVENANCE_SOURCE,
    fit_symbol,
)
from lazystats.regimes.persist import write_failure, write_fit
from lazystats.regimes.report import Revision, SymbolReport
from lazystats.regimes.report import render_html as render_report
from lazystats.regimes.series import series_key
from lazystats.regimes.tiers import tier_of, volatility_tiers


def _window_start(as_of: date, lookback_years: int | None) -> str:
    """The first date a window admits, or empty for all available history."""
    if lookback_years is None:
        return ""
    return (as_of - timedelta(days=365 * lookback_years)).isoformat()


def _make_plan_run(cfg: RegimeConfig, *, window: str, as_of: date,
                   market_db: str, production_db: str):
    """Resolve what this run will do, without touching prices or the depot."""

    def plan_run(arg: str) -> dict:
        chosen = cfg.window(window)
        return {
            "window": chosen.name,
            "variant": chosen.variant,
            "start": _window_start(as_of, chosen.lookback_years),
            "as_of": as_of.isoformat(),
            "symbols": list(cfg.instruments),
            "market_db": market_db,
            "production_db": production_db,
        }

    return plan_run


def _entry(cfg: RegimeConfig, fitted: dict, *, symbol: str,
           revisions: tuple[Revision, ...], changed_today: bool) -> SymbolReport:
    """One symbol's fit, as the report reads it."""
    diagnostics = fitted["diagnostics"]
    states = diagnostics.get("states") or []
    latest = fitted["readings"][-1] if fitted["readings"] else {}
    current = latest.get("state")
    labels = diagnostics.get("labels") or []
    chart = fitted.get("chart")

    return SymbolReport(
        symbol=symbol,
        name=cfg.names.get(symbol),
        n_states=int(diagnostics.get("n_states", 0)),
        current_state=current,
        current_label=labels[current] if current is not None and current < len(labels)
        else None,
        current_tier=tier_of(
            volatility_tiers([s["annualized_volatility"] for s in states]), current),
        is_high_vol=bool(latest.get("is_high_vol")),
        prob_high_vol=latest.get("prob_high_vol"),
        current_state_probs=tuple(latest.get("state_probs") or ()),
        changed_today=changed_today,
        states=tuple(states),
        transmat=tuple(tuple(row) for row in diagnostics.get("transmat") or ()),
        bic=diagnostics.get("bic"),
        loglik=diagnostics.get("loglik"),
        data_start=diagnostics.get("data_start"),
        data_end=diagnostics.get("data_end"),
        n_obs=diagnostics.get("n_obs"),
        chart=base64.b64decode(chart) if chart else None,
        revisions=revisions,
    )


def _revisions_for(depot: ResultDepot, series_key: str,
                   dates: tuple[str, ...]) -> tuple[Revision, ...]:
    """The dates whose regime call moved, with what it moved from.

    A revision is a date that had already been read once and now reads
    differently — the model reconsidering the past, which is the whole reason
    the readings are versioned rather than overwritten. A date being stored for
    the first time is simply new, and has one vintage.

    That prior vintage is the entire test. Its predecessor also excluded the
    newest trading date, on the reasoning that the newest date always writes and
    so cannot be a revision. That holds only while every run brings a new
    trading day: on a holiday, or any day the market did not open, the newest
    trading date is one already stored, and a genuine revision to it would have
    been silently dropped. The vintage count says the same thing on every other
    day and the right thing on that one.
    """
    revisions: list[Revision] = []
    for trading_date in dates:
        history = depot.list_series_vintages(series_key, trading_date)
        if len(history) < 2:
            continue
        old, new = history[-2], history[-1]
        revisions.append(Revision(
            trading_date=trading_date,
            old_state=int(old["value"]["state"]),
            new_state=int(new["value"]["state"]),
            old_prob_high_vol=old["value"].get("prob_high_vol"),
            new_prob_high_vol=new["value"].get("prob_high_vol"),
            old_estimation_date=old["estimation_date"],
            new_estimation_date=new["estimation_date"],
        ))
    return tuple(revisions)


def _make_fit_and_persist(cfg: RegimeConfig, *, depot_path: str, dry_run: bool,
                          report_path: Path | None, generated: str):
    """Fit each symbol, write it, and assemble the report — one at a time.

    The prices live only inside this step, and only for one symbol at a time:
    the bundle that comes out carries an outcome per symbol, not a series.

    The report is written **here**, rather than in a step of its own, and that
    is not an accident of convenience. Its charts can only be drawn from the
    fitted model, and a model neither survives a step boundary nor serialises;
    passing ninety base64 images on to a later step would put megabytes through
    a boundary meant for identifiers. So each chart is drawn as its model is
    fitted, the model is released, and only the finished page's path leaves the
    step.
    """

    def fit_and_persist(arg: str) -> dict:
        bundle = json.loads(arg)
        outcomes: list[dict] = []
        entries: list[SymbolReport] = []
        depot = ResultDepot(depot_path)
        try:
            _fit_all(bundle, depot, outcomes, entries)
        finally:
            depot.close()

        bundle["outcomes"] = outcomes
        # Small enough to travel: this record carries no images, which is the
        # whole reason it can be stored and read back later.
        bundle["daily_payload"] = build_daily_payload(
            entries, as_of=bundle["as_of"], periods_per_year=PERIODS_PER_YEAR,
            source=PROVENANCE_SOURCE)

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                render_report(entries,
                              as_of=datetime.strptime(bundle["as_of"], "%Y-%m-%d").date(),
                              generated=generated, window=bundle["window"]),
                encoding="utf-8")
            bundle["report"] = str(report_path)
        return bundle

    def _fit_all(bundle: dict, depot: ResultDepot, outcomes: list[dict],
                 entries: list[SymbolReport]) -> None:
        for symbol in bundle["symbols"]:
            key = series_key(
                symbol,
                market_db=bundle["market_db"],
                production_db=bundle["production_db"],
                variant=bundle["variant"],
            )
            try:
                fitted = fit_symbol(
                    symbol,
                    start=bundle["start"],
                    end=bundle["as_of"],
                    s_max=cfg.s_max,
                    n_starts=cfg.n_starts,
                    random_state=cfg.random_state,
                    with_chart=report_path is not None,
                )
            except Exception as exc:  # one symbol's failure must not end the run
                message = f"{type(exc).__name__}: {exc}"
                if not dry_run:
                    write_failure(depot, symbol=symbol, series_key=key,
                                  estimation_date=bundle["as_of"], error=message)
                outcomes.append({"symbol": symbol, "series_key": key,
                                 "status": "error", "detail": message})
                entries.append(SymbolReport(symbol=symbol, name=cfg.names.get(symbol),
                                            error=message))
                continue

            if dry_run:
                outcomes.append({"symbol": symbol, "series_key": key, "status": "ok",
                                 "n_states": fitted["diagnostics"]["n_states"],
                                 "points_written": 0, "detail": "dry run: nothing written"})
                entries.append(_entry(cfg, fitted, symbol=symbol, revisions=(),
                                      changed_today=False))
                continue

            written = write_fit(
                depot,
                symbol=fitted["symbol"],
                series_key=key,
                estimation_date=bundle["as_of"],
                diagnostics=fitted["diagnostics"],
                dates=fitted["dates"],
                readings=fitted["readings"],
                retro_days=cfg.retro_days,
            )
            outcomes.append({
                "symbol": symbol, "series_key": key, "status": "ok",
                "n_states": fitted["diagnostics"]["n_states"],
                "points_written": written.points_written,
                "detail": written.selection_reason,
            })

            newest = fitted["dates"][-1]
            entries.append(_entry(
                cfg, fitted, symbol=symbol,
                revisions=_revisions_for(depot, key, written.changed_dates),
                changed_today=newest in written.changed_dates,
            ))

    return fit_and_persist


def summarise(arg: str) -> dict:
    """Count what happened, and keep the failures legible."""
    bundle = json.loads(arg)
    outcomes = bundle["outcomes"]
    failures = [o for o in outcomes if o["status"] == "error"]
    bundle["summary"] = {
        "window": bundle["window"],
        "as_of": bundle["as_of"],
        "symbols": len(outcomes),
        "fitted": len(outcomes) - len(failures),
        "failed": len(failures),
        "points_written": sum(o.get("points_written", 0) for o in outcomes),
        "failures": [{"symbol": f["symbol"], "detail": f["detail"]} for f in failures],
    }
    return bundle


def _make_persist_report(*, depot_path: str, dry_run: bool):
    """Store the run's record, so the day can be re-read without refitting."""

    def persist_report(arg: str) -> dict:
        bundle = json.loads(arg)
        if dry_run:
            bundle["report_result_id"] = None
            return bundle

        payload = bundle["daily_payload"]
        depot = ResultDepot(depot_path)
        try:
            bundle["report_result_id"] = depot.save(
                kind=REPORT_KIND,
                produced_by=PRODUCED_BY,
                instruments=sorted(s["symbol"] for s in payload["symbols"]),
                payload=payload,
                provenance=payload["provenance"],
                cadence="stable",
                # One series per window: the eight-year run's record must not
                # upsert into the full-history one's.
                series_key=f"{REPORT_SERIES_KEY}:{bundle['window']}",
            )
        finally:
            depot.close()
        return bundle

    return persist_report


def build_plan(cfg: RegimeConfig, *, window: str, as_of: date, market_db: str,
               production_db: str, depot_path: str, dry_run: bool,
               report_path: Path | None = None, generated: str = "") -> Plan:
    """The pipeline: resolve, fit and write, store the run's record, then count."""
    return Plan(
        Step(_make_plan_run(cfg, window=window, as_of=as_of, market_db=market_db,
                            production_db=production_db), name="plan_run"),
        Step(_make_fit_and_persist(cfg, depot_path=depot_path, dry_run=dry_run,
                                   report_path=report_path, generated=generated),
             name="fit_and_persist"),
        Step(_make_persist_report(depot_path=depot_path, dry_run=dry_run),
             name="persist_report"),
        Step(summarise, name="summarise"),
    )


def write_daily_report(bundle: dict, *, depot_path: str, out_dir: Path) -> Path:
    """Render the run's stored record as the browsable report.

    A saved run is re-read from the depot rather than rendered from the bundle,
    so the page is a function of what was actually written. A dry run has
    nothing to re-read, so the same shape is assembled in memory and marked as
    unsaved — the page then says so, instead of showing a result id that would
    not be there tomorrow.
    """
    from lazystats.regimes.daily_render import render_html

    result_id = bundle.get("report_result_id")
    row = None
    if result_id:
        depot = ResultDepot(depot_path)
        try:
            row = depot.load(result_id)
        finally:
            depot.close()
    if row is None:
        row = {"result_id": "", "kind": REPORT_KIND, "produced_by": PRODUCED_BY,
               "cadence": "stable", "created_at": "(dry run — not saved)",
               "payload": bundle["daily_payload"],
               "provenance": bundle["daily_payload"]["provenance"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"regime_daily_{bundle['as_of']}"
    out_path = out_dir / (f"{stem}_{result_id}.html" if result_id else f"{stem}.html")
    out_path.write_text(render_html(row), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, metavar="PATH",
                   help="Regime configuration (TOML). Required: there is no default preset.")
    p.add_argument("--window", required=True, metavar="NAME",
                   help="Which declared window to fit, by name.")
    p.add_argument("--depot", required=True, metavar="PATH",
                   help="Result depot to write to.")
    p.add_argument("--market-db", required=True, metavar="PATH",
                   help="The market database the prices come from.")
    p.add_argument("--production-db", required=True, metavar="PATH",
                   help="Which database counts as production. When it differs from "
                        "--market-db the series are namespaced, so a staging run "
                        "cannot supersede production's history.")
    p.add_argument("--as-of", metavar="YYYY-MM-DD",
                   help="Estimation date (default: today).")
    p.add_argument("--dry-run", action="store_true",
                   help="Fit and report, write nothing to the depot.")
    p.add_argument("--report-dir", metavar="PATH",
                   help="Write both daily reports into this directory: the "
                        "chart-based one, and the browsable one rendered from "
                        "the run's stored record. Omitted, no charts are drawn "
                        "at all — which is most of the run's cost — and the "
                        "record is still stored.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = load_config(args.config)
        if args.window not in {w.name for w in cfg.windows}:
            raise ConfigError(
                f"--window {args.window!r} is not declared; the configuration has "
                f"{sorted(w.name for w in cfg.windows)}"
            )
    except ConfigError as exc:
        print(f"configuration: {exc}", file=sys.stderr)
        return 2

    as_of = (datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of
             else datetime.now().date())

    report_path = None
    if args.report_dir:
        report_path = (Path(args.report_dir)
                       / f"hmm_regime_report_{as_of.strftime('%Y%m%d')}.html")

    plan = build_plan(
        cfg,
        window=args.window,
        as_of=as_of,
        market_db=args.market_db,
        production_db=args.production_db,
        depot_path=str(Path(args.depot)),
        dry_run=args.dry_run,
        report_path=report_path,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    agent = Agent(engine=plan, name="regime_daily")
    bundle = json.loads(agent(json.dumps({"as_of": as_of.isoformat()})).text())

    summary = bundle["summary"]
    if bundle.get("report"):
        summary["chart_report"] = bundle["report"]
    if args.report_dir:
        summary["daily_report"] = str(write_daily_report(
            bundle, depot_path=str(Path(args.depot)), out_dir=Path(args.report_dir)))
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
