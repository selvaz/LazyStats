#!/usr/bin/env python
"""Compare two estimation windows' regimes, as a deterministic lazybridge Plan.

No LLM in the loop, and no fitting: both windows' readings are already in the
depot, written by ``run_regime_daily.py`` on each of them. This runner reads
them back, contrasts them, stores the verdict as its own depot row and renders
that row as a page.

**A pure read.** Nothing here opens the market database. ``--market-db`` and
``--production-db`` are still required because the depot's series keys are
namespaced by which database an estimate came from — the runner needs the paths
to know which series to *read*, and never opens either file. That is what lets
this job run on a machine with no access to the prices at all.

**Which two windows** comes from ``--comparison``, naming a comparison the
configuration declares. Neither side is privileged: the runner passes a baseline
and a candidate through, and everything below reports them by the names the
preset gave them. "Full history versus eight years" is one such pair, not the
shape of the method.

The bundle threaded between steps carries symbols, window names and depot keys
— never a series and never a fit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from lazybridge import Agent, Plan, Step

from lazystats.io.depot import ResultDepot
from lazystats.regimes.config import Comparison, ConfigError, RegimeConfig, load_config
from lazystats.regimes.estimation import PERIODS_PER_YEAR, PROVENANCE_SOURCE
from lazystats.regimes.retrieve import load_window_fit
from lazystats.regimes.series import series_key
from lazystats.regimes.window_comparison import (
    COMPARISON_KIND,
    COMPARISON_SERIES_KEY,
    build_payload,
)

#: The producer identity a stored comparison is written under, as the daily fit
#: has its own. Downstream selection depends on it.
PRODUCED_BY = "scheduled:run_regime_window_comparison"


def _make_plan_run(cfg: RegimeConfig, chosen: Comparison, *, as_of: str,
                   market_db: str, production_db: str):
    """Resolve which series the comparison will read, without opening anything."""

    def plan_run(arg: str) -> dict:
        baseline = cfg.window(chosen.baseline)
        candidate = cfg.window(chosen.candidate)
        return {
            "comparison": chosen.name,
            "as_of": as_of,
            "baseline_window": baseline.name,
            "candidate_window": candidate.name,
            "symbols": [
                {
                    "symbol": symbol,
                    "baseline_key": series_key(symbol, market_db=market_db,
                                               production_db=production_db,
                                               variant=baseline.variant),
                    "candidate_key": series_key(symbol, market_db=market_db,
                                                production_db=production_db,
                                                variant=candidate.variant),
                }
                for symbol in cfg.instruments
            ],
        }

    return plan_run


def _make_compare(*, depot_path: str):
    """Read both windows' stored fits and contrast them, symbol by symbol.

    The fits live only inside this step: what comes out is the comparison
    record, which carries verdicts and counts rather than any fitted series.
    """

    def compare(arg: str) -> dict:
        bundle = json.loads(arg)
        depot = ResultDepot(depot_path)
        try:
            readings = [
                (
                    entry["symbol"],
                    load_window_fit(depot, series_key=entry["baseline_key"],
                                    window=bundle["baseline_window"]),
                    load_window_fit(depot, series_key=entry["candidate_key"],
                                    window=bundle["candidate_window"]),
                )
                for entry in bundle["symbols"]
            ]
        finally:
            depot.close()

        bundle["payload"] = build_payload(
            readings,
            comparison=bundle["comparison"],
            baseline_window=bundle["baseline_window"],
            candidate_window=bundle["candidate_window"],
            as_of=bundle["as_of"],
            periods_per_year=PERIODS_PER_YEAR,
            source=PROVENANCE_SOURCE,
        )
        return bundle

    return compare


def _make_persist(*, depot_path: str, dry_run: bool):
    """Store the comparison as its own depot row, so it can be re-read later."""

    def persist(arg: str) -> dict:
        bundle = json.loads(arg)
        if dry_run:
            bundle["result_id"] = None
            return bundle

        payload = bundle["payload"]
        depot = ResultDepot(depot_path)
        try:
            bundle["result_id"] = depot.save(
                kind=COMPARISON_KIND,
                produced_by=PRODUCED_BY,
                instruments=sorted(s["symbol"] for s in bundle["symbols"]),
                payload=payload,
                provenance=payload["provenance"],
                cadence="stable",
                # One series per comparison: two comparisons declared by the
                # same preset must not upsert into each other's history.
                series_key=f"{COMPARISON_SERIES_KEY}:{bundle['comparison']}",
            )
        finally:
            depot.close()
        return bundle

    return persist


def build_plan(cfg: RegimeConfig, chosen: Comparison, *, as_of: str, market_db: str,
               production_db: str, depot_path: str, dry_run: bool) -> Plan:
    """The pipeline: resolve which series, read and compare, then store."""
    return Plan(
        Step(_make_plan_run(cfg, chosen, as_of=as_of, market_db=market_db,
                            production_db=production_db), name="plan_run"),
        Step(_make_compare(depot_path=depot_path), name="compare"),
        Step(_make_persist(depot_path=depot_path, dry_run=dry_run), name="persist"),
    )


def rendered_row(bundle: dict, *, depot_path: str) -> dict:
    """The stored row to render, or its unstored equivalent on a dry run.

    A saved comparison is re-read from the depot rather than rendered from the
    bundle, so the page is a function of what was actually written. A dry run
    has nothing to re-read, so the same shape is assembled in memory and marked
    as unsaved — the page then says so instead of showing a result id that
    would not be there tomorrow.
    """
    result_id = bundle.get("result_id")
    if result_id:
        depot = ResultDepot(depot_path)
        try:
            row = depot.load(result_id)
        finally:
            depot.close()
        if row is not None:
            return row

    return {
        "result_id": "",
        "kind": COMPARISON_KIND,
        "produced_by": PRODUCED_BY,
        "created_at": "(dry run — not saved)",
        "payload": bundle["payload"],
        "provenance": bundle["payload"]["provenance"],
    }


def send_telegram(out_path: Path, summary: dict, *, comparison: str, as_of: str) -> int:
    """Deliver the report, or say precisely why it could not be delivered.

    Imported here rather than at module scope: ``lazytools`` is not a dependency
    of this repository, and a run that does not ask for delivery must not need
    it installed.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram not configured: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
              file=sys.stderr)
        return 2

    from lazytools.connectors.telegram import TelegramClient

    text = (
        f"Regime window comparison '{comparison}' — {as_of}\n"
        f"{summary['compared']} symbols compared\n"
        f"Disagreements: {summary['disagree']}\n"
        f"Agreements: {summary['agree']}\n"
        f"Single-state: {summary['single_state']}\n"
        f"Missing: {summary['missing']}"
    )
    with TelegramClient.from_token(token) as client:
        client.send_message(chat_id=chat_id, text=text)
        client.send_document(chat_id=chat_id, document=out_path.read_bytes(),
                             filename=out_path.name, caption="Regime window comparison")
    print("Sent Telegram summary and report.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, metavar="PATH",
                   help="Regime configuration (TOML). Required: there is no default preset.")
    p.add_argument("--comparison", required=True, metavar="NAME",
                   help="Which declared comparison to run, by name.")
    p.add_argument("--depot", required=True, metavar="PATH",
                   help="Result depot to read the fits from and write the verdict to.")
    p.add_argument("--market-db", required=True, metavar="PATH",
                   help="The market database the fits were computed from. Never opened: "
                        "it identifies which series to read.")
    p.add_argument("--production-db", required=True, metavar="PATH",
                   help="Which database counts as production, so a staging run reads "
                        "its own namespaced series rather than production's.")
    p.add_argument("--out-dir", required=True, metavar="PATH",
                   help="Directory the HTML report is written to.")
    p.add_argument("--as-of", metavar="YYYY-MM-DD",
                   help="The date this comparison describes (default: today).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compare and render, write nothing to the depot and send nothing.")
    p.add_argument("--send", action="store_true",
                   help="Send the summary and the report to Telegram.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        cfg = load_config(args.config)
        declared = {c.name: c for c in cfg.comparisons}
        if args.comparison not in declared:
            raise ConfigError(
                f"--comparison {args.comparison!r} is not declared; the configuration "
                f"has {sorted(declared)}"
            )
    except ConfigError as exc:
        print(f"configuration: {exc}", file=sys.stderr)
        return 2

    chosen = declared[args.comparison]
    as_of = (datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of
             else datetime.now().date()).isoformat()
    depot_path = str(Path(args.depot))

    plan = build_plan(cfg, chosen, as_of=as_of, market_db=args.market_db,
                      production_db=args.production_db, depot_path=depot_path,
                      dry_run=args.dry_run)
    agent = Agent(engine=plan, name="regime_window_comparison")
    bundle = json.loads(agent(json.dumps({"as_of": as_of})).text())

    # Imported here so the module stays importable — and testable — without the
    # rendering ever running; it is the only part that touches the filesystem.
    from lazystats.regimes.window_comparison_render import render_html

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_id = bundle.get("result_id")
    stem = f"regime_window_comparison_{chosen.name}_{as_of}"
    out_path = out_dir / (f"{stem}_{result_id}.html" if result_id else f"{stem}.html")
    out_path.write_text(render_html(rendered_row(bundle, depot_path=depot_path)),
                        encoding="utf-8")

    summary = bundle["payload"]["summary"]
    print(json.dumps({**summary, "comparison": chosen.name, "as_of": as_of,
                      "result_id": result_id, "report": str(out_path)}, indent=2))

    if args.send and not args.dry_run:
        code = send_telegram(out_path, summary, comparison=chosen.name, as_of=as_of)
        if code:
            return code

    # A run that could compare nothing has not succeeded, whatever it wrote: every
    # symbol missing means one of the two windows was never fitted, and reporting
    # that as success is how a broken upstream job stays invisible.
    if summary["compared"] and summary["missing"] == summary["compared"]:
        print("every symbol was missing a fit on one side or the other", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
