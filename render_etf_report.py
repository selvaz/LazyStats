#!/usr/bin/env python
"""Re-render the HTML report for any past ``etf_daily_stats`` depot row.

Demonstrates (and is the concrete tool for) the guarantee that the report
is always reconstructable from its saved JSON alone: this script only
reads a row back out of ``lazystats_depot`` and calls
``etf_stats_report.render_html`` -- no live market-data-hub access, no
recomputation.

Usage::

    python render_etf_report.py --result-id res_xxxxxxxxxxxx
    python render_etf_report.py --latest
    python render_etf_report.py --latest --out my_report.html
"""

from __future__ import annotations

import argparse
import sys

import lazytools.registry as lazytools_registry
from lazystats.io.depot import ResultDepot

from etf_stats_report import render_html


def _latest_result_id(depot: ResultDepot) -> str | None:
    results = depot.list(produced_by="scheduled:etf_daily_stats", cadence="stable", limit=1)
    return results[0]["result_id"] if results else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result-id", help="A specific lazystats_depot result_id to re-render")
    group.add_argument("--latest", action="store_true", help="Re-render the most recent etf_daily_stats run")
    parser.add_argument("--out", help="Output HTML path; default: reports/etf_daily_stats_<as_of>_<result_id>.html")
    args = parser.parse_args()

    depot_path = lazytools_registry.resolve_db("lazystats_depot")
    depot = ResultDepot(depot_path)
    try:
        if args.latest:
            result_id = _latest_result_id(depot)
            if result_id is None:
                print("No etf_daily_stats results found in the depot.", file=sys.stderr)
                return 1
        else:
            result_id = args.result_id

        row = depot.load(result_id)
        if row is None:
            print(f"No such result_id: {result_id!r}", file=sys.stderr)
            return 1
    finally:
        depot.close()

    html = render_html(row)
    out_path = args.out or f"reports/etf_daily_stats_{row['payload']['as_of']}_{row['result_id']}.html"
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Re-rendered {result_id} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
