#!/usr/bin/env python
"""Daily anomaly investigation: check deterministically for anything worth
investigating in the latest ``etf_daily_stats`` run; if so, investigate it
with an LLM agent and send the result via Telegram. If not, do nothing --
no agent call, no cost, no message.

Steps, per ``anomaly_gate.InvestigationTarget`` (one per flagged date):
    1. anomaly_explainer.investigate()       -> grounded findings
    2. anomaly_explainer.save_investigation() -> anomaly_explanations depot
    3. anomaly_report.render_html()          -> HTML file on disk
    4. Telegram send (if TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID configured)

Requires ``LAZYSTATS_RESULT_DEPOT_DB`` and ``ANOMALY_EXPLANATIONS_DB`` (see
LazyTools' ``KNOWN_DBS``). ``TELEGRAM_BOT_TOKEN``/``TELEGRAM_CHAT_ID`` are
optional -- the send step logs and skips (does not fail the job) when unset.

Usage:
    python run_daily_anomaly_investigation.py
"""

from __future__ import annotations

import os
import sys

from anomaly_explainer import investigate, save_investigation
from anomaly_gate import find_investigation_targets
from anomaly_report import render_html
from lazystats.io.depot import ResultDepot
import lazytools.registry as lazytools_registry

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _send_telegram(html_path: str, target_date: str, n_items: int) -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return f"Saved report at {html_path} (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID unset, send skipped)"
    from lazytools.connectors.telegram import TelegramClient

    client = TelegramClient.from_token(token)
    with open(html_path, "rb") as f:
        client.send_document(
            chat_id=chat_id,
            document=f.read(),
            filename=os.path.basename(html_path),
            caption=f"Anomaly investigation — {target_date} ({n_items} item(s))",
        )
    return f"Sent Telegram document: {os.path.basename(html_path)}"


def main() -> int:
    targets = find_investigation_targets()
    if not targets:
        print("No anomalies above threshold today -- nothing to investigate.")
        return 0

    depot_path = lazytools_registry.resolve_db("anomaly_explanations")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for target in targets:
        print(f"Investigating {target.date}: {[it.instrument.replace('ticker:', '') for it in target.items]}")
        result = investigate(target)
        result_id = save_investigation(target, result)

        depot = ResultDepot(depot_path)
        try:
            row = depot.load(result_id)
        finally:
            depot.close()

        html = render_html(row)
        html_path = os.path.join(REPORTS_DIR, f"anomaly_explanation_{target.date}_{result_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Saved: {html_path} (result_id={result_id})")

        print(_send_telegram(html_path, target.date, len(target.items)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
