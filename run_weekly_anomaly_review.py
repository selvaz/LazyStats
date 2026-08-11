#!/usr/bin/env python
"""Saturday weekly anomaly review: verify the week's daily explanations
and look for a bigger picture across them, then send via Telegram.

Requires ``LAZYSTATS_RESULT_DEPOT_DB`` and ``ANOMALY_EXPLANATIONS_DB``.
``TELEGRAM_BOT_TOKEN``/``TELEGRAM_CHAT_ID`` are optional -- the send step
logs and skips (does not fail the job) when unset.

Usage:
    python run_weekly_anomaly_review.py
"""

from __future__ import annotations

import os

import lazytools.registry as lazytools_registry
from lazystats.io.depot import ResultDepot
from weekly_anomaly_review import gather_week, review, save_review
from weekly_review_report import render_html

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _send_telegram(html_path: str, week_start: str, week_end: str) -> str:
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
            caption=f"Weekly anomaly review — {week_start} to {week_end}",
        )
    return f"Sent Telegram document: {os.path.basename(html_path)}"


def main() -> int:
    week = gather_week()
    print(f"Reviewing {week['week_start']} -> {week['week_end']}: {len(week['daily_items'])} daily item(s)")

    result = review(week)
    result_id = save_review(week, result)

    depot_path = lazytools_registry.resolve_db("anomaly_explanations")
    depot = ResultDepot(depot_path)
    try:
        row = depot.load(result_id)
    finally:
        depot.close()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    html = render_html(row)
    html_path = os.path.join(REPORTS_DIR, f"weekly_anomaly_review_{week['week_end']}_{result_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {html_path} (result_id={result_id})")

    print(_send_telegram(html_path, week["week_start"], week["week_end"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
