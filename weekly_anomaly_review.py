"""Saturday weekly review: verify the week's daily anomaly explanations
against the latest statistics, and look for a bigger picture across them
-- a new trend, confirmation of a regime change, or a new risk not visible
day to day.

Gathers every ``anomaly_explanation`` row since the last stored
``weekly_anomaly_review`` (or the last 7 days if none exists yet) plus the
freshest ``etf_daily_stats`` outlier list, and asks a ``ClaudeCodeEngine``
agent (same tools as the daily explainer -- LazyCrawler news/search first,
native web last resort) to:

  1. Verify each daily explanation still holds up -- confirmed / questionable
     / unverifiable, with a note.
  2. Synthesize: across the week's flagged anomalies as a set, does
     something emerge that no single day's finding showed on its own?

Saved to the same anomaly_explanations depot as the daily explainer
(kind="weekly_anomaly_review", series_key="weekly_anomaly_review").
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import lazytools.registry as lazytools_registry
from lazystats.io.depot import ResultDepot
from pydantic import BaseModel

DAILY_PRODUCED_BY = "lazystats.anomaly_explainer"
WEEKLY_PRODUCED_BY = "lazystats.weekly_anomaly_review"
WEEKLY_SERIES_KEY = "weekly_anomaly_review"
DAILY_SERIES_KEY = "anomaly_explanations"
ETF_STATS_PRODUCED_BY = "scheduled:etf_daily_stats"

REVIEW_PROMPT = """\
You are a buy-side macro/portfolio analyst doing the Saturday review of \
this week's flagged statistical anomalies (return outliers, volatility \
shifts, correlation shifts, beta divergences across 22 ETFs).

=== This week's daily explanations ({n_daily} items) ===
{daily_block}

=== Freshest statistical snapshot (as of {as_of}) ===
Return outliers, |z| >= {threshold}, last 5 trading days:
{outliers_block}

Do two things:

1. VERIFY each daily explanation above. For each one, decide:
   - "confirmed": the cited cause still holds up and is consistent with
     what you know now.
   - "questionable": something about it looks weak, contradicted, or you
     found reason to doubt it on a fresh check.
   - "unverifiable": you cannot confirm or refute it either way.
   Use your tools (search_cached/get_session_pages first, web_search
   second, native web search only as last resort) only when genuinely
   needed -- don't re-research everything from scratch, most explanations
   just need a sanity check against what you already have.

2. SYNTHESIZE across the whole week: looking at all the flagged anomalies
   together (not any single one), does a bigger pattern emerge? Specifically:
   - new_trends: a theme showing up repeatedly that wasn't obvious from any
     one day (e.g. a sector/asset-class quietly building unusual behavior).
   - regime_confirmations: does this week's pattern of anomalies confirm
     (or contradict) a volatility/correlation regime shift.
   - new_risks: a risk this week's anomalies point to that wasn't already
     flagged individually.
   Leave a list empty if genuinely nothing qualifies -- don't pad it.
"""


class VerificationVerdict(BaseModel):
    instrument: str
    anomaly_type: str
    date: str
    original_category: str
    verdict: Literal["confirmed", "questionable", "unverifiable"]
    note: str


class WeeklySynthesis(BaseModel):
    narrative: str
    new_trends: list[str]
    regime_confirmations: list[str]
    new_risks: list[str]


class WeeklyReview(BaseModel):
    verifications: list[VerificationVerdict]
    synthesis: WeeklySynthesis


def _last_week_end(depot: ResultDepot) -> str | None:
    entries = depot.list(produced_by=WEEKLY_PRODUCED_BY, cadence="stable", limit=1)
    if not entries:
        return None
    row = depot.load(entries[0]["result_id"])
    return row["payload"]["week_end"] if row else None


def gather_week(
    *, explanations_depot_path: str | None = None, stats_depot_path: str | None = None
) -> dict:
    """Every ``anomaly_explanation`` item since the last weekly review (or
    the last 7 days if none yet), plus the freshest outliers_last5 snapshot.
    """
    explanations_depot_path = explanations_depot_path or lazytools_registry.resolve_db("anomaly_explanations")
    stats_depot_path = stats_depot_path or lazytools_registry.resolve_db("lazystats_depot")

    explanations_depot = ResultDepot(explanations_depot_path)
    stats_depot = ResultDepot(stats_depot_path)
    try:
        last_end = _last_week_end(explanations_depot)
        since = last_end or (date.today() - timedelta(days=7)).isoformat()

        daily_items: list[dict] = []
        for entry in explanations_depot.list(produced_by=DAILY_PRODUCED_BY, cadence="stable", limit=200):
            row = explanations_depot.load(entry["result_id"])
            if not row or row["payload"]["date"] <= since:
                continue
            for item in row["payload"]["items"]:
                daily_items.append({**item, "source_result_id": entry["result_id"]})

        stats_entries = stats_depot.list(produced_by=ETF_STATS_PRODUCED_BY, cadence="stable", limit=1)
        latest_stats = stats_depot.load(stats_entries[0]["result_id"]) if stats_entries else None

        return {
            "week_start": since,
            "week_end": (latest_stats["payload"]["as_of"] if latest_stats else date.today().isoformat()),
            "daily_items": daily_items,
            "latest_outliers": latest_stats["payload"]["outliers_last5"] if latest_stats else None,
            "latest_as_of": latest_stats["payload"]["as_of"] if latest_stats else None,
        }
    finally:
        explanations_depot.close()
        stats_depot.close()


def _format_daily_block(items: list[dict]) -> str:
    parts = []
    for i, it in enumerate(items, start=1):
        ticker = it["instrument"]
        parts.append(
            f"{i}. {ticker} -- {it['anomaly_type']} on {it['date']} "
            f"[{it['category']}, {it['confidence']} confidence]\n"
            f"   {it['explanation']}"
        )
    return "\n".join(parts) if parts else "(none)"


def _format_outliers_block(outliers_payload: dict | None) -> str:
    if not outliers_payload or not outliers_payload.get("outliers"):
        return "(no outliers)"
    lines = []
    for o in outliers_payload["outliers"]:
        ticker = o["instrument"].replace("ticker:", "")
        lines.append(f"- {ticker} {o['date']}: z={o['z_score']:.2f} ({o['direction']})")
    return "\n".join(lines)


def review(week: dict) -> WeeklyReview:
    from lazybridge import Agent
    from lazybridge_claude_code import ClaudeCodeEngine
    from lazycrawler import CrawlerDB, DBConfig
    from lazycrawler.tools import CrawlerTools

    news_db_path = lazytools_registry.resolve_db("crawler_raw")
    crawler_tools = CrawlerTools(db=CrawlerDB(DBConfig(db_path=news_db_path)), content="pure") if news_db_path else None
    try:
        agent = Agent(
            engine=ClaudeCodeEngine(model="sonnet", web=True, request_timeout=300.0),
            name="weekly_anomaly_review",
            output=WeeklyReview,
            tools=crawler_tools.as_tools() if crawler_tools else [],
        )
        prompt = REVIEW_PROMPT.format(
            n_daily=len(week["daily_items"]),
            daily_block=_format_daily_block(week["daily_items"]),
            as_of=week["latest_as_of"] or "unknown",
            threshold=2.0,
            outliers_block=_format_outliers_block(week["latest_outliers"]),
        )
        env = agent(prompt)
        if not env.ok:
            raise RuntimeError(f"weekly_anomaly_review failed: {env.error}")
        return env.payload
    finally:
        if crawler_tools:
            crawler_tools.close()


def save_review(week: dict, result: WeeklyReview, *, depot_path: str | None = None) -> str:
    depot_path = depot_path or lazytools_registry.resolve_db("anomaly_explanations")
    if not depot_path:
        raise RuntimeError("ANOMALY_EXPLANATIONS_DB is not set -- see lazytools.registry.KNOWN_DBS")

    depot = ResultDepot(depot_path)
    try:
        instruments = sorted({it["instrument"].replace("ticker:", "") for it in week["daily_items"]})
        result_id = depot.save(
            kind="weekly_anomaly_review",
            produced_by=WEEKLY_PRODUCED_BY,
            instruments=instruments,
            payload={
                "week_start": week["week_start"],
                "week_end": week["week_end"],
                "n_daily_items": len(week["daily_items"]),
                "verifications": [v.model_dump() for v in result.verifications],
                "synthesis": result.synthesis.model_dump(),
            },
            provenance={
                "source": "anomaly_explanations depot + lazystats_depot.etf_daily_stats -> weekly_anomaly_review.review",
                "n_daily_items": len(week["daily_items"]),
            },
            cadence="stable",
            series_key=WEEKLY_SERIES_KEY,
        )
        return result_id
    finally:
        depot.close()
