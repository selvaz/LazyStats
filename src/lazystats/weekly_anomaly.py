"""The weekly review: verify the week's explanations, then look across them.

Two questions, and they are different. The first is retrospective — does
each daily explanation still hold up against what is known now. The second
is only answerable in aggregate: taken together, do the week's anomalies
show something no single day did.

What lives here is the method: which rows constitute a week, how they are
laid out for a reader, what a verdict may say, and what shape an acceptable
answer has. What does not live here is any identity, any path, any model.
The series names arrive in a configuration; the depot paths arrive as
arguments; the agent arrives already built.

That last point is what makes the deterministic half testable. Everything up
to and including the prompt is a pure function of the two depots and the
preset, so it can be compared exactly against the legacy implementation
without a language model being involved at all. Only :func:`review` needs
one, and it is handed in.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from lazystats.weekly_anomaly_config import WeeklyReviewConfig

REVIEW_PROMPT = """\
You are a buy-side macro/portfolio analyst doing the Saturday review of \
this week's flagged statistical anomalies (return outliers, volatility \
shifts, correlation shifts, beta divergences across {universe}).

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


class Depot(Protocol):
    """The little of a result store this module uses.

    Stated as a Protocol so the caller decides what a depot is. Nothing here
    opens one, and nothing here knows where one lives.
    """

    def list(self, *, produced_by: str | None = ..., cadence: str | None = ...,
             limit: int = ...) -> list[dict]: ...

    def load(self, result_id: str) -> dict | None: ...


def last_week_end(depot: Depot, config: WeeklyReviewConfig) -> str | None:
    """Where the previous review stopped, or None if there has not been one.

    This is the boundary the next week starts from. Reading it rather than
    assuming seven days is what stops a missed Saturday quietly dropping a
    week of explanations on the floor.
    """
    entries = depot.list(produced_by=config.weekly_produced_by, cadence="stable", limit=1)
    if not entries:
        return None
    row = depot.load(entries[0]["result_id"])
    return row["payload"]["week_end"] if row else None


def gather_week(*, explanations_depot: Depot, stats_depot: Depot,
                config: WeeklyReviewConfig, today: date | None = None) -> dict:
    """Every explanation since the last review, plus the freshest snapshot.

    Args:
        explanations_depot: Where daily explanations and past reviews live.
        stats_depot: Where the upstream statistics job writes.
        config: Identities and bounds. No defaults.
        today: The date the run is happening on. Passed in rather than read
            from the clock so a comparison can be repeated and get the same
            answer.

    Returns:
        A plain dictionary. Deliberately not a model: this is the boundary a
        shadow comparison checks byte for byte, and a plain structure is what
        both implementations can be asked to produce.
    """
    today = today or date.today()
    since = last_week_end(explanations_depot, config) or (
        today - timedelta(days=config.initial_lookback_days)
    ).isoformat()

    daily_items: list[dict] = []
    for entry in explanations_depot.list(produced_by=config.daily_produced_by,
                                         cadence="stable",
                                         limit=config.daily_scan_limit):
        row = explanations_depot.load(entry["result_id"])
        if not row or row["payload"]["date"] <= since:
            continue
        for item in row["payload"]["items"]:
            daily_items.append({**item, "source_result_id": entry["result_id"]})

    stats_entries = stats_depot.list(produced_by=config.upstream_produced_by,
                                     cadence="stable", limit=1)
    latest_stats = stats_depot.load(stats_entries[0]["result_id"]) if stats_entries else None

    return {
        "week_start": since,
        "week_end": (latest_stats["payload"]["as_of"] if latest_stats
                     else today.isoformat()),
        "daily_items": daily_items,
        "latest_outliers": (latest_stats["payload"]["outliers_last5"]
                            if latest_stats else None),
        "latest_as_of": latest_stats["payload"]["as_of"] if latest_stats else None,
    }


def format_daily_block(items: list[dict]) -> str:
    parts = []
    for i, it in enumerate(items, start=1):
        ticker = it["instrument"]
        parts.append(
            f"{i}. {ticker} -- {it['anomaly_type']} on {it['date']} "
            f"[{it['category']}, {it['confidence']} confidence]\n"
            f"   {it['explanation']}"
        )
    return "\n".join(parts) if parts else "(none)"


def format_outliers_block(outliers_payload: dict | None) -> str:
    if not outliers_payload or not outliers_payload.get("outliers"):
        return "(no outliers)"
    lines = []
    for o in outliers_payload["outliers"]:
        ticker = o["instrument"].replace("ticker:", "")
        lines.append(f"- {ticker} {o['date']}: z={o['z_score']:.2f} ({o['direction']})")
    return "\n".join(lines)


def build_prompt(week: dict, config: WeeklyReviewConfig) -> str:
    """The exact text the agent would be given.

    Separated from sending it, because the prompt is deterministic and the
    answer is not. A shadow comparison can check this to the character
    without a model existing anywhere in the process.
    """
    return REVIEW_PROMPT.format(
        universe=config.universe_description,
        n_daily=len(week["daily_items"]),
        daily_block=format_daily_block(week["daily_items"]),
        as_of=week["latest_as_of"] or "unknown",
        threshold=config.outlier_threshold,
        outliers_block=format_outliers_block(week["latest_outliers"]),
    )


def review(week: dict, config: WeeklyReviewConfig, *, agent) -> WeeklyReview:
    """Ask an already-built agent the week's two questions.

    The agent is injected. This module constructs no engine, opens no
    browser and reads no credential — which is what lets everything above be
    exercised, and compared, with no model in the process at all.

    Raises:
        RuntimeError: The agent reported a failure. Distinguished from a
            review that ran and found nothing to say.
    """
    envelope = agent(build_prompt(week, config))
    if not envelope.ok:
        raise RuntimeError(f"the weekly review agent failed: {envelope.error}")
    return envelope.payload


def review_payload(week: dict, result: WeeklyReview) -> dict:
    """The row body, built without a depot.

    One constructor, so a live run and a shadow run cannot drift into
    storing different structures.
    """
    return {
        "week_start": week["week_start"],
        "week_end": week["week_end"],
        "n_daily_items": len(week["daily_items"]),
        "verifications": [v.model_dump() for v in result.verifications],
        "synthesis": result.synthesis.model_dump(),
    }


def reviewed_instruments(week: dict) -> list[str]:
    return sorted({it["instrument"].replace("ticker:", "") for it in week["daily_items"]})


def save_review(week: dict, result: WeeklyReview, *, depot: Any,
                config: WeeklyReviewConfig) -> str:
    """Persist one review through a depot the caller opened."""
    return depot.save(
        kind="weekly_anomaly_review",
        produced_by=config.weekly_produced_by,
        instruments=reviewed_instruments(week),
        payload=review_payload(week, result),
        provenance={
            "source": (
                f"{config.daily_series_key} + {config.upstream_series_key} "
                f"-> weekly_anomaly.review"
            ),
            "n_daily_items": len(week["daily_items"]),
            "parameters": config.as_provenance(),
        },
        cadence="stable",
        series_key=config.weekly_series_key,
    )


__all__ = [
    "REVIEW_PROMPT",
    "Depot",
    "VerificationVerdict",
    "WeeklyReview",
    "WeeklySynthesis",
    "build_prompt",
    "format_daily_block",
    "format_outliers_block",
    "gather_week",
    "last_week_end",
    "review",
    "review_payload",
    "reviewed_instruments",
    "save_review",
]
