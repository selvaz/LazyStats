"""LLM investigation of one day's flagged statistical anomalies: why did
this happen?

Takes an ``anomaly_gate.InvestigationTarget`` (one date, one or more
flagged instruments/anomaly-types that share it) and asks a
``ClaudeCodeEngine`` agent to find and cite a real-world cause for each --
grounded in LazyCrawler's already-crawled news first (free), its cheap
DuckDuckGo-backed web_search second, native WebSearch/WebFetch only as a
last resort. An item with no groundable cause gets category="unclear"
rather than a confabulated explanation -- the prompt asks for that
explicitly.

Saved to the ``anomaly_explanations`` depot (a store of its own, separate
from lazystats_depot's deterministic quantitative results -- see
docs/decisions or ask why: this holds narrative/evidence content an LLM
produced, not computed statistics).
"""

from __future__ import annotations

from typing import Literal

import lazytools.registry as lazytools_registry
from lazystats.io.depot import ResultDepot
from pydantic import BaseModel

from anomaly_gate import InvestigationTarget

EXPLAINER_PRODUCED_BY = "lazystats.anomaly_explainer"
EXPLAINER_SERIES_KEY = "anomaly_explanations"

_CATEGORIES = (
    "monetary_policy",
    "macro_data",
    "geopolitical",
    "company_specific",
    "liquidity_technical",
    "unclear",
)

EXPLAIN_PROMPT = """\
You are a buy-side macro/portfolio analyst. On {date}, the following \
statistical anomalies were flagged by a deterministic analysis of 22 ETFs' \
returns, volatility, correlation and beta:

{items_block}

For EACH item above, find and cite the real-world cause. Rules:
1. Ground every explanation in a specific, dated source -- a news article, \
a data release, a central bank statement, an earnings report. Use your \
tools in this order of preference (cheapest first): search_cached and \
get_session_pages (LazyCrawler's already-crawled news, free, no network) \
first; web_search (LazyCrawler's DuckDuckGo-backed search, cheap) second; \
your own native web search/fetch tools only as a last resort.
2. If you cannot find a grounded cause after a genuine attempt, say so --
category="unclear", confidence="low", evidence=[], and explanation states \
plainly that no clear cause was found. Do NOT invent a plausible-sounding \
cause you have not actually verified.
3. category is one of: monetary_policy, macro_data, geopolitical, \
company_specific, liquidity_technical, unclear.
4. Don't over-research -- most items resolve from one or two good sources; \
move on once you have a grounded answer.

Return one finding per item above (same instrument/anomaly_type/date), in \
the same order.
"""


class EvidenceItem(BaseModel):
    source: str
    date: str | None = None
    detail: str


class AnomalyFinding(BaseModel):
    instrument: str
    anomaly_type: Literal["return_outlier", "volatility_shift", "correlation_shift", "beta_divergence"]
    date: str
    category: Literal[
        "monetary_policy", "macro_data", "geopolitical", "company_specific", "liquidity_technical", "unclear"
    ]
    confidence: Literal["high", "medium", "low"]
    explanation: str
    evidence: list[EvidenceItem]


class InvestigationResult(BaseModel):
    findings: list[AnomalyFinding]


def _format_item(i: int, item) -> str:
    ticker = item.instrument.replace("ticker:", "")
    lines = [f"{i}. {ticker} -- {item.anomaly_type} on {item.date}"]
    for k, v in item.detail.items():
        lines.append(f"   {k}: {v}")
    return "\n".join(lines)


def investigate(target: InvestigationTarget) -> InvestigationResult:
    """Run the explanation agent for one date's investigation target."""
    from lazybridge import Agent
    from lazybridge_claude_code import ClaudeCodeEngine
    from lazycrawler import CrawlerDB, DBConfig
    from lazycrawler.tools import CrawlerTools

    news_db_path = lazytools_registry.resolve_db("crawler_raw")
    # content="pure": cheap clean-text extraction, no LLM call per page.
    # The default "smart" mode needs OPENAI_API_KEY (unset in this
    # deployment) for its own extraction/sentiment step and fails silently
    # per-page otherwise -- the agent itself (Claude) reads the raw text
    # directly and reasons over it, no pre-summarization needed here.
    crawler_tools = CrawlerTools(db=CrawlerDB(DBConfig(db_path=news_db_path)), content="pure") if news_db_path else None
    try:
        agent = Agent(
            # request_timeout: research agents legitimately need more than
            # the 120s default -- several tool-using search rounds, not a
            # single text-synthesis call.
            # max_turns: a busy day can flag 5+ items, each needing its own
            # search/crawl round plus synthesis -- the 20-turn default was
            # observed (2026-08-05) to run out mid-investigation, which
            # surfaces as a Claude Agent SDK crash race (a tool call still
            # in flight when the turn-limit result lands) rather than a
            # clean "out of turns" error. Deterministic given the same
            # prompt/tools, so retrying alone doesn't help -- give it room.
            engine=ClaudeCodeEngine(model="sonnet", web=True, request_timeout=300.0, max_turns=40),
            name="anomaly_explainer",
            output=InvestigationResult,
            tools=crawler_tools.as_tools() if crawler_tools else [],
        )
        items_block = "\n".join(_format_item(i, item) for i, item in enumerate(target.items, start=1))
        prompt = EXPLAIN_PROMPT.format(date=target.date, items_block=items_block)
        env = agent(prompt)
        if not env.ok:
            raise RuntimeError(f"anomaly_explainer failed: {env.error}")
        return env.payload
    finally:
        if crawler_tools:
            crawler_tools.close()


def save_investigation(target: InvestigationTarget, result: InvestigationResult, *, depot_path: str | None = None) -> str:
    """Persist one date's investigation into the anomaly_explanations depot.

    Merges each finding with the original deterministic detail (z-score,
    band, delta, ...) from the gate, so the stored row is self-contained --
    both what the gate measured and what the agent concluded.
    """
    depot_path = depot_path or lazytools_registry.resolve_db("anomaly_explanations")
    if not depot_path:
        raise RuntimeError(
            "ANOMALY_EXPLANATIONS_DB is not set -- see lazytools.registry.KNOWN_DBS"
        )
    detail_by_key = {(it.instrument, it.anomaly_type, it.date): it.detail for it in target.items}
    items_payload = []
    for finding in result.findings:
        key = (finding.instrument, finding.anomaly_type, finding.date)
        # ticker: prefix is stripped in the LLM-facing prompt/output but the
        # gate's own items keep it -- try both so a merge never silently drops.
        detail = detail_by_key.get(key) or detail_by_key.get((f"ticker:{finding.instrument}", finding.anomaly_type, finding.date)) or {}
        items_payload.append(
            {
                "instrument": finding.instrument,
                "anomaly_type": finding.anomaly_type,
                "date": finding.date,
                "category": finding.category,
                "confidence": finding.confidence,
                "explanation": finding.explanation,
                "evidence": [e.model_dump() for e in finding.evidence],
                "detail": detail,
            }
        )

    depot = ResultDepot(depot_path)
    try:
        instruments = sorted({it.instrument.replace("ticker:", "") for it in target.items})
        result_id = depot.save(
            kind="anomaly_explanation",
            produced_by=EXPLAINER_PRODUCED_BY,
            instruments=instruments,
            payload={
                "date": target.date,
                "trigger_result_id": target.trigger_result_id,
                "items": items_payload,
            },
            provenance={
                "source": "lazystats_depot.etf_daily_stats -> anomaly_gate.find_investigation_targets -> anomaly_explainer.investigate",
                "trigger_result_id": target.trigger_result_id,
                "n_items": len(items_payload),
            },
            cadence="stable",
            series_key=EXPLAINER_SERIES_KEY,
        )
        return result_id
    finally:
        depot.close()
