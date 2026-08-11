# -*- coding: utf-8 -*-
"""What an explanation of an anomaly must look like to be accepted.

The explaining itself is a language model's job; deciding whether what came
back is usable is not. This module holds the contract and the validator, so
a caller can reject a malformed or hallucinated result before it reaches a
report or a database.

The validator is deliberately strict about identity. A model asked about
five anomalies can return four, or six, or five for instruments nobody
asked about — and a report built from that would look complete while
describing something else. So every explanation must correspond to an
anomaly that was actually sent, one-to-one.

Nothing here calls a model, and no engine is constructed. The method is
public; which model, whether it browses, and where its output is stored are
operational choices that belong to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The categories an explanation may claim. Closed on purpose: a free-text
#: category cannot be grouped, filtered or trended, and a model will happily
#: invent one.
CATEGORIES = (
    "monetary_policy",
    "macro_data",
    "geopolitical",
    "company_specific",
    "liquidity_technical",
    "unclear",
)

#: Fields every explanation must carry, with their expected types.
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "instrument": str,
    "date": str,
    "anomaly_type": str,
    "category": str,
    "explanation": str,
    "confidence": (int, float),
}

#: Nothing outside this set may appear. An unknown field usually means the
#: model answered a different question, or the prompt drifted from the
#: contract — either way it should be seen, not silently dropped.
ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS) | {"evidence"}


class ExplanationError(ValueError):
    """The explanation does not satisfy the contract."""


@dataclass(frozen=True)
class Explanation:
    """One accepted explanation for one anomaly."""

    instrument: str
    date: str
    anomaly_type: str
    category: str
    explanation: str
    confidence: float
    evidence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument, "date": self.date,
            "anomaly_type": self.anomaly_type, "category": self.category,
            "explanation": self.explanation, "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def _validate_one(raw: Any, *, expected: set[tuple[str, str, str]], index: int) -> Explanation:
    where = f"explanation[{index}]"
    if not isinstance(raw, dict):
        raise ExplanationError(f"{where}: expected an object, got {type(raw).__name__}")

    unknown = set(raw) - ALLOWED_FIELDS
    if unknown:
        raise ExplanationError(f"{where}: unknown field(s) {sorted(unknown)}")

    for field, kind in REQUIRED_FIELDS.items():
        if field not in raw:
            raise ExplanationError(f"{where}: missing required field '{field}'")
        value = raw[field]
        # bool passes isinstance(int); a confidence of `true` is not 1.0.
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ExplanationError(
                f"{where}: '{field}' must be {getattr(kind, '__name__', kind)}, "
                f"got {type(value).__name__}"
            )

    for field in ("instrument", "date", "anomaly_type", "explanation"):
        if not raw[field].strip():
            raise ExplanationError(f"{where}: '{field}' must not be blank")

    if raw["category"] not in CATEGORIES:
        raise ExplanationError(
            f"{where}: category {raw['category']!r} is not one of {list(CATEGORIES)}"
        )

    confidence = float(raw["confidence"])
    if not (0.0 <= confidence <= 1.0):
        raise ExplanationError(f"{where}: 'confidence' must lie in [0, 1], got {confidence}")

    identity = (raw["instrument"], raw["date"], raw["anomaly_type"])
    if identity not in expected:
        raise ExplanationError(
            f"{where}: explains {identity}, which was not among the anomalies sent"
        )

    evidence = raw.get("evidence", [])
    if not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence):
        raise ExplanationError(f"{where}: 'evidence' must be a list of strings")

    return Explanation(
        instrument=raw["instrument"], date=raw["date"], anomaly_type=raw["anomaly_type"],
        category=raw["category"], explanation=raw["explanation"],
        confidence=confidence, evidence=tuple(evidence),
    )


def validate_explanations(raw: Any, *, anomalies: list[dict]) -> tuple[Explanation, ...]:
    """Check a model's response against the anomalies it was asked about.

    Args:
        raw: The decoded response — expected to be a list of objects.
        anomalies: The anomalies sent, as produced by
            :func:`lazystats.anomaly_gate.evaluate_gate`.

    Returns:
        One validated explanation per anomaly, in the order the anomalies
        were given, so a caller can zip the two without re-matching.

    Raises:
        ExplanationError: Any field, type, category, confidence range or
            identity is wrong, or the set of explanations does not
            correspond one-to-one with the anomalies sent. Partial results
            are rejected rather than trimmed: an explanation missing for
            one anomaly means the model did not do what was asked, and
            silently reporting the rest would hide that.
    """
    if not isinstance(raw, list):
        raise ExplanationError(f"expected a list of explanations, got {type(raw).__name__}")

    expected = {(a["instrument"], a["date"], a["anomaly_type"]) for a in anomalies}
    if len(expected) != len(anomalies):
        raise ExplanationError("the anomalies sent are not uniquely identifiable")

    validated = [_validate_one(item, expected=expected, index=i) for i, item in enumerate(raw)]

    seen = {(e.instrument, e.date, e.anomaly_type) for e in validated}
    if len(seen) != len(validated):
        raise ExplanationError("the same anomaly is explained more than once")
    if seen != expected:
        missing = sorted(expected - seen)
        raise ExplanationError(f"no explanation for {len(missing)} anomaly/anomalies: {missing}")

    by_identity = {(e.instrument, e.date, e.anomaly_type): e for e in validated}
    return tuple(by_identity[(a["instrument"], a["date"], a["anomaly_type"])] for a in anomalies)


__all__ = [
    "ALLOWED_FIELDS",
    "CATEGORIES",
    "REQUIRED_FIELDS",
    "Explanation",
    "ExplanationError",
    "validate_explanations",
]
