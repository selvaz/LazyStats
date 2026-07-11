"""Shared data shapes. Stdlib-only, no pandas/numpy required."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ReturnDataset"]


@dataclass(frozen=True)
class ReturnDataset:
    """A long-form return panel, the input shape of every core statistic.

    Shape-compatible with ``lazytools.statistical_analysis.ReturnDataset`` —
    the LazyTools wrapper and this library exchange data without conversion.

    instruments: canonical lazydatacore ids (e.g. ``"ticker:SPY"``).
    rows: ``{"date": "YYYY-MM-DD", "<instrument>": float | None}`` per row,
          in date order; ``None`` marks a missing observation.
    metadata: free-form provenance (source, window, frequency, ...).
    """

    instruments: list[str]
    rows: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
