"""Notebook-only local loaders — NEVER exposed to LLM profiles.

Plan v3.1 (§6, no-go list): "nessun loader file nel profilo LLM". These
helpers exist so a notebook can hand-feed a CSV or DataFrame into the core
statistics; the LazyTools bridge must not wrap them as tools, and the
boundary test in this repo asserts this module is not imported by anything
that could reach an agent.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from lazystats.models import ReturnDataset

__all__ = ["returns_from_csv", "returns_from_dataframe"]


def returns_from_csv(path: str | Path, *, date_column: str = "date") -> ReturnDataset:
    """A wide CSV (date column + one column per instrument) -> ReturnDataset.

    Empty cells become ``None``; column names are used verbatim as instrument
    ids (canonicalise upstream if needed).
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or date_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain a {date_column!r} column")
        instruments = [c for c in reader.fieldnames if c != date_column]
        rows: list[dict[str, Any]] = []
        for record in reader:
            entry: dict[str, Any] = {"date": record[date_column]}
            for instrument in instruments:
                cell = (record.get(instrument) or "").strip()
                entry[instrument] = float(cell) if cell else None
            rows.append(entry)
    return ReturnDataset(
        instruments=instruments,
        rows=rows,
        metadata={"source": "local-csv", "path": str(path), "notebook_only": True},
    )


def returns_from_dataframe(frame: Any) -> ReturnDataset:
    """A pandas DataFrame (DatetimeIndex or 'date' column, one column per
    instrument) -> ReturnDataset. pandas is imported lazily and only here."""
    import pandas as pd  # local import: pandas is NOT a lazystats dependency

    df = frame.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    instruments = [str(c) for c in df.columns]
    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        entry: dict[str, Any] = {
            "date": str(index.date() if hasattr(index, "date") else index)
        }
        for instrument in instruments:
            value = row[instrument]
            entry[instrument] = None if pd.isna(value) else float(value)
        rows.append(entry)
    return ReturnDataset(
        instruments=instruments,
        rows=rows,
        metadata={"source": "local-dataframe", "notebook_only": True},
    )
