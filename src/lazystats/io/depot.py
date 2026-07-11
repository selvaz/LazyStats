"""SQLite result depot with full provenance (plan v3.1 §4.4 / Fase 6).

Every stored analysis carries who produced it, over which instruments, with
which parameters and inputs — the acceptance requirement "provenance completa
nel depot". Stdlib-only (sqlite3 + json).

This is the seed of the depot that will absorb LazyHMM's richer SQLite store
(time series, model params, plots) when its engines migrate here; the schema
is versioned from day one for that reason.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

__all__ = ["ResultDepot"]

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS depot_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS analysis_results (
    result_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,          -- report | signal | model
    produced_by TEXT NOT NULL,          -- e.g. lazystats.core.return_volatility
    instruments TEXT NOT NULL,          -- JSON list of canonical ids
    payload     TEXT NOT NULL,          -- JSON result payload
    provenance  TEXT NOT NULL,          -- JSON: source, window, params, versions
    created_at  TEXT NOT NULL           -- ISO-8601 UTC
);
"""


class ResultDepot:
    """Append-only store of analysis results keyed by ``result_id``.

    ``provenance`` is mandatory at save time: a result without its inputs
    (source, window, parameters, library version) is not reproducible and is
    refused.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._con = sqlite3.connect(path)
        self._con.executescript(_SCHEMA)
        self._con.execute(
            "INSERT OR IGNORE INTO depot_meta (key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._con.commit()

    def save(
        self,
        *,
        kind: str,
        produced_by: str,
        instruments: list[str],
        payload: dict[str, Any],
        provenance: dict[str, Any],
    ) -> str:
        if not provenance:
            raise ValueError("provenance is mandatory: a result without its "
                             "inputs is not reproducible")
        result_id = f"res_{uuid.uuid4().hex[:12]}"
        self._con.execute(
            "INSERT INTO analysis_results VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                result_id,
                kind,
                produced_by,
                json.dumps(instruments),
                json.dumps(payload, default=str),
                json.dumps(provenance, default=str),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._con.commit()
        return result_id

    def load(self, result_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT result_id, kind, produced_by, instruments, payload, "
            "provenance, created_at FROM analysis_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "result_id": row[0],
            "kind": row[1],
            "produced_by": row[2],
            "instruments": json.loads(row[3]),
            "payload": json.loads(row[4]),
            "provenance": json.loads(row[5]),
            "created_at": row[6],
        }

    def list(self, *, produced_by: str | None = None, limit: int = 50
             ) -> list[dict[str, Any]]:
        """Bounded index of stored results (no payloads)."""
        where = ""
        params: tuple[Any, ...] = ()
        if produced_by:
            where, params = "WHERE produced_by = ?", (produced_by,)
        rows = self._con.execute(
            f"SELECT result_id, kind, produced_by, instruments, created_at "
            f"FROM analysis_results {where} ORDER BY created_at DESC LIMIT ?",
            (*params, max(1, int(limit))),
        ).fetchall()
        return [
            {
                "result_id": r[0],
                "kind": r[1],
                "produced_by": r[2],
                "instruments": json.loads(r[3]),
                "created_at": r[4],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._con.close()
