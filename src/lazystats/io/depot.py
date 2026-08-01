"""SQLite result depot with full provenance (plan v3.1 §4.4 / Fase 6).

Every stored analysis carries who produced it, over which instruments, with
which parameters and inputs — the acceptance requirement "provenance completa
nel depot". Stdlib-only (sqlite3 + json).

This is the seed of the depot that will absorb LazyHMM's richer SQLite store
(time series, model params, plots) when its engines migrate here; the schema
is versioned from day one for that reason.

Schema v2 adds the two-cadence model that migration will need:
    - ``analysis_results`` gained ``cadence`` ('adhoc' | 'stable') and
      ``series_key`` so a stable/scheduled series (e.g. a daily regime
      monitor) can tie together the one row written per run.
    - ``stable_series_points`` holds the per-trading-date history of a stable
      series with vintage/append-on-change semantics: a past
      (as_of_date, estimation_date) reading is never overwritten, only
      superseded by a new row when the value for that date actually changes.
      This mirrors market-data-hub's hmm_regime_estimates revision logic.
    - ``analysis_detail`` is an opt-in table for heavy payloads (prediction
      arrays, residuals, plots) that a caller explicitly asks to persist
      alongside a result_id — never written automatically.
"""

from __future__ import annotations

import builtins
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

__all__ = ["ResultDepot"]

_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS depot_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS analysis_results (
    result_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,          -- open-ended: report | signal | model | regime | ols |
                                         -- lasso | ridge | ...
    produced_by TEXT NOT NULL,          -- e.g. lazystats.core.return_volatility
    instruments TEXT NOT NULL,          -- JSON list of canonical ids
    payload     TEXT NOT NULL,          -- JSON result payload
    provenance  TEXT NOT NULL,          -- JSON: source, window, params, versions
    created_at  TEXT NOT NULL           -- ISO-8601 UTC
);
CREATE TABLE IF NOT EXISTS stable_series_points (
    series_key      TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,   -- ISO date, the trading date this point is about
    estimation_date TEXT NOT NULL,   -- ISO date, when this point was computed/revised
    value_json      TEXT NOT NULL,   -- JSON-encoded compact value (e.g. state/label/prob_high_vol)
    result_id       TEXT,            -- informational only, no FK constraint enforced
    created_at      TEXT NOT NULL,
    PRIMARY KEY (series_key, as_of_date, estimation_date)
);
CREATE INDEX IF NOT EXISTS idx_ssp_series_asof ON stable_series_points (series_key, as_of_date);
CREATE TABLE IF NOT EXISTS analysis_detail (
    result_id   TEXT NOT NULL,
    detail_type TEXT NOT NULL,   -- e.g. 'predictions', 'residuals', 'state_sequence', 'plot'
    blob        BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (result_id, detail_type)
);
"""


class ResultDepot:
    """Append-only store of analysis results keyed by ``result_id``.

    ``provenance`` is mandatory at save time: a result without its inputs
    (source, window, parameters, library version) is not reproducible and is
    refused.

    Every result additionally carries a ``cadence``: ``"adhoc"`` (default)
    for a one-off analysis, or ``"stable"`` for a recurring/scheduled series
    (e.g. a daily regime monitor), in which case ``series_key`` identifies
    the series across runs. Per-trading-date history for a stable series
    lives in :attr:`stable_series_points`, written via
    :meth:`save_stable_point`.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._con = sqlite3.connect(path)
        self._con.executescript(_SCHEMA)
        self._migrate()
        self._con.execute(
            "INSERT OR REPLACE INTO depot_meta (key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._con.commit()

    def _migrate(self) -> None:
        """Idempotent ALTER-based migration: add v2 columns if not already present.

        ``CREATE TABLE IF NOT EXISTS`` above handles brand-new tables; existing
        ``analysis_results`` tables created under schema v1 need the two new
        columns added explicitly. Checking ``PRAGMA table_info`` first makes
        re-running this against an already-migrated file a no-op.
        """
        existing = {
            row[1]
            for row in self._con.execute("PRAGMA table_info(analysis_results)").fetchall()
        }
        if "cadence" not in existing:
            self._con.execute(
                "ALTER TABLE analysis_results ADD COLUMN cadence TEXT NOT NULL DEFAULT 'adhoc'"
            )
        if "series_key" not in existing:
            self._con.execute("ALTER TABLE analysis_results ADD COLUMN series_key TEXT")

    def save(
        self,
        *,
        kind: str,
        produced_by: str,
        instruments: list[str],
        payload: dict[str, Any],
        provenance: dict[str, Any],
        cadence: str = "adhoc",
        series_key: str | None = None,
    ) -> str:
        if not provenance:
            raise ValueError("provenance is mandatory: a result without its "
                             "inputs is not reproducible")
        if cadence not in ("stable", "adhoc"):
            raise ValueError(f"cadence must be 'stable' or 'adhoc', got {cadence!r}")
        if cadence == "stable" and not series_key:
            raise ValueError(
                "series_key is mandatory when cadence='stable': a stable "
                "series must be identifiable across runs"
            )
        if cadence != "stable":
            series_key = None
        result_id = f"res_{uuid.uuid4().hex[:12]}"
        self._con.execute(
            "INSERT INTO analysis_results "
            "(result_id, kind, produced_by, instruments, payload, provenance, "
            "created_at, cadence, series_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result_id,
                kind,
                produced_by,
                json.dumps(instruments),
                json.dumps(payload, default=str),
                json.dumps(provenance, default=str),
                datetime.now(UTC).isoformat(),
                cadence,
                series_key,
            ),
        )
        self._con.commit()
        return result_id

    def load(self, result_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT result_id, kind, produced_by, instruments, payload, "
            "provenance, created_at, cadence, series_key FROM analysis_results "
            "WHERE result_id = ?",
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
            "cadence": row[7],
            "series_key": row[8],
        }

    def list(
        self,
        *,
        produced_by: str | None = None,
        cadence: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Bounded index of stored results (no payloads)."""
        clauses: list[str] = []
        params: list[Any] = []
        if produced_by:
            clauses.append("produced_by = ?")
            params.append(produced_by)
        if cadence:
            clauses.append("cadence = ?")
            params.append(cadence)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._con.execute(
            f"SELECT result_id, kind, produced_by, instruments, created_at, "
            f"cadence, series_key FROM analysis_results {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            (*params, max(1, int(limit))),
        ).fetchall()
        return [
            {
                "result_id": r[0],
                "kind": r[1],
                "produced_by": r[2],
                "instruments": json.loads(r[3]),
                "created_at": r[4],
                "cadence": r[5],
                "series_key": r[6],
            }
            for r in rows
        ]

    # ── Stable series (vintage / append-on-change) ────────────────────────

    def save_stable_point(
        self,
        *,
        series_key: str,
        as_of_date: str,
        estimation_date: str,
        value: Any,
        result_id: str | None = None,
        compare_keys: builtins.list[str] | None = None,
    ) -> bool:
        """Append-on-change write of one trading date's reading.

        Looks up the existing row for ``(series_key, as_of_date)`` with the
        highest ``estimation_date`` that is not later than the one being
        written (i.e. this call's own row if it already exists, otherwise
        the most recent prior vintage) — never a later, out-of-order vintage
        that happens to exist for the same ``as_of_date``. If none exists, or
        the new ``value`` differs from that baseline, inserts/upserts a row
        (with ``estimation_date`` = the one passed in) and returns ``True``.
        If unchanged, does nothing and returns ``False``.

        A past ``(as_of_date, estimation_date)`` row is never overwritten by
        a *later* one — only superseded by a new row for a new
        ``estimation_date`` once the reading for that ``as_of_date`` actually
        differs. A rerun of the *same* ``estimation_date`` does replace that
        one row in place (see the upsert below).

        Args:
            compare_keys: If given, only these keys of ``value`` (and of the
                baseline it's compared against) decide whether the reading
                "changed" — e.g. a caller whose ``value`` also carries
                continuously-varying fields (probabilities, scores) that
                would otherwise make every call look like a change. The full
                ``value`` is still what gets stored either way. If a key is
                absent from either side, it's treated as ``None`` for the
                comparison rather than raising.
        """
        row = self._con.execute(
            "SELECT value_json FROM stable_series_points "
            "WHERE series_key = ? AND as_of_date = ? AND estimation_date <= ? "
            "ORDER BY estimation_date DESC LIMIT 1",
            (series_key, as_of_date, estimation_date),
        ).fetchone()

        def _comparable(v: Any) -> str:
            if compare_keys is not None:
                v = {k: v.get(k) for k in compare_keys} if isinstance(v, dict) else v
            return json.dumps(v, sort_keys=True, default=str)

        new_value_json = json.dumps(value, sort_keys=True, default=str)
        if row is not None:
            existing_value = json.loads(row[0])
            if _comparable(existing_value) == _comparable(value):
                return False
        # ON CONFLICT, not a plain INSERT: a same-day rerun (the estimation_date
        # passed in is identical to the row just compared above, not a new one)
        # whose refit produced a different value must replace that row in
        # place -- it's the same estimation event, not a new vintage -- rather
        # than violating the (series_key, as_of_date, estimation_date) UNIQUE
        # constraint and crashing the caller.
        self._con.execute(
            "INSERT INTO stable_series_points "
            "(series_key, as_of_date, estimation_date, value_json, result_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(series_key, as_of_date, estimation_date) DO UPDATE SET "
            "value_json=excluded.value_json, result_id=excluded.result_id, "
            "created_at=excluded.created_at",
            (
                series_key,
                as_of_date,
                estimation_date,
                new_value_json,
                result_id,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._con.commit()
        return True

    # NOTE: return-annotated as ``builtins.list`` (not bare ``list``) below —
    # this class already defines a method named ``list``, and mypy resolves
    # a bare ``list[...]`` forward-reference annotation against that class
    # member rather than the builtin type once it is in scope.
    def list_series_vintages(
        self, series_key: str, as_of_date: str
    ) -> builtins.list[dict[str, Any]]:
        """All revisions stored for one ``(series_key, as_of_date)``, oldest first."""
        rows = self._con.execute(
            "SELECT estimation_date, value_json, result_id FROM stable_series_points "
            "WHERE series_key = ? AND as_of_date = ? ORDER BY estimation_date ASC",
            (series_key, as_of_date),
        ).fetchall()
        return [
            {
                "estimation_date": r[0],
                "value": json.loads(r[1]),
                "result_id": r[2],
            }
            for r in rows
        ]

    def get_series_latest(
        self, series_key: str, *, since: str | None = None
    ) -> builtins.list[dict[str, Any]]:
        """Current best read of a whole series: one row per ``as_of_date``,
        taking only the value from its most recent ``estimation_date``.

        Optionally filtered to ``as_of_date >= since``. Ordered ascending by
        ``as_of_date``.
        """
        clause = "AND as_of_date >= ?" if since else ""
        params: tuple[Any, ...] = (series_key, since) if since else (series_key,)
        rows = self._con.execute(
            f"""
            SELECT s.as_of_date, s.estimation_date, s.value_json
            FROM stable_series_points s
            JOIN (
                SELECT as_of_date, MAX(estimation_date) AS max_estimation_date
                FROM stable_series_points
                WHERE series_key = ? {clause}
                GROUP BY as_of_date
            ) latest
              ON s.as_of_date = latest.as_of_date
             AND s.estimation_date = latest.max_estimation_date
            WHERE s.series_key = ?
            ORDER BY s.as_of_date ASC
            """,
            (*params, series_key),
        ).fetchall()
        return [
            {
                "as_of_date": r[0],
                "estimation_date": r[1],
                "value": json.loads(r[2]),
            }
            for r in rows
        ]

    # ── Detail blobs (opt-in, never written automatically) ────────────────

    def save_detail(self, result_id: str, detail_type: str, blob: bytes) -> None:
        """Persist a heavy opt-in payload alongside ``result_id``.

        ``INSERT OR REPLACE``: re-saving the same ``(result_id, detail_type)``
        replaces the stored blob. Nothing in this module calls this
        automatically — it is only invoked when a caller explicitly wants to
        persist a heavy artifact (predictions, residuals, state sequence, plot).
        """
        self._con.execute(
            "INSERT OR REPLACE INTO analysis_detail "
            "(result_id, detail_type, blob, created_at) VALUES (?, ?, ?, ?)",
            (result_id, detail_type, blob, datetime.now(UTC).isoformat()),
        )
        self._con.commit()

    def get_detail(self, result_id: str, detail_type: str) -> bytes | None:
        row = self._con.execute(
            "SELECT blob FROM analysis_detail WHERE result_id = ? AND detail_type = ?",
            (result_id, detail_type),
        ).fetchone()
        if row is None:
            return None
        return bytes(row[0])

    def close(self) -> None:
        self._con.close()
