# -*- coding: utf-8 -*-
"""
lazystats.regimes.db — SQLite persistence layer for the LazyHMM tool layer
===============================================================
Single-file depot. Drop-in replacement for the in-process _MODULE_STORE.

Call once at startup:
    from lazystats.regimes import init_regime_db
    init_regime_db("my_project.db")

After that, all LazyHMM tools persist automatically:
    load_time_series / load_ticker  → time_series table
    fit_regimes                      → model_results + state_sequences tables
    RegimeRun.save_all_plots_to_db  → plots table

LLM tools (register with lazybridge Tool.wrap):
    load_ticker, db_list_series, db_get_series_info,
    db_list_results, db_get_result_summary,
    db_list_plots, db_export_plot,
    db_compare_results, db_get_state_sequence,
    db_delete_series, db_delete_result,

External requirements: numpy, pandas, (yfinance — optional, only for load_ticker)
"""
from __future__ import annotations

import io
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

_SCHEMA_VERSION = 2

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS time_series (
    data_key      TEXT    PRIMARY KEY,
    source        TEXT    NOT NULL,
    columns_json  TEXT    NOT NULL,
    index_json    TEXT    NOT NULL,
    values_blob   BLOB    NOT NULL,
    n_rows        INTEGER NOT NULL,
    n_cols        INTEGER NOT NULL,
    date_start    TEXT,
    date_end      TEXT,
    fillna_method TEXT,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS series_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    data_key    TEXT    NOT NULL REFERENCES time_series(data_key) ON DELETE CASCADE,
    ticker      TEXT,
    column_name TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    interval    TEXT,
    date_start  TEXT,
    date_end    TEXT,
    extra_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sm_data_key ON series_metadata(data_key);
CREATE INDEX IF NOT EXISTS idx_sm_ticker   ON series_metadata(ticker);

CREATE TABLE IF NOT EXISTS model_results (
    result_key       TEXT    PRIMARY KEY,
    data_key         TEXT,
    mode             TEXT    NOT NULL,
    criterion        TEXT    NOT NULL,
    n_timesteps      INTEGER NOT NULL,
    series_meta_json TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mr_data_key ON model_results(data_key);

CREATE TABLE IF NOT EXISTS state_sequences (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    result_key         TEXT    NOT NULL REFERENCES model_results(result_key) ON DELETE CASCADE,
    series_name        TEXT    NOT NULL,
    S                  INTEGER NOT NULL,
    n_timesteps        INTEGER NOT NULL,
    states_blob        BLOB    NOT NULL,
    state_probs_blob   BLOB    NOT NULL,
    high_vol_blob      BLOB    NOT NULL,
    prob_high_vol_blob BLOB    NOT NULL,
    index_json         TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ss ON state_sequences(result_key, series_name);

CREATE TABLE IF NOT EXISTS plots (
    plot_key    TEXT    PRIMARY KEY,
    result_key  TEXT,
    data_key    TEXT,
    series_name TEXT,
    plot_type   TEXT    NOT NULL,
    title       TEXT,
    width_px    INTEGER,
    height_px   INTEGER,
    png_blob    BLOB    NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pl_result_key ON plots(result_key);

CREATE TABLE IF NOT EXISTS generic_store (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    written_at TEXT NOT NULL
);

-- Trained model parameters with data provenance. Lets a fitted model be
-- discovered and reloaded for inference on new data, indexed by the data it
-- was estimated on. The full JSON record is kept in record_json; the other
-- columns are denormalized for cheap discovery queries.
CREATE TABLE IF NOT EXISTS model_params (
    params_key      TEXT PRIMARY KEY,
    result_key      TEXT,
    data_key        TEXT,
    model           TEXT    NOT NULL,
    cov_type        TEXT    NOT NULL,
    shared_mean     INTEGER NOT NULL,
    layout          TEXT    NOT NULL,
    n_series        INTEGER NOT NULL,
    n_timesteps     INTEGER NOT NULL,
    date_start      TEXT,
    date_end        TEXT,
    criterion       TEXT,
    series_json     TEXT    NOT NULL,
    record_json     TEXT    NOT NULL,
    lazyhmm_version TEXT,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mp_data_key   ON model_params(data_key);
CREATE INDEX IF NOT EXISTS idx_mp_result_key ON model_params(result_key);

CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_version(version) VALUES (2);
"""


# ══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _numpy_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_numpy(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _df_to_blobs(df: pd.DataFrame) -> Tuple[bytes, str, str]:
    """Return (values_blob, columns_json, index_json)."""
    values_blob  = _numpy_to_blob(df.values.astype(np.float64))
    columns_json = json.dumps(list(df.columns))
    index_json   = json.dumps([str(i)[:26] for i in df.index])
    return values_blob, columns_json, index_json


def _blobs_to_df(values_blob: bytes, columns_json: str, index_json: str) -> pd.DataFrame:
    arr  = _blob_to_numpy(values_blob)
    cols = json.loads(columns_json)
    idx  = pd.to_datetime(json.loads(index_json), errors="coerce")
    return pd.DataFrame(arr, index=idx, columns=cols)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# RegimeDB CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RegimeDB:
    """SQLite-backed depot for time series, model results, and plots."""

    def __init__(self, db_path: str) -> None:
        self.db_path   = str(Path(db_path).expanduser().resolve())
        self._local    = threading.local()
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Thread-local SQLite connection with WAL mode and 15-second busy timeout."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            current = row["version"] if row else 0
            if current < _SCHEMA_VERSION:
                self._migrate(current)

    def _migrate(self, current_version: int) -> None:
        # The schema itself is (re)created idempotently by executescript with
        # CREATE TABLE IF NOT EXISTS, so migrations here only bump the recorded
        # version. v1 -> v2 added the model_params table (created above).
        with self._conn() as conn:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (_SCHEMA_VERSION,))

    # ── Time series I/O ──────────────────────────────────────────────────

    def write_series(
        self,
        data_key: str,
        df: pd.DataFrame,
        source: str = "file",
        fillna_method: str = "ffill",
        ticker_meta: Optional[List[Dict]] = None,
    ) -> None:
        values_blob, columns_json, index_json = _df_to_blobs(df)
        idx = df.index
        date_start = str(idx[0])[:10]  if len(idx) > 0 else None
        date_end   = str(idx[-1])[:10] if len(idx) > 0 else None
        now        = _now_iso()

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO time_series
                (data_key, source, columns_json, index_json, values_blob,
                 n_rows, n_cols, date_start, date_end, fillna_method, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (data_key, source, columns_json, index_json, values_blob,
                  len(df), len(df.columns), date_start, date_end, fillna_method, now))

            # Refresh ticker metadata
            conn.execute("DELETE FROM series_metadata WHERE data_key=?", (data_key,))
            if ticker_meta:
                for tm in ticker_meta:
                    conn.execute("""
                        INSERT INTO series_metadata
                        (data_key, ticker, column_name, source, interval,
                         date_start, date_end, extra_json)
                        VALUES (?,?,?,?,?,?,?,?)
                    """, (data_key,
                          tm.get("ticker"),
                          tm.get("column_name", ""),
                          tm.get("source", source),
                          tm.get("interval"),
                          tm.get("date_start", date_start),
                          tm.get("date_end", date_end),
                          json.dumps(tm.get("extra", {}))))

    def read_series(self, data_key: str) -> Optional[Dict[str, Any]]:
        """Returns {"Y": ndarray, "columns": list, "index": list} or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT values_blob, columns_json, index_json FROM time_series WHERE data_key=?",
                (data_key,)
            ).fetchone()
        if row is None:
            return None
        Y    = _blob_to_numpy(row["values_blob"])
        cols = json.loads(row["columns_json"])
        idx  = json.loads(row["index_json"])
        return {"Y": Y, "columns": cols, "index": idx}

    def series_exists(self, data_key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM time_series WHERE data_key=?", (data_key,)
            ).fetchone()
        return row is not None

    def list_series(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT data_key, source, columns_json, n_rows, n_cols,
                       date_start, date_end, fillna_method, created_at
                FROM time_series ORDER BY created_at DESC
            """).fetchall()
        return [
            {
                "data_key":     r["data_key"],
                "source":       r["source"],
                "columns":      json.loads(r["columns_json"]),
                "n_rows":       r["n_rows"],
                "n_cols":       r["n_cols"],
                "date_start":   r["date_start"],
                "date_end":     r["date_end"],
                "fillna_method": r["fillna_method"],
                "created_at":   r["created_at"],
            }
            for r in rows
        ]

    def get_series_info(self, data_key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM time_series WHERE data_key=?", (data_key,)
            ).fetchone()
            if row is None:
                return None
            tickers = conn.execute(
                "SELECT * FROM series_metadata WHERE data_key=?", (data_key,)
            ).fetchall()
        ticker_meta = [
            {k: r[k] for k in ("ticker", "column_name", "source", "interval",
                                "date_start", "date_end")}
            for r in tickers
        ]
        return {
            "data_key":     row["data_key"],
            "source":       row["source"],
            "columns":      json.loads(row["columns_json"]),
            "n_rows":       row["n_rows"],
            "n_cols":       row["n_cols"],
            "date_start":   row["date_start"],
            "date_end":     row["date_end"],
            "fillna_method": row["fillna_method"],
            "created_at":   row["created_at"],
            "ticker_meta":  ticker_meta,
        }

    def delete_series(self, data_key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM time_series WHERE data_key=?", (data_key,))
        return cur.rowcount > 0

    # ── Model result I/O ─────────────────────────────────────────────────

    def write_result(
        self,
        result_key: str,
        full_output: Dict[str, Any],
        data_key: str = "",
        index_json: Optional[str] = None,
    ) -> None:
        mode        = full_output.get("mode", "")
        criterion   = full_output.get("criterion", "")
        n_timesteps = int(full_output.get("n_timesteps", 0))
        series_data = full_output.get("series", {})
        now         = _now_iso()

        # Build compact series_meta_json (no T-length arrays)
        series_meta: Dict[str, Any] = {}
        for col, sd in series_data.items():
            series_meta[col] = {k: v for k, v in sd.items()
                                if k not in ("states", "state_probs",
                                             "high_vol_flag", "prob_high_vol")}

        # Prefer the index the fit already resolved (real dates) so the depot
        # round-trips dates even when the result is not linked to a data_key.
        if index_json is None:
            fo_index = full_output.get("index")
            if fo_index:
                index_json = json.dumps([str(x) for x in fo_index])

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_results
                (result_key, data_key, mode, criterion, n_timesteps,
                 series_meta_json, created_at)
                VALUES (?,?,?,?,?,?,?)
            """, (result_key, data_key or "", mode, criterion, n_timesteps,
                  json.dumps(series_meta), now))

            # Delete old state_sequences rows for this result_key
            conn.execute("DELETE FROM state_sequences WHERE result_key=?", (result_key,))

            for col, sd in series_data.items():
                S = int(sd.get("S", 0))
                T = n_timesteps

                states    = np.array(sd.get("states",        [0] * T), dtype=np.int32)
                sp        = sd.get("state_probs",   [[0.0] * S] * T)
                sp_arr    = np.array(sp, dtype=np.float64)
                hv        = np.array(sd.get("high_vol_flag", [0] * T), dtype=np.int32)
                phv       = np.array(sd.get("prob_high_vol", [0.0] * T), dtype=np.float64)

                # Index: try to get from stored data_key first, fallback to range
                if index_json is None:
                    if data_key:
                        # Load from time_series table if available
                        row = conn.execute(
                            "SELECT index_json FROM time_series WHERE data_key=?",
                            (data_key,)
                        ).fetchone()
                        idx_j = row["index_json"] if row else json.dumps(list(range(T)))
                    else:
                        idx_j = json.dumps(list(range(T)))
                else:
                    idx_j = index_json

                conn.execute("""
                    INSERT INTO state_sequences
                    (result_key, series_name, S, n_timesteps,
                     states_blob, state_probs_blob, high_vol_blob,
                     prob_high_vol_blob, index_json)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (result_key, col, S, T,
                      _numpy_to_blob(states), _numpy_to_blob(sp_arr),
                      _numpy_to_blob(hv),     _numpy_to_blob(phv),
                      idx_j))

    def read_result(self, result_key: str) -> Optional[Dict[str, Any]]:
        """Reconstruct the full dict that fit_regimes returned (with T-length lists)."""
        with self._conn() as conn:
            mr = conn.execute(
                "SELECT * FROM model_results WHERE result_key=?", (result_key,)
            ).fetchone()
            if mr is None:
                return None
            ss_rows = conn.execute(
                "SELECT * FROM state_sequences WHERE result_key=?", (result_key,)
            ).fetchall()

        series_meta = json.loads(mr["series_meta_json"])

        # Reconstruct series dict with T-length arrays from BLOBs
        top_index: Optional[list] = None
        for row in ss_rows:
            col = row["series_name"]
            if col not in series_meta:
                series_meta[col] = {}
            states    = _blob_to_numpy(row["states_blob"]).astype(int).tolist()
            sp_arr    = _blob_to_numpy(row["state_probs_blob"]).tolist()
            hv        = _blob_to_numpy(row["high_vol_blob"]).astype(int).tolist()
            phv       = _blob_to_numpy(row["prob_high_vol_blob"]).tolist()
            series_meta[col]["states"]        = states
            series_meta[col]["state_probs"]   = sp_arr
            series_meta[col]["high_vol_flag"] = hv
            series_meta[col]["prob_high_vol"]  = phv
            if top_index is None:                       # all series share one index
                try:
                    top_index = [str(x) for x in json.loads(row["index_json"])]
                except Exception:
                    top_index = None

        if top_index is None:
            top_index = [str(i) for i in range(int(mr["n_timesteps"]))]

        return {
            "result_key":  result_key,
            "data_key":    mr["data_key"],
            "mode":        mr["mode"],
            "criterion":   mr["criterion"],
            "n_timesteps": mr["n_timesteps"],
            "series":      series_meta,
            # top-level date index so get_regime_changes / get_current_regime
            "index":       top_index,
        }

    def result_exists(self, result_key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM model_results WHERE result_key=?", (result_key,)
            ).fetchone()
        return row is not None

    def list_results(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT result_key, data_key, mode, criterion, n_timesteps,
                       series_meta_json, created_at
                FROM model_results ORDER BY created_at DESC
            """).fetchall()
        results = []
        for r in rows:
            sm = json.loads(r["series_meta_json"])
            series_summary = [
                {
                    "name":          col,
                    "S":             int(d.get("S", 0)),
                    "bic":           float(d.get("bic", float("nan"))),
                    "current_label": d.get("current_label", ""),
                }
                for col, d in sm.items()
            ]
            results.append({
                "result_key":     r["result_key"],
                "data_key":       r["data_key"],
                "mode":           r["mode"],
                "criterion":      r["criterion"],
                "n_timesteps":    r["n_timesteps"],
                "created_at":     r["created_at"],
                "series_summary": series_summary,
            })
        return results

    def get_result_summary(self, result_key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_results WHERE result_key=?", (result_key,)
            ).fetchone()
        if row is None:
            return None
        series_meta = json.loads(row["series_meta_json"])
        return {
            "result_key":  result_key,
            "data_key":    row["data_key"],
            "mode":        row["mode"],
            "criterion":   row["criterion"],
            "n_timesteps": row["n_timesteps"],
            "created_at":  row["created_at"],
            "series":      series_meta,   # compact — no T-arrays
        }

    def delete_result(self, result_key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM model_results WHERE result_key=?", (result_key,)
            )
        return cur.rowcount > 0

    # ── Plot I/O ─────────────────────────────────────────────────────────

    def write_plot(
        self,
        fig,
        result_key: str  = "",
        data_key: str    = "",
        series_name: str = "",
        plot_type: str   = "custom",
        title: str       = "",
    ) -> str:
        """Render fig to PNG bytes and store in DB. Returns plot_key."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
        png_bytes = buf.getvalue()

        w_in, h_in = fig.get_size_inches()
        dpi         = fig.dpi
        width_px    = int(w_in * dpi)
        height_px   = int(h_in * dpi)

        ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_ser = (series_name or "all").replace(" ", "_").replace("/", "_")
        plot_key = f"{result_key or data_key}__{plot_type}__{safe_ser}__{ts}"
        now      = _now_iso()

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO plots
                (plot_key, result_key, data_key, series_name, plot_type,
                 title, width_px, height_px, png_blob, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (plot_key, result_key or None, data_key or None,
                  series_name or None, plot_type, title or None,
                  width_px, height_px, png_bytes, now))
        return plot_key

    def list_plots(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT plot_key, result_key, data_key, series_name,
                       plot_type, title, width_px, height_px, created_at
                FROM plots ORDER BY created_at DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_plot(self, plot_key: str) -> bytes:
        """Return the stored PNG bytes for a plot (no filesystem side effect).

        This is the in-process read path a downstream consumer (e.g. a report
        renderer embedding the chart) uses; ``export_plot`` stays the
        write-to-disk variant. Raises KeyError if the plot is absent.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT png_blob FROM plots WHERE plot_key=?", (plot_key,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Plot '{plot_key}' not found in DB.")
        return bytes(row["png_blob"])

    def export_plot(self, plot_key: str, output_path: str) -> str:
        path = Path(output_path)
        path.write_bytes(self.get_plot(plot_key))
        return str(path.resolve())

    def delete_plot(self, plot_key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM plots WHERE plot_key=?", (plot_key,))
        return cur.rowcount > 0

    # ── Generic key-value (backs _swrite/_sread/_slist) ──────────────────

    def generic_write(self, key: str, value: Any) -> None:
        """Smart routing: fit result → write_result; time series → write_series; else JSON."""
        # Case 0: parameter record (has provenance) → indexed model_params table
        if (isinstance(value, dict)
                and str(value.get("schema", "")).startswith("lazyhmm.params")):
            self.write_params(key, value)
            return

        # Case 1: output of fit_regimes (has "series" + "n_timesteps")
        if (isinstance(value, dict)
                and "series" in value
                and "n_timesteps" in value
                and isinstance(value.get("series"), dict)):
            data_key = value.get("data_key", "") or value.get("result_key", "")
            self.write_result(key, value, data_key=str(data_key))
            return

        # Case 2: output of load_time_series (has "Y" ndarray + "columns")
        if (isinstance(value, dict)
                and "Y" in value
                and isinstance(value.get("Y"), np.ndarray)):
            Y    = value["Y"]
            cols = value.get("columns", [f"col_{i}" for i in range(Y.shape[1] if Y.ndim > 1 else 1)])
            idx  = value.get("index", [str(i) for i in range(len(Y))])
            df   = pd.DataFrame(
                Y.reshape(len(Y), -1),
                index=pd.to_datetime(idx, errors="coerce"),
                columns=cols,
            )
            self.write_series(key, df, source="file")
            return

        # Case 3: fallback — store as JSON in generic_store
        now = _now_iso()
        try:
            value_json = json.dumps(value, default=str)
        except (TypeError, ValueError):
            value_json = json.dumps({"_repr": repr(value)[:1000]})
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO generic_store(key, value_json, written_at) VALUES (?,?,?)",
                (key, value_json, now),
            )

    def generic_read(self, key: str) -> Any:
        """Read from model_params → model_results → time_series → generic_store."""
        # 0. Try model_params (the JSON record is stored verbatim)
        rec = self.read_params(key)
        if rec is not None:
            return rec

        # 1. Try model_results (reconstruct full output including T-arrays)
        if self.result_exists(key):
            return self.read_result(key)

        # 2. Try time_series
        if self.series_exists(key):
            return self.read_series(key)

        # 3. Try generic_store
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value_json FROM generic_store WHERE key=?", (key,)
            ).fetchone()
        if row is not None:
            return json.loads(row["value_json"])

        raise KeyError(f"Key '{key}' not found in depot.")

    def generic_list(self) -> List[str]:
        with self._conn() as conn:
            mr  = [r[0] for r in conn.execute("SELECT result_key FROM model_results").fetchall()]
            ts  = [r[0] for r in conn.execute("SELECT data_key   FROM time_series").fetchall()]
            gs  = [r[0] for r in conn.execute("SELECT key        FROM generic_store").fetchall()]
            mp  = [r[0] for r in conn.execute("SELECT params_key FROM model_params").fetchall()]
        return sorted(set(mr) | set(ts) | set(gs) | set(mp))

    # ── Model parameters (trained models, indexed by provenance) ─────────────

    def write_params(self, params_key: str, record: Dict[str, Any]) -> None:
        """Persist a parameter record (with provenance) into model_params."""
        prov = record.get("provenance", {}) or {}
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_params
                (params_key, result_key, data_key, model, cov_type, shared_mean,
                 layout, n_series, n_timesteps, date_start, date_end, criterion,
                 series_json, record_json, lazyhmm_version, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    params_key,
                    record.get("result_key", ""),
                    prov.get("data_key", ""),
                    record.get("model", ""),
                    record.get("cov_type", ""),
                    1 if record.get("shared_mean") else 0,
                    record.get("layout", ""),
                    int(prov.get("n_series", len(record.get("series", [])))),
                    int(prov.get("n_timesteps", 0)),
                    prov.get("date_start", ""),
                    prov.get("date_end", ""),
                    prov.get("criterion", ""),
                    json.dumps(record.get("series", [])),
                    json.dumps(record, default=str),
                    record.get("lazyhmm_version", ""),
                    record.get("created_at", now),
                ),
            )

    def params_exists(self, params_key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM model_params WHERE params_key=?", (params_key,)
            ).fetchone()
        return row is not None

    def read_params(self, params_key: str) -> Optional[Dict[str, Any]]:
        """Return the stored parameter record, or None if absent."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT record_json FROM model_params WHERE params_key=?", (params_key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["record_json"])

    def list_params(self, data_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """List trained-model parameter records (compact metadata, no params).

        Optionally filter by the data_key the model was estimated on.
        """
        sql = (
            "SELECT params_key, result_key, data_key, model, cov_type, layout, "
            "n_series, n_timesteps, date_start, date_end, criterion, series_json, "
            "created_at FROM model_params"
        )
        args: Tuple[Any, ...] = ()
        if data_key:
            sql += " WHERE data_key=?"
            args = (data_key,)
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["series"] = json.loads(d.pop("series_json"))
            out.append(d)
        return out

    def delete_params(self, params_key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM model_params WHERE params_key=?", (params_key,))
        return cur.rowcount > 0


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE & API
# ══════════════════════════════════════════════════════════════════════════════

_DB: Optional[RegimeDB]  = None
_CACHE: Dict[str, Any]   = {}   # write-through in-process cache


def init_regime_db(db_path: str) -> RegimeDB:
    """Create / open the SQLite depot. Call once at startup.

    After this call, swrite/sread/slist route to SQLite + in-process cache.
    All LazyHMM tools persist automatically.

    Args:
        db_path: Path to the SQLite file (created if it does not exist).

    Returns:
        The RegimeDB instance (also stored in the module-level _DB).
    """
    global _DB, _CACHE
    _DB    = RegimeDB(db_path)
    _CACHE = {}
    return _DB


def get_db() -> RegimeDB:
    """Return the active RegimeDB. Raises RuntimeError if init_regime_db() was not called."""
    if _DB is None:
        raise RuntimeError(
            "regime_db not initialized. Call init_regime_db('my_project.db') first."
        )
    return _DB


def swrite(key: str, value: Any) -> None:
    """Write-through: update in-process cache AND SQLite depot."""
    _CACHE[key] = value
    if _DB is not None:
        _DB.generic_write(key, value)


def sread(key: str) -> Any:
    """Read from in-process cache first; fall back to SQLite depot."""
    if key in _CACHE:
        return _CACHE[key]
    if _DB is not None:
        val = _DB.generic_read(key)   # raises KeyError if not found
        _CACHE[key] = val
        return val
    raise KeyError(f"Key '{key}' not found in depot or cache.")


def slist() -> List[str]:
    """Union of in-process cache keys and all SQLite depot keys."""
    keys = set(_CACHE.keys())
    if _DB is not None:
        keys.update(_DB.generic_list())
    return sorted(keys)


def list_params(data_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discovery: list trained-model parameter records, optionally by data_key."""
    if _DB is None:
        return []
    return _DB.list_params(data_key)


def save_figure(
    fig,
    result_key: str  = "",
    data_key: str    = "",
    series_name: str = "",
    plot_type: str   = "custom",
    title: str       = "",
) -> str:
    """Render a matplotlib Figure to PNG and persist to DB. Returns plot_key."""
    return get_db().write_plot(fig, result_key, data_key, series_name, plot_type, title)


# ══════════════════════════════════════════════════════════════════════════════
# LLM TOOL: load_ticker
# ══════════════════════════════════════════════════════════════════════════════

def load_ticker(
    tickers: Annotated[
        list,
        "List of ticker symbols to download from Yahoo Finance, "
        "e.g. ['SPY', 'TLT', 'GLD']. Each ticker becomes one column in the stored data. "
        "If the data_key already exists in the depot and force_download=False, "
        "returns cached metadata without downloading.",
    ],
    start_date: Annotated[
        str,
        "Start date in ISO format 'YYYY-MM-DD', e.g. '2010-01-01'.",
    ],
    end_date: Annotated[
        str,
        "End date in ISO format 'YYYY-MM-DD', e.g. '2024-12-31'. "
        "Use today's date to get the most recent available data.",
    ],
    data_key: Annotated[
        str,
        "Key under which to store the downloaded data. "
        "Pass this to fit_regimes(data_key=...). Example: 'spy_tlt'.",
    ],
    column: Annotated[
        Literal["Close", "Adj Close", "Open", "High", "Low", "Volume"],
        "Which OHLCV column to extract. "
        "'Adj Close' is recommended for total-return analysis. "
        "Note: when auto_adjust=True (default in yfinance 0.2+), 'Close' already "
        "contains the adjusted price — fallback is applied automatically.",
    ] = "Adj Close",
    interval: Annotated[
        Literal["1d", "1wk", "1mo"],
        "Data frequency: '1d'=daily, '1wk'=weekly, '1mo'=monthly.",
    ] = "1wk",
    force_download: Annotated[
        bool,
        "If True, re-download from Yahoo Finance even if the data_key already "
        "exists in the depot. Use to refresh stale data or extend the date range.",
    ] = False,
) -> dict:
    """Download ticker price data from Yahoo Finance and store in the depot.

    Checks the depot first: if data_key already exists and force_download=False,
    returns the cached metadata immediately.  Otherwise downloads via yfinance,
    stores in the depot, and populates the in-process cache.

    LLM WORKFLOW:
        load_ticker(['SPY', 'TLT'], '2010-01-01', '2024-12-31', data_key='spy_tlt')
        fit_regimes(data_key='spy_tlt', result_key='spy_tlt_regimes')

    Returns:
        dict with keys:
            data_key (str): key to use in fit_regimes(data_key=...).
            tickers (list[str]): tickers requested.
            column (str): OHLCV column extracted.
            interval (str): data frequency.
            n_rows (int): number of timesteps.
            n_cols (int): number of columns (= len(tickers)).
            date_range (list[str]): [first_date, last_date].
            source (str): 'yfinance_cached' or 'yfinance_downloaded'.
            missing_pct (dict): {ticker: pct_missing} after download.
    """
    db = get_db()

    # Cache check
    if not force_download and db.series_exists(data_key):
        info = db.get_series_info(data_key)
        # Also populate in-process cache if missing
        if data_key not in _CACHE:
            stored = db.read_series(data_key)
            if stored:
                _CACHE[data_key] = stored
        return {
            "data_key":   data_key,
            "tickers":    tickers,
            "column":     column,
            "interval":   interval,
            "n_rows":     info["n_rows"],
            "n_cols":     info["n_cols"],
            "date_range": [info["date_start"], info["date_end"]],
            "source":     "yfinance_cached",
            "missing_pct": {},
        }

    # lazystats migration (plan v3.1): direct provider downloads are forbidden
    # here — market-data-hub is the only component allowed to fetch financial
    # data. The cached path above still works; a cache miss must go through
    # the hub instead of Yahoo.
    raise RuntimeError(
        "load_ticker no longer downloads directly in lazystats: ingest via "
        "market-data-hub (datahub_ensure_price_history) and load with "
        "lazystats.regimes.datasources.load_from_datahub(...) instead."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM TOOLS: DB INSPECTION
# ══════════════════════════════════════════════════════════════════════════════

def db_list_series() -> dict:
    """List all time series stored in the depot.

    Returns:
        dict with keys:
            series (list[dict]): each entry has data_key, source, columns,
                n_rows, n_cols, date_start, date_end, created_at.
            count (int): total number of stored series.
    """
    series = get_db().list_series()
    return {"series": series, "count": len(series)}


def db_get_series_info(
    data_key: Annotated[
        str,
        "Key of the stored time series to inspect. Get keys from db_list_series().",
    ],
) -> dict:
    """Return detailed metadata about a stored time series.

    Includes per-column ticker provenance when the series was loaded via load_ticker().

    Returns:
        dict with keys:
            data_key, source, columns, n_rows, n_cols,
            date_start, date_end, fillna_method, created_at,
            ticker_meta (list[dict]): per-column provenance.
    """
    info = get_db().get_series_info(data_key)
    if info is None:
        raise KeyError(f"data_key '{data_key}' not found in depot.")
    return info


def db_list_results() -> dict:
    """List all stored HMM fit results.

    Returns:
        dict with keys:
            results (list[dict]): each entry has result_key, data_key, mode,
                criterion, n_timesteps, created_at,
                series_summary (list of {name, S, bic, current_label}).
            count (int).
    """
    results = get_db().list_results()
    return {"results": results, "count": len(results)}


def db_get_result_summary(
    result_key: Annotated[
        str,
        "Key of the stored fit result. Get keys from db_list_results() "
        "or the result_key field returned by fit_regimes().",
    ],
) -> dict:
    """Return compact regime summary for a stored result (no large arrays).

    Contains regime_stats, transition_matrix, BIC, current regime label per series.
    Does NOT return T-length state sequences — use db_get_state_sequence() for those.

    Returns:
        Same structure as the compact output of fit_regimes() plus created_at.
    """
    summary = get_db().get_result_summary(result_key)
    if summary is None:
        raise KeyError(f"result_key '{result_key}' not found in depot.")
    return summary


def db_list_plots() -> dict:
    """List all stored plots.

    Returns:
        dict with keys:
            plots (list[dict]): each entry has plot_key, result_key, data_key,
                series_name, plot_type, title, width_px, height_px, created_at.
            count (int).
    """
    plots = get_db().list_plots()
    return {"plots": plots, "count": len(plots)}


def db_export_plot(
    plot_key: Annotated[
        str,
        "Key of the stored plot to export. Get keys from db_list_plots().",
    ],
    output_path: Annotated[
        str,
        "Absolute or relative file path where the PNG should be saved. "
        "The directory must already exist. Example: 'C:/reports/spy_regimes.png'.",
    ],
) -> dict:
    """Save a stored plot PNG to disk.

    Returns:
        dict with keys:
            plot_key (str), output_path (str), size_bytes (int), success (bool).
    """
    out = get_db().export_plot(plot_key, output_path)
    size = Path(out).stat().st_size
    return {"plot_key": plot_key, "output_path": out,
            "size_bytes": size, "success": True}


def db_compare_results(
    result_keys: Annotated[
        list,
        "List of result_key strings to compare side by side. "
        "All keys must exist in the depot. "
        "Example: ['spy_2y', 'spy_5y', 'spy_full'].",
    ],
) -> dict:
    """Compare regime statistics across multiple stored fit results.

    For each series that appears in at least one result, shows:
    - Number of regimes (S) per result
    - BIC per result
    - Current regime label per result
    - Per-regime vol, mean, occupancy_pct, expected_duration per result

    Returns:
        dict with keys:
            result_keys (list[str]): the keys compared.
            series (dict): per series name:
                S_by_result (dict): {result_key: S}.
                bic_by_result (dict): {result_key: float}.
                current_label_by_result (dict): {result_key: str}.
                regime_stats_by_result (dict): {result_key: list[dict]}.
    """
    db = get_db()
    summaries: Dict[str, Dict] = {}
    for rk in result_keys:
        s = db.get_result_summary(rk)
        if s is None:
            raise KeyError(f"result_key '{rk}' not found in depot.")
        summaries[rk] = s

    all_series: set = set()
    for s in summaries.values():
        all_series.update(s.get("series", {}).keys())

    comparison: Dict[str, Any] = {}
    for col in sorted(all_series):
        S_by         = {}
        bic_by       = {}
        cur_label_by = {}
        stats_by     = {}
        for rk, s in summaries.items():
            sd = s.get("series", {}).get(col)
            if sd is None:
                S_by[rk]         = None
                bic_by[rk]       = None
                cur_label_by[rk] = "not available"
                stats_by[rk]     = []
            else:
                S_by[rk]         = int(sd.get("S", 0))
                bic_by[rk]       = float(sd.get("bic", float("nan")))
                cur_label_by[rk] = sd.get("current_label", "")
                stats_by[rk]     = sd.get("regime_stats", [])
        comparison[col] = {
            "S_by_result":            S_by,
            "bic_by_result":          bic_by,
            "current_label_by_result": cur_label_by,
            "regime_stats_by_result": stats_by,
        }

    return {"result_keys": list(result_keys), "series": comparison}


def db_get_state_sequence(
    result_key: Annotated[
        str,
        "Key of the stored fit result.",
    ],
    series_name: Annotated[
        str,
        "Name of the series to retrieve. Must be one of the series in the result.",
    ],
    last_n: Annotated[
        int,
        "Return only the last N timesteps. Pass 0 to return all timesteps. "
        "Default 52 (approximately one year of weekly data).",
    ] = 52,
) -> dict:
    """Return the Viterbi state sequence and posteriors for a specific series.

    Returns:
        dict with keys:
            result_key (str), series_name (str), S (int),
            n_timesteps (int): total stored timesteps,
            last_n (int): actual number of timesteps returned,
            index (list[str]): date labels for the returned window,
            states (list[int]): Viterbi state per timestep (0 = lowest vol),
            high_vol (list[int]): 1 if in highest-vol regime else 0,
            prob_high_vol (list[float]): P(high-vol regime) per timestep,
            state_probs (list[list[float]]): full T×S posterior matrix.
    """
    db = get_db()
    with db._conn() as conn:
        row = conn.execute("""
            SELECT S, n_timesteps, states_blob, state_probs_blob,
                   high_vol_blob, prob_high_vol_blob, index_json
            FROM state_sequences
            WHERE result_key=? AND series_name=?
        """, (result_key, series_name)).fetchone()

    if row is None:
        raise KeyError(
            f"No state sequence found for result_key='{result_key}', "
            f"series_name='{series_name}'."
        )

    T          = int(row["n_timesteps"])
    S          = int(row["S"])
    n          = min(last_n, T) if last_n > 0 else T
    states     = _blob_to_numpy(row["states_blob"]).astype(int).tolist()[-n:]
    sp_arr     = _blob_to_numpy(row["state_probs_blob"]).tolist()[-n:]
    hv         = _blob_to_numpy(row["high_vol_blob"]).astype(int).tolist()[-n:]
    phv        = _blob_to_numpy(row["prob_high_vol_blob"]).tolist()[-n:]
    idx        = json.loads(row["index_json"])[-n:]

    return {
        "result_key":   result_key,
        "series_name":  series_name,
        "S":            S,
        "n_timesteps":  T,
        "last_n":       n,
        "index":        idx,
        "states":       states,
        "high_vol":     hv,
        "prob_high_vol": phv,
        "state_probs":  sp_arr,
    }


def db_delete_series(
    data_key: Annotated[str, "Key of the stored time series to delete."],
) -> dict:
    """Delete a stored time series from the depot.

    Returns:
        dict with keys: deleted (str), success (bool), remaining_count (int).
    """
    db      = get_db()
    ok      = db.delete_series(data_key)
    _CACHE.pop(data_key, None)
    return {"deleted": data_key, "success": ok,
            "remaining_count": len(db.list_series())}


def db_delete_result(
    result_key: Annotated[str, "Key of the stored fit result to delete."],
) -> dict:
    """Delete a stored fit result (and its state_sequences rows) from the depot.

    Returns:
        dict with keys: deleted (str), success (bool), remaining_count (int).
    """
    db = get_db()
    ok = db.delete_result(result_key)
    _CACHE.pop(result_key, None)
    return {"deleted": result_key, "success": ok,
            "remaining_count": len(db.list_results())}


# ══════════════════════════════════════════════════════════════════════════════
# LLM TOOL: compute_returns
# ══════════════════════════════════════════════════════════════════════════════

def compute_returns(
    data_key: Annotated[
        str,
        "Key of the stored price series (from load_ticker or load_time_series). "
        "The stored data must contain price levels — this tool converts them to returns.",
    ],
    output_key: Annotated[
        str,
        "Key under which to store the computed returns. "
        "Pass this key to fit_regimes(data_key=...). "
        "Example: 'spy_tlt_returns'. If empty, overwrites data_key in place.",
    ] = "",
    method: Annotated[
        Literal["log", "pct", "diff"],
        "'log': log-returns  ln(P_t / P_{t-1})  — recommended for most HMM use. "
        "'pct': simple percentage returns  (P_t - P_{t-1}) / P_{t-1}. "
        "'diff': arithmetic differences  P_t - P_{t-1}  — useful for spread/level series.",
    ] = "log",
    periods: Annotated[
        int,
        "Number of periods for the return calculation. "
        "1 = one-period returns (standard). "
        "4 = monthly returns on weekly data (rolling 4-week). "
        "52 = annual returns on weekly data.",
    ] = 1,
    drop_na: Annotated[
        bool,
        "If True (default), drop the first `periods` rows that become NaN after differencing. "
        "If False, fill with 0.0.",
    ] = True,
) -> dict:
    """Compute returns from a stored price series and save the result.

    ALWAYS call this before fit_regimes() when your data contains price levels
    (from load_ticker or load_time_series with raw prices).  HMMs fitted on
    price levels will detect spurious regimes driven by the trend, not volatility.

    LLM WORKFLOW:
        load_ticker(['SPY','TLT'], '2010-01-01', '2024-12-31', data_key='prices')
        compute_returns(data_key='prices', output_key='returns', method='log')
        fit_regimes(data_key='returns', result_key='regimes')

    Args:
        data_key: Key of the stored price series.
        output_key: Key for the computed returns (defaults to data_key + '_returns').
        method: 'log', 'pct', or 'diff'.
        periods: Return horizon in data frequency units.
        drop_na: Drop first `periods` NaN rows (True) or fill with 0 (False).

    Returns:
        dict with keys:
            output_key (str): key to use in fit_regimes(data_key=...).
            input_key (str): original data_key.
            method (str): method used.
            periods (int): periods used.
            n_rows (int): timesteps in the output (= input - periods if drop_na).
            n_cols (int): number of series.
            columns (list[str]): column names.
            date_range (list[str]): [first_date, last_date] of the return series.
            sample_stats (dict): per column: mean, std, min, max of the computed returns.
    """
    db      = get_db()
    stored  = db.read_series(data_key)
    if stored is None:
        raise KeyError(f"data_key '{data_key}' not found in depot.")

    Y    = stored["Y"]                            # (T, k) price array
    cols = stored["columns"]
    idx  = stored["index"]

    # Reconstruct DataFrame with proper DatetimeIndex
    df = pd.DataFrame(Y, index=pd.to_datetime(idx, errors="coerce"), columns=cols)

    if method == "log":
        ret = np.log(df / df.shift(periods))
    elif method == "pct":
        ret = df.pct_change(periods=periods)
    elif method == "diff":
        ret = df.diff(periods=periods)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log', 'pct', or 'diff'.")

    if drop_na:
        ret = ret.iloc[periods:]
    else:
        ret = ret.fillna(0.0)

    out_key = output_key or (data_key + "_returns")

    # Store computed returns
    db.write_series(out_key, ret, source="computed",
                    fillna_method="none",
                    ticker_meta=[
                        {"ticker": None, "column_name": c,
                         "source": "computed",
                         "interval": None,
                         "extra": {"from": data_key, "method": method,
                                   "periods": periods}}
                        for c in cols
                    ])

    # Populate cache
    _CACHE[out_key] = {
        "Y":       ret.values.astype(float),
        "columns": list(ret.columns),
        "index":   [str(i)[:10] for i in ret.index],
    }

    # Sample statistics
    sample_stats = {}
    for c in ret.columns:
        col_data = ret[c].dropna()
        sample_stats[c] = {
            "mean": round(float(col_data.mean()), 6),
            "std":  round(float(col_data.std()),  6),
            "min":  round(float(col_data.min()),  6),
            "max":  round(float(col_data.max()),  6),
        }

    date_range: List[str] = []
    if len(ret) > 0:
        date_range = [str(ret.index[0])[:10], str(ret.index[-1])[:10]]

    return {
        "output_key":   out_key,
        "input_key":    data_key,
        "method":       method,
        "periods":      periods,
        "n_rows":       int(len(ret)),
        "n_cols":       int(len(ret.columns)),
        "columns":      list(ret.columns),
        "date_range":   date_range,
        "sample_stats": sample_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LLM TOOL: merge_series
# ══════════════════════════════════════════════════════════════════════════════

def merge_series(
    data_keys: Annotated[
        list,
        "List of data_key strings to merge. Each must exist in the depot. "
        "Example: ['spy_returns', 'tlt_returns', 'gld_returns']. "
        "The order determines the column order in the merged dataset.",
    ],
    output_key: Annotated[
        str,
        "Key under which to store the merged dataset. "
        "Pass this to fit_regimes(data_key=..., model='joint_full'). "
        "Example: 'spy_tlt_gld_returns'.",
    ],
    align: Annotated[
        Literal["inner", "outer"],
        "'inner': keep only timesteps present in ALL series (intersection). "
        "Recommended when series have different start/end dates. "
        "'outer': keep all timesteps, fill missing values with NaN then forward-fill. "
        "Use when you want to preserve the full history of the longest series.",
    ] = "inner",
    column_names: Annotated[
        list,
        "Optional list of new column names for the merged dataset. "
        "If empty, uses the original column names from each series. "
        "Length must match the total number of columns across all data_keys.",
    ] = None,
) -> dict:
    """Merge multiple stored series into one dataset for multivariate HMM estimation.

    Use when you have series loaded separately (different files, different tickers
    downloaded at different times) that you want to fit jointly with model='joint_full'.

    LLM WORKFLOW (separate sources → joint multivariate fit):
        load_ticker(['SPY'], '2010-01-01', '2024-12-31', data_key='spy_prices')
        load_time_series('vix_history.csv', ['VIX'], data_key='vix')
        compute_returns(data_key='spy_prices', output_key='spy_ret')
        merge_series(['spy_ret', 'vix'], output_key='spy_vix', align='inner')
        fit_regimes(data_key='spy_vix', model='joint_full', result_key='joint_regimes')

    LLM WORKFLOW (all from Yahoo Finance → joint fit):
        load_ticker(['SPY', 'TLT', 'GLD'], '2010-01-01', '2024-12-31', data_key='multi')
        # Already T×3 — no merge needed, go straight to fit_regimes

    Args:
        data_keys: List of depot keys to merge.
        output_key: Key for the merged dataset.
        align: 'inner' (intersection) or 'outer' (union, forward-filled).
        column_names: Optional rename of merged columns.

    Returns:
        dict with keys:
            output_key (str): key to use in fit_regimes(data_key=...).
            source_keys (list[str]): input data_keys.
            align (str): alignment used.
            n_rows (int): timesteps in merged dataset.
            n_cols (int): total columns.
            columns (list[str]): column names in merged dataset.
            date_range (list[str]): [first, last] date of merged series.
            dropped_rows (int): rows dropped due to alignment (inner only).
    """
    db = get_db()

    frames: List[pd.DataFrame] = []
    for dk in data_keys:
        stored = db.read_series(dk)
        if stored is None:
            raise KeyError(f"data_key '{dk}' not found in depot.")
        Y    = stored["Y"]
        cols = stored["columns"]
        idx  = stored["index"]
        df   = pd.DataFrame(Y, index=pd.to_datetime(idx, errors="coerce"), columns=cols)
        frames.append(df)

    how      = "inner" if align == "inner" else "outer"
    merged   = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how=how, rsuffix="_dup")
        # Drop duplicate columns (same name from different sources)
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged   = merged.drop(columns=dup_cols)

    if align == "outer":
        merged = merged.ffill().bfill()

    n_before = sum(len(f) for f in frames) // len(frames)  # avg input rows
    n_after  = len(merged)
    dropped  = max(0, n_before - n_after)

    # Optional rename
    if column_names:
        if len(column_names) != len(merged.columns):
            raise ValueError(
                f"column_names has {len(column_names)} entries but merged dataset "
                f"has {len(merged.columns)} columns: {list(merged.columns)}"
            )
        merged.columns = list(column_names)

    # Store
    db.write_series(output_key, merged, source="merged",
                    fillna_method="ffill" if align == "outer" else "none",
                    ticker_meta=[
                        {"ticker": None, "column_name": c,
                         "source": "merged",
                         "interval": None,
                         "extra": {"from_keys": list(data_keys), "align": align}}
                        for c in merged.columns
                    ])

    _CACHE[output_key] = {
        "Y":       merged.values.astype(float),
        "columns": list(merged.columns),
        "index":   [str(i)[:10] for i in merged.index],
    }

    date_range: List[str] = []
    if len(merged) > 0:
        date_range = [str(merged.index[0])[:10], str(merged.index[-1])[:10]]

    return {
        "output_key":  output_key,
        "source_keys": list(data_keys),
        "align":       align,
        "n_rows":      int(n_after),
        "n_cols":      int(len(merged.columns)),
        "columns":     list(merged.columns),
        "date_range":  date_range,
        "dropped_rows": dropped,
    }
