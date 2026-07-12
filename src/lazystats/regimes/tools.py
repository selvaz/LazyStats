# -*- coding: utf-8 -*-
"""
lazystats.regimes.tools — LLM Tool API layer for LazyHMM regime detection
===============================================================
Thin tool/integration layer on top of the pure core in ``lazystats.regimes.core``.

This module imports every modelling primitive (data structures, engines,
fitting functions, discrete/multivariate models) from ``lazystats.regimes.core`` and adds
only:

  * an in-process result store (optionally backed by ``lazystats.regimes.db`` SQLite or a
    lazybridge Store) so an LLM never passes large arrays between tool calls;
  * the LLM Tool API (§10): fit_regimes, scan_state_counts, get_current_regime,
    get_regime_summary, compare_emission_models, fit_categorical_regimes,
    fit_regimes_window, compare_regime_windows;
  * a DB-aware RegimeRun variant (DBRegimeRun) that can persist plots to a depot.

The single, logical model schema lives in the core:
  model = "panel" | "joint_diag" | "joint_full" | "categorical"
  (panel = independent univariate HMM per column; joint_* = one joint
   multivariate HMM with diagonal/full covariance; categorical = discrete).

External requirements: numpy, pandas, matplotlib, hmmlearn, scikit-learn
(all pulled in transitively by lazystats.regimes.core).

SECTIONS
  §1  Imports, core re-exports & exports
  §1b In-process store  (backing the LLM Tool API)
  §3  Visualization     DBRegimeRun (DB-aware plots)
  §9  High-level engine RegimeEngine  (DataFrame → DBRegimeRun)
  §10 LLM Tool API      fit_regimes · scan_state_counts · get_current_regime ·
                        get_regime_summary · compare_emission_models ·
                        fit_categorical_regimes · fit_regimes_window ·
                        compare_regime_windows
"""

# ══════════════════════════════════════════════════════════════════════════════
# §1  IMPORTS, CORE RE-EXPORTS & EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Annotated, Any, Dict, List, Literal, Optional,
    Sequence, Tuple,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# ── Core: the single source of truth for all modelling primitives ──────────
from .core import (
    # type aliases / schema
    ModelKind, _resolve_model,
    # data structures
    FitWarning, FitResult, PlotTheme, RegimeRun,
    # gaussian engine + selection + post-processing
    MSGaussianHMM, select_num_regimes, fit_with_auto_S,
    reorder_fitresult, regime_labels_from_S,
    # user-supplied parameters: persistence + fixed inference
    HMMParams, infer_with_params, RegimeEngine as _CoreRegimeEngine, DiscreteFitResult, fit_discrete_hmm, fit_with_auto_S_categorical,
    # multivariate independent-emission core
    MultiVarFitResult, fit_multivar_hmm, fit_with_auto_S_multivar,
    # private helpers reused by the tool layer
    _ensure_2d, _count_free_params, _state_vol_measure,
)

__all__ = [
    # Re-exported core data structures / engines
    "FitWarning", "FitResult", "PlotTheme", "RegimeRun",
    "MSGaussianHMM", "select_num_regimes", "fit_with_auto_S",
    "reorder_fitresult", "regime_labels_from_S", "ModelKind",
    "DiscreteFitResult", "fit_discrete_hmm", "fit_with_auto_S_categorical",
    "MultiVarFitResult", "fit_multivar_hmm", "fit_with_auto_S_multivar",
    # Tool-layer engine + visualization
    "RegimeEngine", "MSRegimeEngine", "DBRegimeRun",
    # Store management
    "connect_lazy_store", "regime_store_load", "regime_store_list",
    "regime_store_delete",
    # Data ingestion
    "load_time_series",
    # Fitting / querying tools
    "fit_regimes", "scan_state_counts",
    "get_current_regime", "get_regime_summary",
    "compare_emission_models",
    "fit_categorical_regimes",
    "fit_regimes_window", "compare_regime_windows",
    # Parameter persistence + fixed-parameter inference
    "HMMParams", "regime_params_save", "regime_params_load", "apply_regime_params",
    "regime_params_list",
    # DB depot (re-exported for convenience)
    "init_regime_db",
]

# ── Touch point 1: optional SQLite depot via lazystats.regimes.db ───────────────────
try:
    from . import db as _rdb
    _REGIME_DB_AVAILABLE = True
    from .db import init_regime_db   # re-export
except ImportError:
    _REGIME_DB_AVAILABLE = False
    def init_regime_db(db_path: str):      # type: ignore[misc]
        raise ImportError(
            "lazystats.regimes.db could not be imported. "
            "Ensure the lazyhmm package is installed correctly."
        )


# ══════════════════════════════════════════════════════════════════════════════
# §1b  IN-PROCESS STORE  (backing the LLM Tool API)
# ══════════════════════════════════════════════════════════════════════════════
# The LLM never passes raw time-series data or large result dicts between tool
# calls — it would overflow context.  Instead:
#
#   1. load_time_series(file_path, ...)  →  stores data,  returns data_key (str)
#   2. fit_regimes(data_key=..., ...)    →  stores result, returns result_key (str)
#   3. get_current_regime(result_key=...)→  reads stored result, returns small dict
#
# The store is module-level by default.  To use a lazybridge Store (persistent
# SQLite, cross-process), call connect_lazy_store(store) once at startup.
# ══════════════════════════════════════════════════════════════════════════════

_MODULE_STORE: Dict[str, Any] = {}   # in-process fallback
_LAZY_STORE   = None                 # optional Store instance (lazybridge 0.9+)


def connect_lazy_store(store) -> None:
    """Attach a lazybridge Store so tool results persist across processes.

    Call once at startup before any tool is invoked:
        from lazybridge import Store
        from lazystats.regimes import connect_lazy_store
        connect_lazy_store(Store("regime_results.db"))
    """
    global _LAZY_STORE
    _LAZY_STORE = store


def _swrite(key: str, value: Any) -> None:
    # Touch point 2: write-through to SQLite depot when available
    _MODULE_STORE[key] = value
    if _REGIME_DB_AVAILABLE and _rdb._DB is not None:
        _rdb.swrite(key, value)
    elif _LAZY_STORE is not None:
        try:
            _LAZY_STORE.write(key, value)
        except Exception:
            pass


def _sread(key: str) -> Any:
    # Touch point 2: in-process cache → SQLite depot → lazybridge Store
    if key in _MODULE_STORE:
        return _MODULE_STORE[key]
    if _REGIME_DB_AVAILABLE and _rdb._DB is not None:
        try:
            val = _rdb.sread(key)
            _MODULE_STORE[key] = val   # populate cache
            return val
        except KeyError:
            pass
    if _LAZY_STORE is not None:
        # Store.read() returns None on miss (lazybridge 0.9+), not KeyError
        val = _LAZY_STORE.read(key)
        if val is not None:
            return val
    raise KeyError(
        f"Key '{key}' not found in store. "
        f"Available keys: {list(_MODULE_STORE)}"
    )


def _slist() -> List[str]:
    # Touch point 2: union all three stores
    keys = set(_MODULE_STORE.keys())
    if _REGIME_DB_AVAILABLE and _rdb._DB is not None:
        try:
            keys.update(_rdb.slist())
        except Exception:
            pass
    if _LAZY_STORE is not None:
        try:
            keys.update(_LAZY_STORE.keys())  # Store.keys() in lazybridge 0.9+
        except Exception:
            pass
    return sorted(keys)


# ── Public store management tools ──────────────────────────────────────────

def regime_store_list() -> dict:
    """List all keys currently in the regime store.

    Returns:
        dict with keys:
            keys (list[str]): all available store keys.
            count (int): number of keys.
    """
    keys = _slist()
    return {"keys": keys, "count": len(keys)}


def regime_store_load(key: Annotated[str, "Store key to load and inspect."]) -> dict:
    """Peek at a stored object: return its type, shape, and top-level keys.

    Use this to inspect what is stored under a given key without loading the full object.

    Returns:
        dict with keys:
            key (str): the key.
            type (str): Python type name of the stored value.
            info (str): human-readable summary (shape for arrays, keys for dicts, etc.).
    """
    val = _sread(key)
    if isinstance(val, np.ndarray):
        info = f"ndarray shape={val.shape} dtype={val.dtype}"
    elif isinstance(val, pd.DataFrame):
        info = f"DataFrame shape={val.shape} columns={list(val.columns)[:10]}"
    elif isinstance(val, dict):
        info = f"dict keys={list(val.keys())}"
    elif isinstance(val, list):
        info = f"list len={len(val)}"
    else:
        info = str(val)[:200]
    return {"key": key, "type": type(val).__name__, "info": info}


def regime_store_delete(key: Annotated[str, "Store key to remove."]) -> dict:
    """Delete a key from the store to free memory.

    Returns:
        dict with keys:
            deleted (str): the key that was removed.
            remaining (list[str]): keys still in store.
    """
    _MODULE_STORE.pop(key, None)
    if _LAZY_STORE is not None:
        try:
            _LAZY_STORE.delete(key)
        except Exception:
            pass
    return {"deleted": key, "remaining": _slist()}


# ══════════════════════════════════════════════════════════════════════════════
# §3  VISUALIZATION  —  DBRegimeRun (DB-aware plots)
# ══════════════════════════════════════════════════════════════════════════════
# DBRegimeRun is the depot-aware variant of the core RegimeRun: same tidy panel
# DataFrame, but its plot methods can persist figures to regime_db when
# save_to_db=True. PlotTheme is imported from the core.

_THEMES: Dict[str, PlotTheme] = {
    "dark": PlotTheme(
        figure_facecolor="#0d1117", axes_facecolor="#161b22",
        savefig_facecolor="#0d1117", axes_edgecolor="#30363d",
        axes_labelcolor="#8b949e", xtick_color="#8b949e", ytick_color="#8b949e",
        text_color="#e6edf3", grid_color="#30363d", grid_alpha=0.4,
        axes_grid_default=True, font_size=11,
        line_color="#58a6ff", line_width=1.0,
        regime_alpha=0.35, legend_patch_alpha=0.7,
        regime_palette=("#3fb950", "#d29922", "#f78166", "#d2a8ff", "#79c0ff"),
        binary_palette=("#30363d", "#f78166"),
    ),
    "light": PlotTheme(
        figure_facecolor="white", axes_facecolor="white",
        savefig_facecolor="white", axes_edgecolor="#cccccc",
        axes_labelcolor="#555555", xtick_color="#333333", ytick_color="#333333",
        text_color="#111111", grid_color="#e0e0e0", grid_alpha=0.6,
        axes_grid_default=True, font_size=11,
        line_color="#1f77b4", line_width=1.0,
        regime_alpha=0.30, legend_patch_alpha=0.7,
        regime_palette=("#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#1f77b4"),
        binary_palette=("#eeeeee", "#d62728"),
    ),
    "minimal": PlotTheme(
        figure_facecolor="white", axes_facecolor="white",
        savefig_facecolor="white", axes_edgecolor="#aaaaaa",
        axes_labelcolor="#555555", xtick_color="#555555", ytick_color="#555555",
        text_color="#111111", grid_color="#dddddd", grid_alpha=0.3,
        axes_grid_default=False, font_size=10,
        line_color="#333333", line_width=0.8,
        regime_alpha=0.20, legend_patch_alpha=0.6,
        regime_palette=("#4e9a4e", "#d08c1a", "#c0392b", "#7d5ba6", "#2471a3"),
        binary_palette=("#e8e8e8", "#c0392b"),
    ),
}


@dataclass
class DBRegimeRun:
    """Output of RegimeEngine.fit(): a tidy panel DataFrame + depot-aware plots."""
    panel: pd.DataFrame
    rows: List[str]
    labels_map: Dict[str, List[str]]
    meta: Dict[str, Any]
    _theme: PlotTheme = field(default_factory=lambda: _THEMES["dark"], repr=False)

    # ── theme control ──────────────────────────────────────────────────────
    def set_theme(self, name: str) -> "DBRegimeRun":
        if name not in _THEMES:
            raise ValueError(f"Unknown theme '{name}'. Available: {list(_THEMES)}")
        object.__setattr__(self, "_theme", _THEMES[name])
        return self

    def available_themes(self) -> List[str]:
        return list(_THEMES)

    # ── internal helpers ───────────────────────────────────────────────────
    def _slice_last(self, last_years: int, points_per_year: int) -> pd.DataFrame:
        n = last_years * points_per_year
        return self.panel.iloc[-n:] if len(self.panel) > n else self.panel

    def _format_time_axis(self, ax, idx, n_xticks: int) -> None:
        step = max(1, len(idx) // n_xticks)
        positions = list(range(0, len(idx), step))
        labels = [str(idx[p])[:10] for p in positions]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    @contextmanager
    def _theme_context(self):
        with plt.rc_context(self._theme.rc_dict()):
            yield

    # ── plot helpers ───────────────────────────────────────────────────────
    def _maybe_save(self, fig, save_to_db: bool, result_key: str,
                    data_key: str, series_name: str,
                    plot_type: str, title: str) -> Optional[str]:
        """Save fig to depot if save_to_db=True. Called BEFORE plt.show()."""
        if not save_to_db:
            return None
        if not _REGIME_DB_AVAILABLE or _rdb._DB is None:
            return None
        try:
            return _rdb.save_figure(fig, result_key, data_key,
                                    series_name, plot_type, title)
        except Exception:
            return None

    # ── plot methods ───────────────────────────────────────────────────────
    def plot_series_with_regimes(
        self,
        name: str,
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        alpha: Optional[float] = None,
        figsize: Tuple[int, int] = (14, 3),
        title: Optional[str] = None,
        cmap_name: Optional[str] = None,
        save_to_db: bool = False,
        result_key: str = "",
        data_key: str = "",
    ) -> Optional[str]:
        """Line chart of one series with coloured background spans per regime.

        Returns plot_key (str) if save_to_db=True and depot is active, else None.
        """
        with self._theme_context():
            df = self._slice_last(last_years, points_per_year)
            s  = df[f"{name}_value"].dropna()
            st = df[f"{name}_state"].reindex(s.index).astype(int)

            S_local = int(st.max() + 1)
            labels  = self.labels_map.get(name, [f"State {i}" for i in range(S_local)])[:S_local]
            cmap    = (cm.get_cmap(cmap_name, S_local) if cmap_name
                       else self._theme.discrete_cmap(S_local))
            a  = float(self._theme.regime_alpha if alpha is None else alpha)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(s.index, s.values,
                    color=self._theme.line_color, linewidth=self._theme.line_width)

            cur, start = int(st.iloc[0]), s.index[0]
            for t in range(1, len(st)):
                now = int(st.iloc[t])
                if now != cur:
                    ax.axvspan(start, st.index[t], color=cmap(cur), alpha=a, lw=0)
                    cur, start = now, st.index[t]
            ax.axvspan(start, s.index[-1], color=cmap(cur), alpha=a, lw=0)

            patches = [mpatches.Patch(color=cmap(i), alpha=self._theme.legend_patch_alpha,
                                      label=labels[i]) for i in range(S_local)]
            ax.legend(handles=patches, loc="upper left", frameon=False)
            ax.set_title(title or name, pad=10)
            ax.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plot_key = self._maybe_save(fig, save_to_db, result_key, data_key,
                                        name, "series_with_regimes", title or name)
            plt.show()
            return plot_key

    def plot_barcode_states(
        self,
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        figsize: Tuple[int, int] = (16, 6),
        n_xticks: int = 10,
        title: str = "Regime States",
        save_to_db: bool = False,
        result_key: str = "",
        data_key: str = "",
    ) -> Optional[str]:
        """Heatmap grid: rows = series, columns = time, colour = state id.

        Returns plot_key (str) if save_to_db=True and depot is active, else None.
        """
        with self._theme_context():
            df  = self._slice_last(last_years, points_per_year)
            mat = np.vstack([df[f"{r}_state"].values for r in self.rows]).astype(float)
            Sg  = int(np.nanmax(mat)) + 1
            cmap   = self._theme.discrete_cmap(Sg)
            bounds = np.arange(-0.5, Sg + 0.5, 1.0)
            norm   = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            ax.set_yticks(range(len(self.rows)))
            ax.set_yticklabels(self.rows)
            self._format_time_axis(ax, df.index, n_xticks)
            ax.set_title(title, pad=12)
            fig.colorbar(im, ax=ax, ticks=np.arange(0, Sg)).set_label("State id")
            plt.tight_layout()
            plot_key = self._maybe_save(fig, save_to_db, result_key, data_key,
                                        "", "barcode_states", title)
            plt.show()
            return plot_key

    def plot_barcode_highvol(
        self,
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        figsize: Tuple[int, int] = (16, 6),
        n_xticks: int = 10,
        title: str = "High-Volatility Regime (binary)",
        save_to_db: bool = False,
        result_key: str = "",
        data_key: str = "",
    ) -> Optional[str]:
        """Binary heatmap: grey = low vol, coloured = high-vol regime.

        Returns plot_key (str) if save_to_db=True and depot is active, else None.
        """
        with self._theme_context():
            df  = self._slice_last(last_years, points_per_year)
            mat = np.vstack([df[f"{r}_highvol"].values for r in self.rows]).astype(float)
            cmap   = self._theme.binary_cmap()
            bounds = np.array([-0.5, 0.5, 1.5])
            norm   = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
            ax.set_yticks(range(len(self.rows)))
            ax.set_yticklabels(self.rows)
            self._format_time_axis(ax, df.index, n_xticks)
            ax.set_title(title, pad=12)
            fig.colorbar(im, ax=ax, ticks=[0, 1]).set_label("High-Vol (0/1)")
            plt.tight_layout()
            plot_key = self._maybe_save(fig, save_to_db, result_key, data_key,
                                        "", "barcode_highvol", title)
            plt.show()
            return plot_key

    def plot_small_multiples(
        self,
        names: Sequence[str],
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        figsize_per_row: Tuple[int, int] = (14, 3),
        cmap_name: Optional[str] = None,
        save_to_db: bool = False,
        result_key: str = "",
        data_key: str = "",
    ) -> List[Optional[str]]:
        """One plot_series_with_regimes per name, stacked vertically.

        Returns list of plot_keys (one per name) when save_to_db=True.
        """
        keys = []
        for nm in names:
            pk = self.plot_series_with_regimes(
                nm, last_years=last_years, points_per_year=points_per_year,
                figsize=figsize_per_row, title=nm, cmap_name=cmap_name,
                save_to_db=save_to_db, result_key=result_key, data_key=data_key,
            )
            keys.append(pk)
        return keys

    def save_all_plots_to_db(
        self,
        result_key: str = "",
        data_key: str = "",
        last_years: int = 20,
        points_per_year: int = 52,
    ) -> List[str]:
        """Render all plot types to the depot without displaying them.

        Saves: one series_with_regimes per series + barcode_states + barcode_highvol.
        Uses Agg backend headlessly — does not call plt.show().

        Returns:
            List of plot_keys stored in the depot.
        """
        if not _REGIME_DB_AVAILABLE or _rdb._DB is None:
            raise RuntimeError(
                "No active depot. Call init_regime_db('project.db') first."
            )
        import matplotlib
        prev_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt  # re-import with Agg backend

        plot_keys: List[str] = []
        try:
            for nm in self.rows:
                pk = self.plot_series_with_regimes(
                    nm, last_years=last_years, points_per_year=points_per_year,
                    save_to_db=True, result_key=result_key, data_key=data_key,
                )
                if pk:
                    plot_keys.append(pk)
                _plt.close("all")

            pk = self.plot_barcode_states(
                last_years=last_years, points_per_year=points_per_year,
                save_to_db=True, result_key=result_key, data_key=data_key,
            )
            if pk:
                plot_keys.append(pk)
            _plt.close("all")

            pk = self.plot_barcode_highvol(
                last_years=last_years, points_per_year=points_per_year,
                save_to_db=True, result_key=result_key, data_key=data_key,
            )
            if pk:
                plot_keys.append(pk)
            _plt.close("all")

        finally:
            matplotlib.use(prev_backend)

        return plot_keys


# ══════════════════════════════════════════════════════════════════════════════
# §9  HIGH-LEVEL ENGINE  —  RegimeEngine
# ══════════════════════════════════════════════════════════════════════════════
# Thin wrapper over the core RegimeEngine that returns a depot-aware DBRegimeRun
# (so .save_all_plots_to_db() is available) instead of the plain core RegimeRun.

class RegimeEngine(_CoreRegimeEngine):
    """DataFrame → DBRegimeRun. Auto-selects S, vol-orders states, builds a tidy panel.

    Identical configuration and dispatch to ``lazystats.regimes.core.RegimeEngine`` — see the
    core for the constructor parameters. The only difference is the return type:
    a :class:`DBRegimeRun` whose plot methods can persist figures to regime_db.
    """

    def fit(
        self,
        df: Any,  # pandas.DataFrame
        *,
        model: ModelKind = "panel",
        dropna: Literal["any", "all"] = "all",
    ) -> DBRegimeRun:
        run = super().fit(df, model=model, dropna=dropna)
        return DBRegimeRun(
            panel=run.panel, rows=run.rows,
            labels_map=run.labels_map, meta=run.meta,
        )


MSRegimeEngine = RegimeEngine


# ══════════════════════════════════════════════════════════════════════════════
# §10  LLM TOOL API
# ══════════════════════════════════════════════════════════════════════════════
# Design rules for this section:
#   • All inputs are simple Python types (list, str, int, float, bool).
#   • All outputs are JSON-serializable dicts.
#   • Each function does exactly ONE thing.
#   • Annotated[type, "description"] used for every parameter — highest priority
#     in LazyBridge's tool-schema builder (signature mode).
#   • Literal["a","b"] used for enum-like str params → schema gets "enum" field.
#   • Google-style docstring Args + Returns blocks as fallback descriptions.
#
# HOW DATA AND RESULTS FLOW (LLM use):
#
#   Step 1 — load data from disk → data_key (short string)
#     load_time_series(file_path="returns.csv", data_key="spy")
#     → {"data_key": "spy", "n_rows": 1040, "columns": ["SPY"], ...}
#
#   Step 2 — fit regimes using data_key → result_key
#     fit_regimes(data_key="spy", result_key="spy_regimes", model="panel")
#     → {"result_key": "spy_regimes", "model": "panel", "series": {...summary...}}
#
#   Step 3 — query using result_key (no large data passed)
#     get_current_regime(result_key="spy_regimes", series_name="SPY")
#     → {"current_label": "High Vol", "prob_high_vol": 0.89, ...}
#
#   The LLM only ever passes SHORT STRINGS between tool calls.
#   Raw data and large result dicts live in the store, never in context.
#
# Registering as LazyBridge tools (lazybridge 0.9+):
#   from lazybridge import Tool, Store
#   from lazystats.regimes import connect_lazy_store, load_time_series, fit_regimes, ...
#
#   connect_lazy_store(Store("regime.db"))   # optional: persist to SQLite
#
#   load_tool    = Tool.wrap(load_time_series)
#   list_tool    = Tool.wrap(regime_store_list)
#   fit_tool     = Tool.wrap(fit_regimes)
#   scan_tool    = Tool.wrap(scan_state_counts)
#   current_tool = Tool.wrap(get_current_regime)
#   summary_tool = Tool.wrap(get_regime_summary)
#   compare_tool = Tool.wrap(compare_emission_models)
#   cat_tool     = Tool.wrap(fit_categorical_regimes)
#   window_tool  = Tool.wrap(fit_regimes_window)
#   cmp_win_tool = Tool.wrap(compare_regime_windows)
# ══════════════════════════════════════════════════════════════════════════════


def load_time_series(
    file_path: Annotated[
        str,
        "Absolute or relative path to the data file. "
        "Supported formats: CSV (.csv), Excel (.xlsx, .xls), Parquet (.parquet). "
        "Example: '/data/returns.csv' or 'C:/data/vix_weekly.xlsx'.",
    ],
    value_columns: Annotated[
        list,
        "List of column names to use as time series values. "
        "These become the series that the HMM is fitted on. "
        "Example: ['SPY', 'TLT'] or ['Close'] or ['returns'].",
    ],
    date_column: Annotated[
        str,
        "Name of the column containing dates or timestamps. "
        "Pass an empty string '' if the index is already a date or if there is no date column.",
    ] = "",
    data_key: Annotated[
        str,
        "Short string key under which the loaded data is stored for later use. "
        "Use this key in fit_regimes(data_key=...) or scan_state_counts(data_key=...). "
        "Example: 'spy_returns' or 'vix_weekly'.",
    ] = "data",
    fillna_method: Annotated[
        Literal["ffill", "drop", "zero"],
        "'ffill': forward-fill missing values. "
        "'drop': drop rows with any NaN. "
        "'zero': fill NaN with 0.",
    ] = "ffill",
) -> dict:
    """Load a time series file from disk and store it for subsequent HMM tool calls.

    The LLM never passes raw data between tools — it loads once, gets a data_key,
    and all fitting tools use that key.  This keeps tool call payloads small.

    Args:
        file_path: Path to CSV, Excel, or Parquet file.
        value_columns: Column names to use as time series values.
        date_column: Date column name, or '' if index is already a date.
        data_key: Key for storing the loaded data. Use in fit_regimes(data_key=...).
        fillna_method: How to handle missing values.

    Returns:
        dict with keys:
            data_key (str): key to pass to fitting tools.
            n_rows (int): number of timesteps loaded.
            n_cols (int): number of series (= len(value_columns)).
            columns (list[str]): the value column names as loaded.
            date_range (list[str]): [first_date, last_date] as ISO strings, or [] if no dates.
            missing_pct (dict): {col: pct_missing} BEFORE fill/drop.
            fillna_method (str): method applied.
    """
    path = str(file_path)
    ext  = path.rsplit(".", 1)[-1].lower()

    if ext == "csv":
        df = pd.read_csv(path)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(path)
    elif ext == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension '.{ext}'. Use csv, xlsx, or parquet.")

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.set_index(date_column).sort_index()

    missing_pct = {
        col: round(float(df[col].isna().mean() * 100), 2)
        for col in value_columns if col in df.columns
    }
    missing_cols = [c for c in value_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in file: {missing_cols}. "
                         f"Available: {list(df.columns)}")

    df = df[value_columns]

    if fillna_method == "ffill":
        df = df.ffill().bfill()
    elif fillna_method == "drop":
        df = df.dropna()
    elif fillna_method == "zero":
        df = df.fillna(0.0)

    Y = df.values.astype(float)

    # Store both the raw array and the column names for downstream tools
    _swrite(data_key, {"Y": Y, "columns": list(df.columns),
                       "index": list(df.index.astype(str))})

    date_range: List[str] = []
    if hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype):
        date_range = [str(df.index[0])[:10], str(df.index[-1])[:10]]
    elif len(df.index) > 0:
        date_range = [str(df.index[0]), str(df.index[-1])]

    return {
        "data_key":     data_key,
        "n_rows":       int(Y.shape[0]),
        "n_cols":       int(Y.shape[1]),
        "columns":      list(df.columns),
        "date_range":   date_range,
        "missing_pct":  missing_pct,
        "fillna_method": fillna_method,
    }


def _resolve_data(
    data: list,
    data_key: str,
    series_names: list,
) -> Tuple[np.ndarray, List[str]]:
    """Internal: resolve data from either raw list or store key."""
    if data_key:
        stored = _sread(data_key)
        Y      = stored["Y"]
        cols   = series_names if series_names else stored["columns"]
        return _ensure_2d(Y), list(cols)
    else:
        if not data:
            raise ValueError("Provide either 'data' (nested list) or 'data_key' (store key).")
        Y    = _ensure_2d(np.array(data, dtype=float))
        cols = list(series_names)
        return Y, cols


def _resolve_result(fit_result: dict, result_key: str) -> dict:
    """Internal: resolve fit_result from either raw dict or store key."""
    if result_key:
        return _sread(result_key)
    if not fit_result:
        raise ValueError("Provide either 'fit_result' (dict) or 'result_key' (store key).")
    return fit_result


def _series_to_dict(
    col: str, res: FitResult, Y_col: np.ndarray, reorder_by: str = "vol",
) -> Dict[str, Any]:
    """Build the per-series output dict shared by fit_regimes and compare_emission_models."""
    S          = int(res.S)
    high_state = S - 1   # vol-ascending order guaranteed
    labels     = regime_labels_from_S(S)
    state      = res.viterbi_path_.astype(int)
    gamma      = np.asarray(res.gamma_, dtype=float)
    vols       = _state_vol_measure(res.covars_, res.cov_type)  # (S,) vol measure per state

    regime_stats = []
    for s in range(S):
        mask = state == s
        regime_stats.append({
            "state":             s,
            "label":             labels[s],
            "mean":              float(np.mean(Y_col[mask])) if mask.any() else 0.0,
            "vol":               float(vols[s]),
            "occupancy_pct":     float(np.round(gamma[:, s].mean() * 100, 1)),
            "expected_duration": float(np.round(1 / (1 - np.clip(res.transmat_[s, s], 0, 0.9999)), 1)),
        })

    return {
        "S":                 S,
        "labels":            labels,
        "states":            state.tolist(),
        "high_vol_flag":     (state == high_state).astype(int).tolist(),
        "prob_high_vol":     gamma[:, high_state].tolist(),
        "state_probs":       gamma.tolist(),
        "regime_stats":      regime_stats,
        "transition_matrix": res.transmat_.tolist(),
        "bic":               float(res.bic),
        "loglik":            float(res.loglik),
        "warnings":          [str(w) for w in res.warnings
                              if w.severity in ("warning", "critical")],
    }


def _params_store_key(result_key: str) -> str:
    """Conventional store key for the parameter record tied to a result_key."""
    return f"{result_key}::params"


def _resolve_index(data_key: str, n: int) -> List[str]:
    """Date/index labels for a fit, length ``n``.

    Pulled from the data payload stored under ``data_key`` (what
    load_from_datahub / load_time_series write) when available; otherwise
    integer positions as strings, so downstream tools always have an index.
    """
    idx: List[str] = []
    if data_key:
        try:
            idx = [str(x) for x in (_sread(data_key).get("index") or [])]
        except Exception:
            idx = []
    if len(idx) != n:
        idx = [str(i) for i in range(n)]
    return idx


def _regime_change_points(states: list, labels: list, index: list):
    """Return ``(changes, last_change_pos)`` from a Viterbi state path.

    ``changes`` is a list of dicts (date + from/to state & label) at each
    transition; ``last_change_pos`` is the timestep of the most recent change
    (0 when the series never changes regime).
    """
    st = [int(s) for s in states]
    changes: List[Dict[str, Any]] = []
    last_pos = 0
    for t in range(1, len(st)):
        if st[t] != st[t - 1]:
            changes.append({
                "date": index[t],
                "from_state": st[t - 1], "from_label": labels[st[t - 1]],
                "to_state": st[t], "to_label": labels[st[t]],
            })
            last_pos = t
    return changes, last_pos


def _build_params_record(
    *, result_key: str, model: str, cov_type: str, shared_mean: bool,
    cols: List[str], T: int, data_key: str, criterion: str, S_max: int,
    n_starts: int, sticky: float, random_state: int, S_min: int = 1,
    params_by_series: Optional[Dict[str, Any]], joint_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble a JSON-serializable parameter record with data provenance.

    The record captures *what data the model was estimated on* (data_key,
    series names, sample length and date range) alongside the parameters, so a
    reloaded model is self-describing and can be safely applied to new data.
    """
    try:
        from . import __version__ as _ver
    except Exception:
        _ver = "unknown"

    date_start = date_end = ""
    if data_key:
        try:
            idx = _sread(data_key).get("index") or []
            if idx:
                date_start, date_end = str(idx[0]), str(idx[-1])
        except Exception:
            pass

    record: Dict[str, Any] = {
        "schema":          "lazyhmm.params/1",
        "result_key":      result_key,
        "model":           model,
        "cov_type":        cov_type,
        "shared_mean":     bool(shared_mean),
        "layout":          "joint" if joint_params is not None else "panel",
        "series":          list(cols),
        "created_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lazyhmm_version": _ver,
        "provenance": {
            "data_key":     data_key or "",
            "n_timesteps":  int(T),
            "n_series":     len(cols),
            "date_start":   date_start,
            "date_end":     date_end,
            "criterion":    criterion,
            "S_min":        int(S_min),
            "S_max":        int(S_max),
            "n_starts":     int(n_starts),
            "sticky":       float(sticky),
            "random_state": int(random_state),
        },
    }
    if joint_params is not None:
        record["joint_params"] = joint_params      # k == len(series)
    else:
        record["params_by_series"] = params_by_series or {}
    return record


def regime_params_save(
    result_key: Annotated[str, "Existing result_key whose model parameters to persist."],
    params_key: Annotated[str, "Store key for the parameter record. Empty = '<result_key>::params'."] = "",
) -> dict:
    """Explicitly (re)persist the parameter record for an existing fit result.

    Normally ``fit_regimes`` auto-saves parameters when a ``result_key`` is
    given; use this only to re-save under a different key or after manual edits.

    Returns:
        dict with keys: params_key (str), model (str), layout (str), series (list).
    """
    rec = _sread(_params_store_key(result_key))
    out_key = params_key or _params_store_key(result_key)
    _swrite(out_key, rec)
    return {"params_key": out_key, "model": rec.get("model"),
            "layout": rec.get("layout"), "series": rec.get("series", [])}


def regime_params_list(
    data_key: Annotated[
        str,
        "Optional filter: only models trained on this data_key. Empty = all.",
    ] = "",
) -> dict:
    """Discover trained models already saved in the store.

    Use this BEFORE fitting to check whether a ready-to-use model already
    exists (avoids refitting). Returns compact provenance metadata only — no
    parameter arrays — so it is cheap to call.

    Returns:
        dict with keys:
            count (int): number of models found.
            data_key_filter (str): the filter applied (empty = none).
            models (list[dict]): each with params_key, result_key, data_key,
                model, layout, series, n_series, n_timesteps, date_start,
                date_end, criterion, created_at.
    """
    models: List[Dict[str, Any]] = []

    # Preferred path: the indexed model_params table (authoritative, filterable).
    if _REGIME_DB_AVAILABLE and _rdb._DB is not None:
        try:
            models = _rdb.list_params(data_key or None)
        except Exception:
            models = []

    # Fallback: scan the key/value store for '*::params' records (covers the
    # in-memory store and a connected lazybridge.Store when no SQLite depot).
    if not models:
        for key in _slist():
            if not key.endswith("::params"):
                continue
            try:
                rec = _sread(key)
            except Exception:
                continue
            if not (isinstance(rec, dict)
                    and str(rec.get("schema", "")).startswith("lazyhmm.params")):
                continue
            prov = rec.get("provenance", {})
            if data_key and prov.get("data_key", "") != data_key:
                continue
            models.append({
                "params_key":  key,
                "result_key":  rec.get("result_key", ""),
                "data_key":    prov.get("data_key", ""),
                "model":       rec.get("model", ""),
                "layout":      rec.get("layout", ""),
                "series":      rec.get("series", []),
                "n_series":    prov.get("n_series", len(rec.get("series", []))),
                "n_timesteps": prov.get("n_timesteps", 0),
                "date_start":  prov.get("date_start", ""),
                "date_end":    prov.get("date_end", ""),
                "criterion":   prov.get("criterion", ""),
                "created_at":  rec.get("created_at", ""),
            })

    return {"count": len(models), "data_key_filter": data_key, "models": models}


def regime_params_load(
    params_key: Annotated[
        str,
        "Parameter store key. Either '<result_key>::params' or the params_key "
        "returned by fit_regimes.",
    ],
) -> dict:
    """Load a stored parameter record (metadata + provenance + parameters).

    The record is small (no T-length arrays): it contains startprob/transmat/
    means/covars plus the data provenance describing where the model came from.

    Returns:
        dict: the full parameter record, including 'provenance', and either
        'params_by_series' (panel) or 'joint_params' (joint).
    """
    # Allow passing a bare result_key for convenience.
    try:
        rec = _sread(params_key)
        if isinstance(rec, dict) and rec.get("schema", "").startswith("lazyhmm.params"):
            return rec
    except KeyError:
        rec = None
    return _sread(_params_store_key(params_key))


def apply_regime_params(
    params_key: Annotated[
        str, "Parameter store key (from fit_regimes / regime_params_load)."],
    data: Annotated[
        list, "New T×k data as nested list. Leave [] if using data_key."] = None,
    data_key: Annotated[
        str, "Key of data loaded with load_time_series(). Preferred over 'data'."] = "",
    series_names: Annotated[
        list, "Column names for 'data'. Omit when data_key carries them."] = None,
    result_key: Annotated[
        str, "Optional store key to save the inference output for downstream tools."] = "",
) -> dict:
    """Apply stored parameters to (new) data via fixed-parameter inference.

    Loads the model parameters saved by ``fit_regimes`` and runs inference with
    them held *fixed* — no refitting. Decodes regimes/posteriors on the supplied
    data. Both panel and joint parameters are matched to series by name (joint
    data columns are reordered to the trained series order before inference).

    Returns:
        dict: same compact shape as fit_regimes (per-series current regime +
        regime_stats), plus 'applied_from' = params_key.
    """
    rec = regime_params_load(params_key)
    Y, cols = _resolve_data(data or [], data_key, series_names or [])
    model = rec.get("model", "")
    layout = rec.get("layout", "panel")

    full_result: Dict[str, Any] = {}
    if layout == "joint":
        p = HMMParams.from_dict(rec["joint_params"])
        if Y.shape[1] != p.n_features:
            raise ValueError(
                f"Joint params expect {p.n_features} features but data has {Y.shape[1]}. "
                f"Trained series order: {rec.get('series')}.")
        # The joint means_/covars_ columns are tied to the TRAINING series
        # order, so align the new data by name before inference — otherwise a
        # same-width but reordered input would map emissions to the wrong
        # series and silently corrupt the decoded regimes.
        trained = list(rec.get("series", []))
        if trained and all(t in cols for t in trained):
            order = [cols.index(t) for t in trained]
            Y_use = Y[:, order]
            out_cols = trained
        elif not trained or list(cols) == trained:
            # No recorded series (legacy) or already aligned: use positional order.
            Y_use = Y
            out_cols = list(cols) if cols else [f"f{i}" for i in range(Y.shape[1])]
        else:
            raise ValueError(
                f"Joint model was trained on series {trained}, but data columns "
                f"are {list(cols)}. Provide the same series (any order) so columns "
                f"can be aligned by name.")
        res = infer_with_params(Y_use, p)
        for j, col in enumerate(out_cols):
            full_result[col] = _series_to_dict(col, res, Y_use[:, j])
    else:
        pbs = rec.get("params_by_series", {})
        for j, col in enumerate(cols):
            if col not in pbs:
                raise ValueError(
                    f"No stored parameters for series '{col}'. "
                    f"Available: {sorted(pbs)}.")
            p = HMMParams.from_dict(pbs[col])
            res = infer_with_params(Y[:, j:j+1], p)
            full_result[col] = _series_to_dict(col, res, Y[:, j])

    full_output = {"model": model, "criterion": rec.get("provenance", {}).get("criterion", ""),
                   "n_timesteps": int(Y.shape[0]), "series": full_result,
                   "applied_from": params_key}
    if result_key:
        _swrite(result_key, full_output)

    compact_series: Dict[str, Any] = {}
    for col, sd in full_result.items():
        compact_series[col] = {k: v for k, v in sd.items()
                               if k not in ("states", "high_vol_flag",
                                            "prob_high_vol", "state_probs")}
        cur = int(sd["states"][-1])
        compact_series[col]["current_state"] = cur
        compact_series[col]["current_label"] = sd["labels"][cur]
        compact_series[col]["prob_high_vol_now"] = round(float(sd["prob_high_vol"][-1]), 4)

    return {
        "result_key":  result_key,
        "applied_from": params_key,
        "model":       model,
        "n_timesteps": int(Y.shape[0]),
        "series":      compact_series,
    }


def fit_regimes(
    data: Annotated[
        list,
        "Time series data as a nested list of shape T×k. "
        "Leave as empty list [] if using data_key instead.",
    ] = None,
    series_names: Annotated[
        list,
        "Names for each of the k series columns. "
        "Leave as empty list [] if data_key is used and column names were saved with load_time_series.",
    ] = None,
    data_key: Annotated[
        str,
        "Key of data previously loaded with load_time_series(). "
        "When provided, 'data' and 'series_names' can be omitted. "
        "Preferred path for LLM use — avoids passing large arrays in tool calls.",
    ] = "",
    result_key: Annotated[
        str,
        "Key under which to store the full fit result for downstream tools. "
        "Pass this key to get_current_regime(), get_regime_summary(), etc. "
        "Example: 'spy_regimes'. Leave empty to skip storing.",
    ] = "",
    model: Annotated[
        Literal["panel", "joint_diag", "joint_full"],
        "Modelling mode (single axis). "
        "'panel': one independent univariate HMM per series — each series can be in a "
        "different regime at the same timestep (diagonal covariance, k=1). "
        "'joint_diag': one joint multivariate HMM with diagonal covariance — all series "
        "share the same latent regime, features treated independently within a state. "
        "'joint_full': one joint multivariate HMM with full covariance — shared regime "
        "plus cross-series correlations (use with small k, e.g. 2–3).",
    ] = "panel",
    S_max: Annotated[
        int,
        "Maximum number of regimes to consider. The optimal count is selected "
        "automatically using the chosen criterion. Typical range: 2–5.",
    ] = 4,
    criterion: Annotated[
        Literal["bic", "aic", "hqic"],
        "Model selection criterion. 'bic' is most conservative (prefers fewer regimes). "
        "'aic' is least conservative (allows more regimes). 'hqic' is intermediate.",
    ] = "bic",
    n_starts: Annotated[
        int,
        "Number of independent EM restarts per candidate state count. "
        "More restarts improve solution quality at the cost of speed. "
        "Recommended: 20 for exploration, 50 for production.",
    ] = 30,
    shared_mean: Annotated[
        bool,
        "If True, expected return is the same across all regimes — only volatility switches "
        "(variance-regime model). If False, both mean and volatility vary per regime.",
    ] = False,
    sticky: Annotated[
        float,
        "Regime persistence initialization prior between 0.80 and 0.99. "
        "Higher values assume longer-lasting regimes in the initialization. "
        "0.95 gives expected initial duration of 20 timesteps.",
    ] = 0.95,
    random_state: Annotated[int, "Random seed for reproducibility."] = 42,
    S_min: Annotated[
        int,
        "Minimum number of regimes to scan (default 1). Use S_min=2 to force "
        "at least two regimes. Must satisfy 1 <= S_min <= S_max.",
    ] = 1,
) -> dict:
    """Fit Gaussian HMM regime detection on financial time series.

    Automatically selects the optimal number of regimes using BIC/AIC.
    Regimes are always ordered by volatility: state 0 = calmest, state S-1 = most volatile.

    LLM WORKFLOW (preferred):
        1. load_time_series(file_path="returns.csv", data_key="spy")
        2. fit_regimes(data_key="spy", result_key="spy_regimes", model="panel")
        3. get_current_regime(result_key="spy_regimes", series_name="SPY")

    REUSE BEFORE REFITTING:
        Fitting is expensive. Before calling this, check whether a trained model
        already exists for the data with regime_params_list(data_key="spy"). If a
        suitable model is listed, apply it to (new) data with apply_regime_params()
        instead of refitting — same regime output, no EM. When a result_key is
        given, this function auto-persists the parameters (with data provenance)
        under '<result_key>::params', so future sessions can discover and reuse it.

    Args:
        data: T×k nested list. Leave empty if data_key is provided.
        series_names: Column names. Leave empty if data_key was used with load_time_series.
        data_key: Store key from load_time_series(). Preferred for LLM use.
        result_key: Store key for the result (pass to downstream tools).
        model: 'panel' = independent HMM per series; 'joint_diag'/'joint_full' = one joint HMM.
        S_max: Maximum regimes to try (BIC selects the optimal count).
        S_min: Minimum number of regimes to scan (default 1).
        criterion: 'bic', 'aic', or 'hqic'.
        n_starts: EM random restarts per candidate S.
        shared_mean: True = variance-only regime model.
        sticky: Regime persistence prior (0.80–0.99).
        random_state: Seed.

    Returns:
        dict with keys:
            result_key (str): store key to use in downstream tools (empty if not stored).
            params_key (str): store key of the saved parameters for reuse via
                apply_regime_params()/regime_params_load() (empty if not stored).
            model, criterion, n_timesteps (str/int): fit metadata.
            series (dict): per series — regime_stats, bic, loglik, current regime summary.
                NOTE: state_probs (T×S matrix) is stored in the result_key store,
                not returned here, to keep the response small.
    """
    Y, cols = _resolve_data(data or [], data_key, series_names or [])
    T, k    = Y.shape

    fit_mode, cov_type = _resolve_model(model)   # 'panel'/'joint' + diag/full
    rng       = np.random.RandomState(random_state)

    common = dict(S_max=S_max, S_min=S_min, criterion=criterion, n_starts=n_starts,
                  cov_type=cov_type, shared_mean=shared_mean, sticky=sticky)
    full_result: Dict[str, Any] = {}
    params_by_series: Dict[str, Any] = {}   # panel: one HMMParams per series
    joint_params: Optional[Dict[str, Any]] = None  # joint: single HMMParams

    if fit_mode == "panel":
        for j, col in enumerate(cols):
            # panel: independent univariate HMM per column → diagonal covariance
            out = fit_with_auto_S(Y[:, j:j+1], random_state=int(rng.randint(0, 2**31-1)),
                                  **{**common, "cov_type": "diag"})
            res = reorder_fitresult(out["final_result"], by="vol", ascending=True)
            full_result[col] = _series_to_dict(col, res, Y[:, j])
            params_by_series[col] = HMMParams.from_fitresult(res).to_dict()
    else:
        out = fit_with_auto_S(Y, random_state=int(rng.randint(0, 2**31-1)), **common)
        res = reorder_fitresult(out["final_result"], by="vol", ascending=True)
        for j, col in enumerate(cols):
            full_result[col] = _series_to_dict(col, res, Y[:, j])
        joint_params = HMMParams.from_fitresult(res).to_dict()

    full_output = {"model": model, "criterion": criterion, "n_timesteps": T,
                   "series": full_result,
                   # date/index labels (from the data payload) so downstream
                   # tools can map Viterbi state changes back to dates.
                   "index": _resolve_index(data_key, T),
                   # provenance for generate_regime_plots: which stored data
                   # payload this fit was computed on ("" for inline data)
                   "data_key": data_key}

    # Store full result (with T×S state_probs) for downstream tools
    params_key = ""
    if result_key:
        _swrite(result_key, full_output)
        # Auto-persist the model parameters with data provenance so the model
        # can be reloaded for inference on new data (infer_with_params), not
        # just re-read for its decoded states.
        params_key = _params_store_key(result_key)
        record = _build_params_record(
            result_key=result_key, model=model, cov_type=cov_type,
            shared_mean=shared_mean, cols=cols, T=T, data_key=data_key,
            criterion=criterion, S_max=S_max, S_min=S_min, n_starts=n_starts, sticky=sticky,
            random_state=random_state,
            params_by_series=params_by_series or None, joint_params=joint_params,
        )
        _swrite(params_key, record)
        full_output["params_key"] = params_key

    # Return a compact version: drop state_probs from the inline response
    # to keep tool output small (the LLM reads regime_stats, not raw posteriors)
    compact_series: Dict[str, Any] = {}
    for col, sd in full_result.items():
        compact_series[col] = {k: v for k, v in sd.items()
                               if k not in ("states", "high_vol_flag",
                                            "prob_high_vol", "state_probs")}
        # Add current regime summary for immediate LLM use
        cur = int(sd["states"][-1])
        compact_series[col]["current_state"]  = cur
        compact_series[col]["current_label"]  = sd["labels"][cur]
        compact_series[col]["prob_high_vol_now"] = round(float(sd["prob_high_vol"][-1]), 4)

    return {
        "result_key":  result_key,
        "params_key":  params_key,
        "model":       model,
        "criterion":   criterion,
        "n_timesteps": T,
        "series":      compact_series,
    }


def scan_state_counts(
    data: Annotated[
        list,
        "Time series data as a T×k nested list. Same format as fit_regimes.",
    ],
    series_names: Annotated[
        list,
        "Names for each series column. Length must match width of data.",
    ],
    model: Annotated[
        Literal["panel", "joint_diag", "joint_full"],
        "'panel': scan independently per series. "
        "'joint_diag'/'joint_full': scan once on all columns jointly (diag/full covariance).",
    ] = "panel",
    S_max: Annotated[
        int,
        "Maximum number of regimes to scan. Scores are returned for S = S_min up to S_max.",
    ] = 6,
    criterion: Annotated[
        Literal["bic", "aic", "hqic"],
        "Criterion to report as the primary score. 'bic' is most conservative.",
    ] = "bic",
    n_starts: Annotated[
        int,
        "EM restarts per candidate S. Fewer restarts are acceptable here (20 is enough for scanning).",
    ] = 20,
    S_min: Annotated[
        int,
        "Minimum number of regimes to scan (default 1). Must satisfy 1 <= S_min <= S_max.",
    ] = 1,
) -> dict:
    """Scan candidate state counts and return selection scores without fitting a final model.

    Use this before fit_regimes() to audit which S values are viable, check whether
    any are rejected as ghost/flickering states, and identify the BIC elbow.

    Args:
        data: T×k nested list.
        series_names: Column names, length k.
        model: 'panel' scans each series independently; 'joint_diag'/'joint_full' scan jointly.
        S_max: Upper bound for S.
        S_min: Minimum number of regimes to scan (default 1).
        criterion: Score to return ('bic', 'aic', 'hqic').
        n_starts: EM restarts per S.

    Returns:
        dict with key 'series' mapping each series name to:
            best_S (int): recommended number of regimes.
            best_score (float): criterion value at best_S.
            scores (list[dict]): per S from S_min to S_max:
                S (int), score (float), rejected (bool), rejection_reasons (list[str]).
    """
    Y    = _ensure_2d(np.array(data, dtype=float))
    T, k = Y.shape
    fit_mode, cov_type = _resolve_model(model)
    common   = dict(S_max=S_max, S_min=S_min, criterion=criterion, n_starts=n_starts,
                    cov_type=cov_type)
    out: Dict[str, Any] = {}

    if fit_mode == "panel":
        for j, col in enumerate(series_names):
            sel = select_num_regimes(Y[:, j:j+1], **{**common, "cov_type": "diag"})
            out[col] = _format_selection(sel, S_max, S_min)
    else:
        sel = select_num_regimes(Y, **common)
        for col in series_names:
            out[col] = _format_selection(sel, S_max, S_min)

    return {"model": model, "criterion": criterion, "series": out}


def _format_selection(sel: Dict[str, Any], S_max: int, S_min: int = 1) -> Dict[str, Any]:
    scores = []
    for s in range(S_min, S_max + 1):
        info = sel["per_S"].get(s, {})
        reasons = []
        for r in info.get("reasons", []):
            reasons.append(str(r))
        scores.append({
            "S":                 s,
            "score":             float(info.get("score", np.inf)),
            "rejected":          bool(info.get("rejected", True)),
            "rejection_reasons": reasons,
        })
    return {
        "best_S":      int(sel["best_S"]),
        "best_score":  float(sel["best_score"]),
        "scores":      scores,
    }


def get_current_regime(
    result_key: Annotated[
        str,
        "Store key returned by fit_regimes(result_key=...). Preferred for LLM use. "
        "Leave empty if passing fit_result directly.",
    ] = "",
    series_name: Annotated[
        str,
        "Name of the series to query. Must be one of the series fitted.",
    ] = "",
    fit_result: Annotated[
        dict,
        "Output dict from fit_regimes(). Only needed if result_key is not used. "
        "For LLM use, prefer result_key.",
    ] = None,
) -> dict:
    """Return the current (most recent timestep) regime for a single series.

    LLM WORKFLOW:
        fit_regimes(data_key="spy", result_key="spy_regimes")
        get_current_regime(result_key="spy_regimes", series_name="SPY")

    Args:
        result_key: Store key from fit_regimes(result_key=...). Use this in LLM pipelines.
        series_name: Name of the series to query.
        fit_result: Raw dict output of fit_regimes(). Only if not using result_key.

    Returns:
        dict with keys:
            series (str): series name.
            current_state (int): state index at last timestep (0 = lowest vol).
            current_label (str): human-readable label e.g. 'High Vol'.
            prob_current_state (float): posterior probability of the current state.
            prob_high_vol (float): posterior probability of the highest-vol regime.
            is_high_vol (bool): True if currently in the highest-vol regime.
            current_mean (float): mean return of current regime.
            current_vol (float): annualised volatility of current regime.
            expected_duration (float): expected steps remaining in current regime.
            transition_to_high_vol (float): one-step probability of entering high-vol regime.
    """
    resolved = _resolve_result(fit_result or {}, result_key)

    if "series" not in resolved:
        raise ValueError("fit_result must be the output of fit_regimes().")
    if series_name not in resolved["series"]:
        raise KeyError(f"Series '{series_name}' not found. "
                       f"Available: {list(resolved['series'])}")

    s_data      = resolved["series"][series_name]
    S           = int(s_data["S"])
    high_state  = S - 1
    cur_state   = int(s_data["states"][-1])
    cur_label   = s_data["labels"][cur_state]
    prob_cur    = float(s_data["state_probs"][-1][cur_state])
    prob_hv     = float(s_data["prob_high_vol"][-1])
    stats       = s_data["regime_stats"][cur_state]
    A           = s_data["transition_matrix"]
    trans_to_hv = float(A[cur_state][high_state])

    # When did the current regime start? Derive from the Viterbi path + dates.
    states  = s_data["states"]
    index   = resolved.get("index") or [str(i) for i in range(len(states))]
    changes, last_pos = _regime_change_points(states, s_data["labels"], index)

    return {
        "series":                 series_name,
        "current_state":          cur_state,
        "current_label":          cur_label,
        "prob_current_state":     round(prob_cur,    4),
        "prob_high_vol":          round(prob_hv,     4),
        "is_high_vol":            cur_state == high_state,
        "current_mean":           round(stats["mean"], 6),
        "current_vol":            round(stats["vol"],  6),
        "expected_duration":      stats["expected_duration"],
        "transition_to_high_vol": round(trans_to_hv, 4),
        # how long we've actually been in this regime (vs expected_duration)
        "last_change_date":       index[last_pos] if changes else index[0],
        "steps_in_current_regime": len(states) - last_pos,
        "n_changes":              len(changes),
    }


def get_regime_changes(
    result_key: Annotated[
        str,
        "Store key returned by fit_regimes(result_key=...). Preferred for LLM use.",
    ] = "",
    series_name: Annotated[
        str,
        "Name of the series to query. Must be one of the series fitted.",
    ] = "",
    last_n: Annotated[
        int,
        "If > 0, return only the most recent N changes (the summary fields are "
        "always over the full history).",
    ] = 0,
    fit_result: Annotated[
        dict,
        "Raw fit_regimes() output. Only needed if result_key is not used.",
    ] = None,
) -> dict:
    """Dates of regime changes for one series, and how long the current regime
    has lasted.

    Built from the Viterbi state path and the fit's stored date index, so it
    reports *when* each regime switch happened (not just the expected duration
    that ``get_current_regime`` gives). Dates are the resampled timestamps of
    the fit (e.g. weekly); if the fit had no dates (inline ``data=``) they are
    integer positions as strings.

    LLM WORKFLOW:
        fit_regimes(data_key="spy", result_key="spy_regimes")
        get_regime_changes(result_key="spy_regimes", series_name="SPY")

    Returns:
        dict with keys:
            series (str), current_state (int), current_label (str),
            last_change_date (str): date of the most recent regime switch
                (the fit's first date if the series never switches),
            steps_in_current_regime (int): observations since that switch
                (in the fit's frequency, e.g. weeks),
            n_changes (int): total number of switches over the history,
            changes (list): one dict per switch with
                {date, from_state, from_label, to_state, to_label}.
    """
    resolved = _resolve_result(fit_result or {}, result_key)
    if "series" not in resolved:
        raise ValueError("fit_result must be the output of fit_regimes().")
    if series_name not in resolved["series"]:
        raise KeyError(f"Series '{series_name}' not found. "
                       f"Available: {list(resolved['series'])}")

    s_data = resolved["series"][series_name]
    states = s_data["states"]
    index  = resolved.get("index") or [str(i) for i in range(len(states))]
    changes, last_pos = _regime_change_points(states, s_data["labels"], index)
    out_changes = changes[-last_n:] if last_n and last_n > 0 else changes
    cur = int(states[-1])
    return {
        "series":                  series_name,
        "current_state":           cur,
        "current_label":           s_data["labels"][cur],
        "last_change_date":        index[last_pos] if changes else index[0],
        "steps_in_current_regime": len(states) - last_pos,
        "n_changes":               len(changes),
        "changes":                 out_changes,
    }


def get_regime_summary(
    result_key: Annotated[
        str,
        "Store key returned by fit_regimes(result_key=...). Preferred for LLM use.",
    ] = "",
    fit_result: Annotated[
        dict,
        "Output dict from fit_regimes(). Only if not using result_key.",
    ] = None,
) -> str:
    """Return a human-readable text summary of all fitted regimes.

    Describes each series: number of regimes, per-regime characteristics
    (mean, volatility, occupancy, expected duration), and current regime.
    Designed for direct LLM consumption — returns plain text, not structured data.

    LLM WORKFLOW:
        fit_regimes(data_key="spy", result_key="spy_regimes")
        get_regime_summary(result_key="spy_regimes")

    Args:
        result_key: Store key from fit_regimes(result_key=...).
        fit_result: Raw output dict. Only if not using result_key.

    Returns:
        Multi-line plain-text summary of all regimes across all series.
    """
    resolved = _resolve_result(fit_result or {}, result_key)
    if "series" not in resolved:
        raise ValueError("fit_result must be the output of fit_regimes().")

    lines = [
        "Regime Detection Summary",
        f"  Model     : {resolved.get('model', 'unknown')}",
        f"  Criterion : {resolved.get('criterion', 'unknown')}",
        f"  Timesteps : {resolved.get('n_timesteps', '?')}",
        "",
    ]

    for name, s_data in resolved["series"].items():
        S          = int(s_data["S"])
        cur_state  = int(s_data.get("current_state", s_data["states"][-1]
                                    if "states" in s_data else 0))
        cur_label  = s_data["labels"][cur_state]
        prob_hv    = round(float(
            s_data.get("prob_high_vol_now",
                       s_data["prob_high_vol"][-1] if "prob_high_vol" in s_data else 0)
        ) * 100, 1)
        warns      = s_data.get("warnings", [])

        lines.append(f"── {name}  (S={S}, BIC={s_data['bic']:.1f}) ──")
        lines.append(f"  Current regime : {cur_label}  (P(high-vol) = {prob_hv}%)")
        lines.append("  Regimes:")

        for st in s_data["regime_stats"]:
            lines.append(
                f"    [{st['state']}] {st['label']:<18s} | "
                f"mean={st['mean']:+.4f}  vol={st['vol']:.4f}  "
                f"occ={st['occupancy_pct']:.1f}%  "
                f"dur={st['expected_duration']:.1f} steps"
            )

        if warns:
            lines.append(f"  Warnings: {'; '.join(warns)}")
        lines.append("")

    return "\n".join(lines)


def compare_emission_models(
    data: Annotated[
        list,
        "Time series data as a T×k nested list. Same format as fit_regimes.",
    ],
    series_names: Annotated[
        list,
        "Names for each series column, length k.",
    ],
    S: Annotated[
        int,
        "Fixed number of regimes to use in all three models. "
        "Choose based on scan_state_counts() results.",
    ],
    model: Annotated[
        Literal["panel", "joint_diag", "joint_full"],
        "Fitting mode (panel vs joint). The covariance variants below are swept "
        "internally, so 'joint_diag' and 'joint_full' behave the same here — only "
        "panel vs joint matters for this comparison.",
    ] = "panel",
    n_starts: Annotated[
        int,
        "EM restarts per model variant. More = more reliable comparison.",
    ] = 30,
    random_state: Annotated[int, "Random seed."] = 42,
) -> dict:
    """Fit and compare three emission model variants at a fixed number of regimes S.

    The three variants are:
      'full'      — different mean AND full covariance per regime.
      'diag'      — different mean AND diagonal variance per regime (features independent).
      'diag_shared_mean' — shared mean, only diagonal variance switches (variance-regime model).

    Use this to choose the right emission model before calling fit_regimes().

    Args:
        data: T×k nested list.
        series_names: Column names, length k.
        S: Fixed number of regimes (same for all three models).
        model: 'panel' fits per series; 'joint_diag'/'joint_full' fit jointly.
        n_starts: EM restarts.
        random_state: Seed.

    Returns:
        dict with key 'series' mapping each series name to a list of three dicts, each with:
            model (str): 'full' | 'diag' | 'diag_shared_mean'.
            bic (float): BIC score — lower is better.
            loglik (float): log-likelihood.
            n_params (int): free parameter count.
            regime_stats (list[dict]): per-regime mean, vol, occupancy_pct, expected_duration.
            warnings (list[str]): quality warnings.
    """
    Y    = _ensure_2d(np.array(data, dtype=float))
    T, k = Y.shape
    fit_mode, _ = _resolve_model(model)
    is_joint = fit_mode == "joint"
    rng  = np.random.RandomState(random_state)

    variants = [
        ("full",             dict(cov_type="full",  shared_mean=False)),
        ("diag",             dict(cov_type="diag",  shared_mean=False)),
        ("diag_shared_mean", dict(cov_type="diag",  shared_mean=True)),
    ]

    out: Dict[str, Any] = {}

    for col_idx, col in enumerate(series_names):
        out[col] = []
        for v_name, v_kwargs in variants:
            seed = int(rng.randint(0, 2**31 - 1))
            Y_fit = Y if is_joint else Y[:, col_idx:col_idx+1]

            ms  = MSGaussianHMM(S=S, n_starts=n_starts, random_state=seed, **v_kwargs)
            ms.fit(Y_fit)
            res = reorder_fitresult(ms.best_result_, by="vol", ascending=True)

            n_params = _count_free_params(S, k if is_joint else 1,
                                          v_kwargs["cov_type"], v_kwargs["shared_mean"])
            s_data   = _series_to_dict(col, res, Y[:, col_idx])
            out[col].append({
                "model":        v_name,
                "bic":          float(res.bic),
                "loglik":       float(res.loglik),
                "n_params":     n_params,
                "regime_stats": s_data["regime_stats"],
                "warnings":     s_data["warnings"],
            })

        # Sort by BIC ascending
        out[col].sort(key=lambda x: x["bic"])

    return {"S": S, "model": model, "series": out}


def fit_categorical_regimes(
    observations: Annotated[
        list,
        "Categorical observations as a flat list of integers (e.g. [0, 1, 2, 1, 0, ...]). "
        "Each integer is a discrete symbol: quantile bucket, sentiment label, volatility tier, etc. "
        "Values must be 0-indexed integers starting from 0. "
        "For multi-feature inputs pass a nested list of shape T×n_features — symbols are "
        "Cartesian-product encoded automatically.",
    ],
    S_max: Annotated[
        int,
        "Maximum number of latent regimes to consider. "
        "The optimal count is selected automatically via BIC. Typical range: 2–5.",
    ] = 5,
    n_starts: Annotated[
        int,
        "Number of independent EM restarts per candidate state count. "
        "More restarts reduce the chance of local-optima. Recommended: 20–50.",
    ] = 20,
    n_iter: Annotated[
        int,
        "Maximum Baum-Welch (EM) iterations per restart.",
    ] = 300,
    random_state: Annotated[int, "Random seed for reproducibility."] = 42,
    S_min: Annotated[
        int,
        "Minimum number of regimes to scan (default 1). Must satisfy 1 <= S_min <= S_max.",
    ] = 1,
) -> dict:
    """Fit a discrete-emission HMM on categorical (integer) observations.

    Use this when your signal is already in discrete form: quantile buckets
    (e.g. 0=bottom-quartile, 3=top-quartile), sentiment scores, volatility
    tiers, or other enumerated categories.  The emission model is a probability
    table B[state, symbol] learned by Baum-Welch EM.  The number of regimes is
    selected automatically via BIC.

    Args:
        observations: List of integers (0-indexed).  Multi-feature inputs as T×F nested list.
        S_max: Maximum regimes to try.
        S_min: Minimum number of regimes to scan (default 1).
        n_starts: EM restarts per candidate S.
        n_iter: Max EM iterations per restart.
        random_state: Seed.

    Returns:
        dict with keys:
            S (int): number of regimes selected.
            M (int): number of distinct symbols in the data.
            criterion (str): always 'bic' for the categorical model.
            best_score (float): BIC value at best_S.
            states (list[int]): Viterbi state per timestep.
            state_probs (list[list[float]]): posterior matrix T×S.
            emission_probs (list[list[float]]): B matrix S×M — B[s][m] = P(symbol m | state s).
            regime_occupancy (list[dict]): per state: state, occupancy_pct, most_likely_symbol.
            transition_matrix (list[list[float]]): S×S row-stochastic A matrix.
            bic (float): BIC of selected model.
            loglik (float): log-likelihood.
            converged (bool): True if EM converged within n_iter.
            scores (list[dict]): per S from S_min to S_max: S, score, selected (bool).
    """
    obs_arr = np.array(observations)
    if obs_arr.ndim == 1:
        obs_arr = obs_arr.reshape(-1, 1)

    out = fit_with_auto_S_categorical(
        obs_arr, S_max=S_max, S_min=S_min, criterion="bic", n_starts=n_starts,
        n_iter=n_iter, random_state=random_state,
    )
    res: DiscreteFitResult = out["final_result"]
    S, M = int(res.S), int(res.K)

    regime_occupancy = []
    for s in range(S):
        occ  = float(np.round(res.gamma_[:, s].mean() * 100, 1))
        most = int(res.emissionprob_[s].argmax())
        regime_occupancy.append({
            "state":               s,
            "occupancy_pct":       occ,
            "most_likely_symbol":  most,
        })

    # core returns all_results: list of {S, best_bic, best_loglik, best_seed, result}
    by_S = {int(r["S"]): r for r in out["all_results"]}
    best_S = int(out["best_S"])
    best_score = float(by_S[best_S]["best_bic"]) if best_S in by_S else float(res.bic)

    scores = []
    for s in range(S_min, S_max + 1):
        info = by_S.get(s, {})
        scores.append({
            "S":        s,
            "score":    float(info.get("best_bic", np.inf)),
            "selected": s == best_S,
        })

    return {
        "S":                 S,
        "M":                 M,
        "criterion":         "bic",
        "best_score":        best_score,
        "states":            res.viterbi_path_.astype(int).tolist(),
        "state_probs":       res.gamma_.tolist(),
        "emission_probs":    res.emissionprob_.tolist(),
        "regime_occupancy":  regime_occupancy,
        "transition_matrix": res.transmat_.tolist(),
        "bic":               float(res.bic),
        "loglik":            float(res.loglik),
        "converged":         bool(res.converged),
        "scores":            scores,
    }


def fit_regimes_window(
    data: Annotated[
        list,
        "Time series data as a T×k nested list. Full dataset from which a window is sliced.",
    ],
    series_names: Annotated[
        list,
        "Names for each of the k series columns.",
    ],
    window_start: Annotated[
        int,
        "Index of the first observation to include (0-based, inclusive). "
        "Use negative values to count from the end: -260 means 'start 260 steps before the end'.",
    ] = 0,
    window_end: Annotated[
        int,
        "Index of the last observation to include (0-based, exclusive, like Python slicing). "
        "Use -1 or omit (pass 0 to mean 'end of data') to include all data up to the end. "
        "Pass a positive int like 520 to end at row 520.",
    ] = 0,
    model: Annotated[
        Literal["panel", "joint_diag", "joint_full"],
        "Same as fit_regimes: 'panel' = independent HMM per series, "
        "'joint_diag'/'joint_full' = one joint HMM (diag/full covariance).",
    ] = "panel",
    S_max: Annotated[int, "Maximum number of regimes to consider."] = 4,
    criterion: Annotated[Literal["bic", "aic", "hqic"], "Model selection criterion."] = "bic",
    n_starts: Annotated[int, "EM random restarts per candidate state count."] = 30,
    shared_mean: Annotated[bool, "True = variance-regime model (shared mean across states)."] = False,
    sticky: Annotated[float, "Regime persistence initialization prior (0.80–0.99)."] = 0.95,
    random_state: Annotated[int, "Random seed."] = 42,
    S_min: Annotated[int, "Minimum number of regimes to scan (default 1). 1 <= S_min <= S_max."] = 1,
) -> dict:
    """Fit regime detection on a specific observation window (date slice).

    Identical to fit_regimes() but operates on a contiguous slice of the data
    defined by [window_start : window_end].  The returned dict is the same
    structure as fit_regimes() with two additional top-level keys:
    'window_start' and 'window_end' (the resolved integer indices used).

    Use this to:
      - Fit a model on a historical sub-period (e.g. post-2020 only).
      - Compare parameters estimated at different points in time.
      - Chain with compare_regime_windows() for a full temporal comparison.

    Args:
        data: Full T×k nested list.
        series_names: Column names.
        window_start: First row index (inclusive). Negative = from end.
        window_end: Last row index (exclusive). 0 means end of data.
        model: 'panel', 'joint_diag', or 'joint_full'.
        S_max: Maximum regimes.
        S_min: Minimum number of regimes to scan (default 1).
        criterion: 'bic', 'aic', or 'hqic'.
        n_starts: EM restarts.
        shared_mean: Variance-only switching.
        sticky: Persistence prior.
        random_state: Seed.

    Returns:
        Same as fit_regimes() plus:
            window_start (int): resolved start index.
            window_end (int): resolved end index.
            window_length (int): number of timesteps in the window.
    """
    Y_full = _ensure_2d(np.array(data, dtype=float))
    T_full = len(Y_full)

    # Resolve negative indices
    start = int(window_start) if window_start >= 0 else T_full + int(window_start)
    end   = int(window_end)   if window_end   >  0 else T_full
    start = max(0, start)
    end   = min(T_full, end)

    Y_slice = Y_full[start:end]
    result  = fit_regimes(
        data=Y_slice.tolist(),
        series_names=series_names,
        model=model, S_max=S_max, S_min=S_min, criterion=criterion, n_starts=n_starts,
        shared_mean=shared_mean, sticky=sticky,
        random_state=random_state,
    )
    result["window_start"]  = start
    result["window_end"]    = end
    result["window_length"] = end - start
    return result


def compare_regime_windows(
    data: Annotated[
        list,
        "Time series data as a T×k nested list. Same format as fit_regimes.",
    ],
    series_names: Annotated[
        list,
        "Names for each of the k series columns.",
    ],
    windows: Annotated[
        list,
        "List of window definitions. Each entry is a dict with keys: "
        "'label' (str, human-readable name e.g. 'Pre-COVID'), "
        "'start' (int, 0-based inclusive row index, negative = from end), "
        "'end' (int, 0-based exclusive row index, 0 = end of data). "
        "Example: [{'label': 'Full', 'start': 0, 'end': 0}, "
        "{'label': 'Post-2020', 'start': -260, 'end': 0}].",
    ],
    model: Annotated[
        Literal["panel", "joint_diag", "joint_full"],
        "Fitting mode, same as fit_regimes.",
    ] = "panel",
    S_max: Annotated[int, "Maximum number of regimes per window."] = 4,
    criterion: Annotated[Literal["bic", "aic", "hqic"], "Model selection criterion."] = "bic",
    n_starts: Annotated[int, "EM restarts per window and candidate S."] = 20,
    shared_mean: Annotated[bool, "Variance-only switching."] = False,
    sticky: Annotated[float, "Persistence prior."] = 0.95,
    random_state: Annotated[int, "Base random seed. Each window uses a derived seed."] = 42,
    S_min: Annotated[int, "Minimum number of regimes to scan (default 1). 1 <= S_min <= S_max."] = 1,
) -> dict:
    """Fit regime models on multiple observation windows and compare parameter estimates.

    For each window in 'windows', fits an independent HMM and collects:
    - Number of regimes selected per series
    - Per-regime means, volatilities, occupancies, expected durations
    - Transition matrix persistence (diagonal A_{ss})
    - BIC and log-likelihood

    Returns a structured comparison dict that an LLM can reason over to detect
    parameter drift, regime count changes, and volatility-regime transitions
    across different historical sub-periods.

    Args:
        data: Full T×k nested list.
        series_names: Column names.
        windows: List of {label, start, end} dicts.
        model: 'panel', 'joint_diag', or 'joint_full'.
        S_max: Maximum regimes per window.
        S_min: Minimum number of regimes to scan (default 1).
        criterion: Selection criterion.
        n_starts: EM restarts per window.
        shared_mean: Variance-only switching.
        sticky: Persistence prior.
        random_state: Base seed (each window gets random_state + window_index).

    Returns:
        dict with keys:
            model (str): model used.
            criterion (str): criterion used.
            windows (list[dict]): per window:
                label (str): window label.
                window_start (int): resolved start index.
                window_end (int): resolved end index.
                window_length (int): T for this window.
                series (dict): per series, same structure as fit_regimes output.
            comparison (dict): per series name:
                S_by_window (dict): {label: S} — regime count per window.
                mean_vol_by_window (dict): {label: list[float]} — per-state vol per window.
                persistence_by_window (dict): {label: list[float]} — diag(A) per window.
                occupancy_by_window (dict): {label: list[float]} — state occupancies per window.
                bic_by_window (dict): {label: float} — BIC per window.
    """
    window_results = []
    for w_idx, w in enumerate(windows):
        label = str(w.get("label", f"window_{w_idx}"))
        start = int(w.get("start", 0))
        end   = int(w.get("end",   0))
        seed  = random_state + w_idx

        res = fit_regimes_window(
            data=data, series_names=series_names,
            window_start=start, window_end=end,
            model=model, S_max=S_max, S_min=S_min, criterion=criterion,
            n_starts=n_starts, shared_mean=shared_mean, sticky=sticky,
            random_state=seed,
        )
        window_results.append({"label": label, **res})

    # Build comparison dict
    comparison: Dict[str, Any] = {}
    for col in series_names:
        S_by_win         = {}
        mean_vol_by_win  = {}
        persistence_by_win = {}
        occupancy_by_win = {}
        bic_by_win       = {}

        for wr in window_results:
            lbl   = wr["label"]
            s_dat = wr["series"].get(col, {})
            S_by_win[lbl]           = int(s_dat.get("S", 0))
            mean_vol_by_win[lbl]    = [round(rs["vol"], 6)
                                       for rs in s_dat.get("regime_stats", [])]
            persistence_by_win[lbl] = [round(float(row[i]), 4)
                                       for i, row in enumerate(
                                           s_dat.get("transition_matrix", []))]
            occupancy_by_win[lbl]   = [round(rs["occupancy_pct"], 1)
                                       for rs in s_dat.get("regime_stats", [])]
            bic_by_win[lbl]         = round(float(s_dat.get("bic", float("nan"))), 2)

        comparison[col] = {
            "S_by_window":           S_by_win,
            "mean_vol_by_window":    mean_vol_by_win,
            "persistence_by_window": persistence_by_win,
            "occupancy_by_window":   occupancy_by_win,
            "bic_by_window":         bic_by_win,
        }

    return {
        "model":      model,
        "criterion":  criterion,
        "windows":    window_results,
        "comparison": comparison,
    }


def generate_regime_plots(
    result_key: Annotated[
        str,
        "Store key of a fit produced by fit_regimes(result_key=...). "
        "The full stored result (states, posteriors, index) is rebuilt from "
        "the store; nothing large passes through the caller.",
    ],
    data_key: Annotated[
        str,
        "Store key of the data the fit was computed on. Empty = use the "
        "data_key recorded in the stored fit result.",
    ] = "",
    theme: Annotated[
        Literal["dark", "light", "minimal"],
        "Plot theme.",
    ] = "dark",
    last_years: Annotated[
        int,
        "Plot only the most recent N years of the fitted history.",
    ] = 20,
    points_per_year: Annotated[
        int,
        "Observations per year at the fit's frequency (52 weekly, 252 daily).",
    ] = 52,
) -> dict:
    """Render ALL regime plots for a stored fit into the SQLite depot.

    Rebuilds the fitted run (values + states + posteriors) from the store,
    renders one series_with_regimes chart per series plus the two barcode
    charts headlessly (Agg), and persists the PNGs in the depot's plots
    table. Use db_list_plots() to enumerate them and db_export_plot() to save
    one to disk. Requires an active depot (init_regime_db).

    Returns:
        dict with keys:
            result_key (str), n_plots (int), plot_keys (list[str]),
            series (list[str]), theme (str).
    """
    rec = _sread(result_key)
    if not isinstance(rec, dict) or "series" not in rec:
        raise ValueError(f"{result_key!r} is not a stored fit_regimes result")
    dkey = data_key or rec.get("data_key") or ""
    if not dkey:
        raise ValueError(
            "the stored fit has no data_key (inline data= fit): pass the "
            "data_key of the stored series explicitly")
    payload = _sread(dkey)
    Y = _ensure_2d(np.asarray(payload["Y"], dtype=float))
    columns = list(payload["columns"])

    idx_raw = rec.get("index") or payload.get("index") or []
    idx = pd.to_datetime(list(idx_raw), errors="coerce")
    if idx.isna().all():
        idx = pd.RangeIndex(len(Y))

    parts: List[pd.DataFrame] = []
    labels_map: Dict[str, List[str]] = {}
    meta: Dict[str, Any] = {}
    names = [c for c in columns if c in rec["series"]]
    if not names:
        raise ValueError("stored fit and stored data share no series names")
    for j, name in enumerate(names):
        sd = rec["series"][name]
        gamma = np.asarray(sd["state_probs"], dtype=float)
        S = int(sd["S"])
        frame = {
            f"{name}_value":   Y[:, columns.index(name)],
            f"{name}_state":   np.asarray(sd["states"], dtype=int),
            f"{name}_highvol": np.asarray(sd["high_vol_flag"], dtype=int),
            f"P_{name}_HV":    np.asarray(sd["prob_high_vol"], dtype=float),
        }
        for s in range(S):
            frame[f"P_{name}_S{s}"] = gamma[:, s]
        parts.append(pd.DataFrame(frame, index=idx))
        labels_map[name] = list(sd["labels"])
        meta[name] = {"S": S, "labels": list(sd["labels"]),
                      "bic": sd.get("bic"), "loglik": sd.get("loglik")}

    run = DBRegimeRun(panel=pd.concat(parts, axis=1), rows=names,
                      labels_map=labels_map, meta=meta)
    run.set_theme(theme)
    plot_keys = run.save_all_plots_to_db(
        result_key=result_key, data_key=dkey,
        last_years=last_years, points_per_year=points_per_year)
    return {"result_key": result_key, "n_plots": len(plot_keys),
            "plot_keys": plot_keys, "series": names, "theme": theme}
