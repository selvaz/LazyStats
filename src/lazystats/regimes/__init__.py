# -*- coding: utf-8 -*-
"""
LazyHMM — Hidden Markov Model regime detection for financial time series.

Two layers:
  * lazystats.regimes.core  — pure modelling core (engines, fitting, data structures, plots)
  * lazystats.regimes.tools — LLM tool API + an in-process/SQLite result store
  * lazystats.regimes.db    — optional SQLite persistence depot

Quick start
-----------
    import pandas as pd
    from lazystats.regimes import MSRegimeEngine

    engine = MSRegimeEngine(S_max=4, n_starts=10, criterion="bic")
    run = engine.fit(df, model="panel")   # "panel" | "joint_diag" | "joint_full"
    run.plot_series_with_regimes("SPY")

The single ``model`` axis replaces the old (mode, cov_type) pair:
  panel       -> independent univariate HMM per column
  joint_diag  -> one joint multivariate HMM, diagonal covariance
  joint_full  -> one joint multivariate HMM, full covariance
  categorical -> discrete-emission HMM (use fit_categorical_regimes / fit_discrete_hmm)
"""
from __future__ import annotations

# ── Core modelling API (engines return the plain RegimeRun) ────────────────
from .core import (
    # type alias
    ModelKind,
    # data structures
    FitWarning, FitResult, PlotTheme, RegimeRun,
    # gaussian engine + selection + post-processing
    MSGaussianHMM, MSRegimeEngine, RegimeEngine,
    select_num_regimes, fit_with_auto_S, fit_autos_Y,
    reorder_fitresult, regime_labels_from_S,
    # user-supplied parameters: warm-start or fixed inference
    HMMParams, infer_with_params, fit_from_params,
    # discrete (categorical) model
    DiscreteFitResult, fit_discrete_hmm, fit_with_auto_S_categorical,
    # multivariate independent-emission model
    MultiVarFitResult, fit_multivar_hmm, fit_with_auto_S_multivar,
)

# ── Tool layer: LLM-facing functions, store management, DB-aware plotting ──
from .tools import (
    DBRegimeRun,
    connect_lazy_store,
    regime_store_list, regime_store_load, regime_store_delete,
    load_time_series,
    fit_regimes, scan_state_counts,
    get_current_regime, get_regime_changes, get_regime_summary, compare_emission_models,
    fit_categorical_regimes, fit_regimes_window, compare_regime_windows,
    generate_regime_plots,
    init_regime_db,
    # parameter persistence + fixed-parameter inference (tool layer)
    regime_params_save, regime_params_load, apply_regime_params,
    regime_params_list,
)

# ── Data-source loaders (external providers → depot under a data_key) ──────
from .datasources import load_from_datahub

# ── Result contract: emit regimes in the shared lazydatacore envelope ──────
from .contract import to_analysis_results

# Submodules importable as lazystats.regimes.core / lazystats.regimes.tools / lazystats.regimes.db
from . import core, tools, db, datasources  # noqa: F401

__version__ = "0.2.0"

__all__ = [
    # data structures
    "FitWarning", "FitResult", "PlotTheme", "RegimeRun", "DBRegimeRun",
    "DiscreteFitResult", "MultiVarFitResult", "ModelKind",
    # engines
    "MSGaussianHMM", "MSRegimeEngine", "RegimeEngine",
    # core functions
    "select_num_regimes", "fit_with_auto_S", "fit_autos_Y",
    "reorder_fitresult", "regime_labels_from_S",
    "HMMParams", "infer_with_params", "fit_from_params",
    "fit_discrete_hmm", "fit_with_auto_S_categorical",
    "fit_multivar_hmm", "fit_with_auto_S_multivar",
    # LLM tool API
    "load_time_series", "fit_regimes", "scan_state_counts",
    "get_current_regime", "get_regime_changes", "get_regime_summary",
    "compare_emission_models",
    "fit_categorical_regimes", "fit_regimes_window", "compare_regime_windows",
    "generate_regime_plots",
    # parameter persistence + fixed-parameter inference
    "regime_params_save", "regime_params_load", "apply_regime_params",
    "regime_params_list",
    # store / persistence
    "connect_lazy_store", "regime_store_list", "regime_store_load",
    "regime_store_delete", "init_regime_db",
    # data-source loaders
    "load_from_datahub",
    # result contract (lazydatacore envelope)
    "to_analysis_results",
    "__version__",
]
