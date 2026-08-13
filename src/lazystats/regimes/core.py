from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import os
import warnings
import numpy as np
import pandas as pd

# Imports per la parte di esempio e visualizzazione
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm

import matplotlib as mpl

# Tentativo importazione librerie specifiche
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as e:
    raise ImportError("hmmlearn is required. Install via: pip install hmmlearn") from e

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ---------------------------------------------------------
# Configurazione Warnings (Windows/MKL)
# ---------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL*",
    category=UserWarning,
)

# ---------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------
CovType = Literal["diag", "full"]          # internal covariance switch (engine only)
Criterion = Literal["bic", "aic", "hqic"]
OrderBy = Literal["vol", "mean"]

# Public, single-axis model schema. Replaces the old (mode, cov_type) pair.
#   panel      -> independent univariate HMM per column (cov irrelevant: k=1)
#   joint_diag -> one joint multivariate HMM, diagonal covariance
#   joint_full -> one joint multivariate HMM, full covariance
#   categorical-> discrete-emission HMM (use fit_categorical_regimes / fit_discrete_hmm)
ModelKind = Literal["panel", "joint_diag", "joint_full", "categorical"]
_FitMode = Literal["panel", "joint"]       # internal driver mode (purges univariate/multivariate)


def _resolve_model(model: str) -> Tuple[str, CovType]:
    """Map a public ModelKind to the internal (fit_mode, cov_type) pair.

    Raises ValueError for 'categorical' (handled by the discrete functions) and
    for any unknown value. The meaningless 'univariate + full' combination is
    impossible by construction: 'panel' always implies diagonal covariance.
    """
    table = {
        "panel": ("panel", "diag"),
        "joint_diag": ("joint", "diag"),
        "joint_full": ("joint", "full"),
    }
    if model in table:
        return table[model]
    if model == "categorical":
        raise ValueError(
            "model='categorical' is not handled by the continuous engine. "
            "Use fit_categorical_regimes()/fit_discrete_hmm() for integer-symbol data."
        )
    raise ValueError(
        f"Unknown model '{model}'. Choose from: 'panel', 'joint_diag', 'joint_full', 'categorical'."
    )


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------
@dataclass
class FitWarning:
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitResult:
    S: int
    cov_type: CovType
    shared_mean: bool
    seed: int

    converged: bool
    n_iter: int
    loglik: float
    bic: float

    startprob_: np.ndarray
    transmat_: np.ndarray
    means_: np.ndarray
    covars_: np.ndarray

    gamma_: np.ndarray
    viterbi_path_: np.ndarray

    warnings: List[FitWarning] = field(default_factory=list)

    def summary(self) -> str:
        if self.gamma_.ndim == 2 and self.gamma_.size > 0:
            occ = self.gamma_.mean(axis=0)
            occ_str = np.array2string(np.round(occ, 3), separator=", ")
        else:
            occ_str = "n/a"

        return (
            f"FitResult(S={self.S}, LogLik={self.loglik:.2f}, BIC={self.bic:.2f}, Converged={self.converged})\n"
            f" - Shared Mean: {self.shared_mean}\n"
            f" - Occupancies: {occ_str}\n"
            f" - Warnings: {len(self.warnings)}"
        )


# --- DROP-IN UPDATE: RegimeRun con tema configurabile, senza side-effect globali ---
# Innestabile: sostituisci la tua classe RegimeRun con questa.
# Dipendenze richieste giÃ  presenti nel file: matplotlib.pyplot as plt, matplotlib.colors as mcolors,
# matplotlib.patches as mpatches, matplotlib.cm as cm

from dataclasses import dataclass, field

# =========================
# THEME SYSTEM
# =========================

@dataclass(frozen=True)
class PlotTheme:
    # rc
    figure_facecolor: str
    axes_facecolor: str
    savefig_facecolor: str
    axes_edgecolor: str
    axes_labelcolor: str
    xtick_color: str
    ytick_color: str
    text_color: str
    grid_color: str
    grid_alpha: float
    axes_grid_default: bool
    font_size: int

    # series
    line_color: str
    line_width: float

    # regime overlays
    regime_alpha: float
    legend_patch_alpha: float

    # palettes
    regime_palette: Tuple[str, ...]
    binary_palette: Tuple[str, str]  # (0,1)

    def rc_dict(self) -> Dict[str, Any]:
        return {
            "figure.facecolor": self.figure_facecolor,
            "axes.facecolor": self.axes_facecolor,
            "savefig.facecolor": self.savefig_facecolor,
            "axes.edgecolor": self.axes_edgecolor,
            "axes.labelcolor": self.axes_labelcolor,
            "xtick.color": self.xtick_color,
            "ytick.color": self.ytick_color,
            "text.color": self.text_color,
            "grid.color": self.grid_color,
            "grid.alpha": self.grid_alpha,
            "axes.grid": self.axes_grid_default,
            "font.size": self.font_size,
        }

    def discrete_cmap(self, S: int) -> mcolors.ListedColormap:
        cols = [self.regime_palette[i % len(self.regime_palette)] for i in range(int(S))]
        return mcolors.ListedColormap(cols, name="theme_discrete")

    def binary_cmap(self) -> mcolors.ListedColormap:
        return mcolors.ListedColormap(list(self.binary_palette), name="theme_binary")


# =========================
# REGIME RUN
# =========================

@dataclass
class RegimeRun:
    panel: Any  # pandas.DataFrame
    rows: List[str]
    labels_map: dict
    meta: dict

    theme: PlotTheme = field(init=False)
    _THEMES: Dict[str, PlotTheme] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._init_themes()
        self.set_theme("bloomberg")

    # -----------------
    # THEME REGISTRY
    # -----------------
    def _init_themes(self) -> None:
        self._THEMES = {
            "bloomberg": PlotTheme(
                figure_facecolor="#0B0F14",
                axes_facecolor="#0B0F14",
                savefig_facecolor="#0B0F14",
                axes_edgecolor="#2B3440",
                axes_labelcolor="#D7E1EA",
                xtick_color="#A9B4C0",
                ytick_color="#A9B4C0",
                text_color="#D7E1EA",
                grid_color="#1F2730",
                grid_alpha=0.35,
                axes_grid_default=False,
                font_size=11,
                line_color="#D7E1EA",
                line_width=1.0,
                regime_alpha=0.22,
                legend_patch_alpha=0.55,
                regime_palette=(
                    "#2D7DD2", "#00B3A4", "#7BDFF2", "#F4D35E",
                    "#FAA916", "#EE6352", "#B388EB", "#5EF38C",
                    "#F7A8B8", "#C2C5CC",
                ),
                binary_palette=("#0B0F14", "#FAA916"),
            ),
            "light": PlotTheme(
                figure_facecolor="#FFFFFF",
                axes_facecolor="#FFFFFF",
                savefig_facecolor="#FFFFFF",
                axes_edgecolor="#333333",
                axes_labelcolor="#111111",
                xtick_color="#222222",
                ytick_color="#222222",
                text_color="#111111",
                grid_color="#DDDDDD",
                grid_alpha=0.7,
                axes_grid_default=False,
                font_size=11,
                line_color="#111111",
                line_width=1.0,
                regime_alpha=0.18,
                legend_patch_alpha=0.6,
                regime_palette=(
                    "#1f77b4", "#ff7f0e", "#2ca02c",
                    "#d62728", "#9467bd", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                ),
                binary_palette=("#FFFFFF", "#FF7F0E"),
            ),
            "minimal": PlotTheme(
                figure_facecolor="#FFFFFF",
                axes_facecolor="#FFFFFF",
                savefig_facecolor="#FFFFFF",
                axes_edgecolor="#000000",
                axes_labelcolor="#000000",
                xtick_color="#000000",
                ytick_color="#000000",
                text_color="#000000",
                grid_color="#CCCCCC",
                grid_alpha=0.4,
                axes_grid_default=False,
                font_size=10,
                line_color="#000000",
                line_width=1.2,
                regime_alpha=0.15,
                legend_patch_alpha=0.5,
                regime_palette=("#000000", "#555555", "#AAAAAA", "#DDDDDD"),
                binary_palette=("#FFFFFF", "#000000"),
            ),
        }

    # -----------------
    # PUBLIC API
    # -----------------
    def set_theme(self, name: str) -> None:
        key = str(name).lower().strip()
        if key not in self._THEMES:
            raise ValueError(f"Theme '{name}' not available. Available: {sorted(self._THEMES.keys())}")
        self.theme = self._THEMES[key]

    # Alias â€œset_colorâ€ come richiesto: stesso comportamento di set_theme
    def set_color(self, name: str) -> None:
        self.set_theme(name)

    def available_themes(self) -> List[str]:
        return sorted(self._THEMES.keys())

    # -----------------
    # INTERNAL UTILS
    # -----------------
    def _slice_last(self, last_years: int, points_per_year: int):
        return self.panel.tail(int(points_per_year * last_years))

    def _format_time_axis(self, ax, idx, n_xticks: int):
        if len(idx) == 0:
            return
        ticks = np.linspace(0, len(idx) - 1, n_xticks).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([idx[i].strftime("%Y-%m") for i in ticks], rotation=0)

    def _with_theme(self):
        # pipeline-safe: niente side-effect globali
        return mpl.rc_context(rc=self.theme.rc_dict())

    # -----------------
    # PLOTS
    # -----------------
    def plot_barcode_states(
        self,
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        figsize: Tuple[int, int] = (16, 6),
        n_xticks: int = 10,
        title: str = "Regimi (state id)",
    ) -> None:
        with self._with_theme():
            df = self._slice_last(last_years, points_per_year)
            mat = np.vstack([df[f"{r}_state"].values for r in self.rows]).astype(float)

            vmax = np.nanmax(mat)
            if not np.isfinite(vmax):
                raise ValueError("No finite state values to plot.")
            Sg = int(vmax) + 1

            cmap = self.theme.discrete_cmap(Sg)
            bounds = np.arange(-0.5, Sg + 0.5, 1.0)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

            ax.set_yticks(range(len(self.rows)))
            ax.set_yticklabels(self.rows)

            self._format_time_axis(ax, df.index, n_xticks)
            ax.set_title(title, pad=12)

            cbar = fig.colorbar(im, ax=ax, ticks=np.arange(0, Sg, 1))
            cbar.set_label("State id")

            plt.tight_layout()
            plt.show()

    def plot_barcode_highvol(
        self,
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        figsize: Tuple[int, int] = (16, 6),
        n_xticks: int = 10,
        title: str = "High-Vol (binary)",
    ) -> None:
        with self._with_theme():
            df = self._slice_last(last_years, points_per_year)
            mat = np.vstack([df[f"{r}_highvol"].values for r in self.rows]).astype(float)

            cmap = self.theme.binary_cmap()
            bounds = np.array([-0.5, 0.5, 1.5])
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

            ax.set_yticks(range(len(self.rows)))
            ax.set_yticklabels(self.rows)

            self._format_time_axis(ax, df.index, n_xticks)
            ax.set_title(title, pad=12)

            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.set_label("High-Vol (0/1)")

            plt.tight_layout()
            plt.show()

    def plot_series_with_regimes(
        self,
        name: str,
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        alpha: Optional[float] = None,
        figsize: Tuple[int, int] = (14, 3),
        title: Optional[str] = None,
        line_color: Optional[str] = None,
        line_width: Optional[float] = None,
        cmap_name: Optional[str] = None,  # override palette: es. "Set2"
    ) -> None:
        with self._with_theme():
            df = self._slice_last(last_years, points_per_year)
            series_col = f"{name}_value"
            state_col = f"{name}_state"

            s = df[series_col].dropna()
            st = df[state_col].reindex(s.index).astype(int)
            if len(st) == 0:
                raise ValueError("No overlap between series and state index.")

            S_local = int(st.max() + 1)
            labels = self.labels_map.get(name, [f"State {i}" for i in range(S_local)])[:S_local]

            if cmap_name is None:
                cmap = self.theme.discrete_cmap(S_local)
                color_fn = lambda i: cmap(i)
            else:
                cmap = cm.get_cmap(cmap_name, S_local)
                color_fn = lambda i: cmap(i)

            a = float(self.theme.regime_alpha if alpha is None else alpha)
            lc = self.theme.line_color if line_color is None else line_color
            lw = float(self.theme.line_width if line_width is None else line_width)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(s.index, s.values, color=lc, linewidth=lw)

            cur = int(st.iloc[0])
            start = s.index[0]
            for t in range(1, len(st)):
                now = int(st.iloc[t])
                if now != cur:
                    end = st.index[t]
                    ax.axvspan(start, end, color=color_fn(cur), alpha=a, lw=0)
                    cur = now
                    start = end
            ax.axvspan(start, s.index[-1], color=color_fn(cur), alpha=a, lw=0)

            patches = [
                mpatches.Patch(color=color_fn(i), alpha=self.theme.legend_patch_alpha, label=labels[i])
                for i in range(S_local)
            ]
            ax.legend(handles=patches, loc="upper left", frameon=False)

            ax.set_title(title if title is not None else name, pad=10)
            ax.grid(True, axis="y", alpha=0.25)

            plt.tight_layout()
            plt.show()

    def plot_small_multiples(
        self,
        names: Sequence[str],
        *,
        last_years: int = 20,
        points_per_year: int = 52,
        alpha: Optional[float] = None,
        figsize_per_row: Tuple[int, int] = (14, 3),
        cmap_name: Optional[str] = None,
    ) -> None:
        for nm in names:
            self.plot_series_with_regimes(
                nm,
                last_years=last_years,
                points_per_year=points_per_year,
                alpha=alpha,
                figsize=figsize_per_row,
                title=nm,
                cmap_name=cmap_name,
            )



# ---------------------------------------------------------
# Helper Functions (Math & Initialization)
# ---------------------------------------------------------
def _ensure_2d(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError("Y must be 1D or 2D (T,) or (T,k).")
    return Y

def _validate_S_bounds(S_min: Any, S_max: Any) -> None:
    """Validate the state-count scan bounds.

    Requires ``S_min`` and ``S_max`` to be integers (bool is rejected) with
    ``1 <= S_min <= S_max``. Raises ``ValueError`` with a clear message otherwise.
    """
    if isinstance(S_min, bool) or isinstance(S_max, bool):
        raise ValueError("S_min and S_max must be integers, not bool.")
    if not isinstance(S_min, (int, np.integer)) or not isinstance(S_max, (int, np.integer)):
        raise ValueError(
            f"S_min and S_max must be integers (got S_min={S_min!r}, S_max={S_max!r})."
        )
    S_min = int(S_min)
    S_max = int(S_max)
    if S_min < 1:
        raise ValueError(f"S_min must be >= 1 (got S_min={S_min}).")
    if S_min > S_max:
        raise ValueError(
            f"S_min must be <= S_max (got S_min={S_min}, S_max={S_max})."
        )

def _count_free_params(S: int, k: int, cov_type: CovType, shared_mean: bool) -> int:
    n_params = (S - 1) + S * (S - 1)  # startprob + transmat
    n_params += (k if shared_mean else S * k)  # means
    if cov_type == "diag":
        n_params += S * k
    elif cov_type == "full":
        n_params += S * (k * (k + 1) // 2)
    else:
        raise ValueError("cov_type must be 'diag' or 'full'.")
    return int(n_params)

def _bic(loglik: float, n_params: int, T: int) -> float:
    return float(n_params * np.log(max(T, 2)) - 2.0 * loglik)

def _aic(loglik: float, n_params: int) -> float:
    return float(2.0 * n_params - 2.0 * loglik)

def _hqic(loglik: float, n_params: int, T: int) -> float:
    T2 = max(T, 3)
    return float(-2.0 * loglik + 2.0 * n_params * np.log(np.log(T2)))

def _criterion_score(criterion: Criterion, loglik: float, n_params: int, T: int) -> float:
    if criterion == "bic":
        return _bic(loglik, n_params, T)
    if criterion == "aic":
        return _aic(loglik, n_params)
    if criterion == "hqic":
        return _hqic(loglik, n_params, T)
    raise ValueError("criterion must be one of: 'bic', 'aic', 'hqic'.")

def _get_global_mean(Y: np.ndarray) -> np.ndarray:
    return np.mean(Y, axis=0, keepdims=True)

def _get_global_cov(Y: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.array([[float(np.var(Y, ddof=1))]])
    C = np.cov(Y, rowvar=False, ddof=1)
    return np.asarray(C, dtype=float)

def _make_sticky_transmat(S: int, sticky: float, rng: np.random.RandomState) -> np.ndarray:
    S = int(S)
    if S <= 0:
        raise ValueError("S must be >= 1.")
    if S == 1:
        return np.array([[1.0]], dtype=float)

    sticky = float(sticky)
    sticky = min(max(sticky, 1.0 / S), 0.999999)

    off = (1.0 - sticky) / (S - 1)
    P = np.full((S, S), off, dtype=float)
    np.fill_diagonal(P, sticky)

    P += 1e-6 * rng.rand(S, S)  # break symmetry
    P /= P.sum(axis=1, keepdims=True)
    return P

def _init_means_kmeans(Y: np.ndarray, S: int, rng: np.random.RandomState) -> np.ndarray:
    T, _ = Y.shape
    if _HAS_SKLEARN and T >= S:
        km = KMeans(n_clusters=S, n_init=10, random_state=int(rng.randint(0, 2**31 - 1)))
        km.fit(Y)
        return np.asarray(km.cluster_centers_, dtype=float)
    idx = rng.choice(T, size=S, replace=(T < S))
    return np.asarray(Y[idx], dtype=float)

def _init_covars_variance_switching(
    Y: np.ndarray,
    S: int,
    cov_type: CovType,
    rng: np.random.RandomState,
    jitter: float,
    var_floor: float,
) -> np.ndarray:
    """Robust initialization for covariance matrices."""
    T, k = Y.shape
    mu = _get_global_mean(Y)
    E = Y - mu

    if k == 1:
        e2 = (E[:, 0] ** 2)
        qs = np.quantile(e2, np.linspace(0.2, 0.8, S))
        vars_init = np.clip(qs, var_floor, None)
        rng.shuffle(vars_init)

        if cov_type == "diag":
            return vars_init.reshape(S, 1)

        covs = np.zeros((S, 1, 1), dtype=float)
        for s in range(S):
            covs[s, 0, 0] = float(vars_init[s] + jitter)
        return covs

    # Multivariate
    e2 = np.sum(E * E, axis=1)
    qs = np.quantile(e2, np.linspace(0.2, 0.8, S))
    scales = np.clip(qs / max(np.mean(e2), 1e-12), 0.1, 10.0)
    rng.shuffle(scales)

    global_cov = _get_global_cov(Y, k)

    if cov_type == "diag":
        base_diag = np.clip(np.diag(global_cov), var_floor, None)
        covars = np.vstack([base_diag * scales[s] for s in range(S)])
        covars = np.clip(covars, var_floor, None)
        return covars

    covars = np.zeros((S, k, k), dtype=float)
    for s in range(S):
        covars[s] = global_cov * scales[s]
        covars[s].flat[:: k + 1] += jitter
    return covars

def _min_eigvals(covars: np.ndarray, cov_type: CovType) -> np.ndarray:
    C = np.asarray(covars)
    if cov_type == "diag":
        # hmmlearn ≥0.3 stores diag covars as (S, k, k) diagonal matrices.
        # Extract only the diagonal to avoid the off-diagonal zeros pulling min to 0.
        if C.ndim == 3:
            return np.array([np.diag(C[s]).min() for s in range(C.shape[0])])
        return C.reshape(C.shape[0], -1).min(axis=1)
    return np.array([np.linalg.eigvalsh(C[s]).min() for s in range(C.shape[0])])

def _expected_durations(P: np.ndarray) -> np.ndarray:
    pdiag = np.diag(P)
    return 1.0 / np.clip(1.0 - pdiag, 1e-12, None)


# ---------------------------------------------------------
# Regime Ordering & Labels
# ---------------------------------------------------------
def _state_vol_measure(covars: np.ndarray, cov_type: CovType) -> np.ndarray:
    C = np.asarray(covars, dtype=float)
    if cov_type == "diag":
        # hmmlearn ≥0.3 stores diag covars as (S, k, k) diagonal matrices.
        # Use only the diagonal values to compute the vol measure.
        if C.ndim == 3:
            return np.array([np.diag(C[s]).mean() for s in range(C.shape[0])])
        return C.reshape(C.shape[0], -1).mean(axis=1)
    return np.array([np.trace(C[s]) for s in range(C.shape[0])], dtype=float)

def _state_mean_measure(means: np.ndarray) -> np.ndarray:
    M = np.asarray(means, dtype=float)
    return M.mean(axis=1)

def reorder_fitresult(res: FitResult, by: OrderBy = "vol", ascending: bool = True) -> FitResult:
    S = int(res.S)
    if S <= 1:
        return res

    if by == "vol":
        key = _state_vol_measure(res.covars_, res.cov_type)
    elif by == "mean":
        key = _state_mean_measure(res.means_)
    else:
        raise ValueError("by must be 'vol' or 'mean'.")

    order = np.argsort(key)
    if not ascending:
        order = order[::-1]

    inv = np.empty_like(order)
    inv[order] = np.arange(S)

    startprob_ = np.asarray(res.startprob_)[order]
    transmat_ = np.asarray(res.transmat_)[order][:, order]
    means_ = np.asarray(res.means_)[order]
    covars_ = np.asarray(res.covars_)[order]
    gamma_ = np.asarray(res.gamma_)[:, order]
    viterbi_path_ = inv[np.asarray(res.viterbi_path_, dtype=int)]

    new_warnings = list(res.warnings)
    new_warnings.append(FitWarning(
        code="INFO_STATE_REORDER",
        severity="info",
        message=f"Regimes reordered by {by} ({'asc' if ascending else 'desc'}).",
        context={"order_new_to_old": order.tolist(), "key": key.tolist()},
    ))

    return replace(
        res,
        startprob_=startprob_,
        transmat_=transmat_,
        means_=means_,
        covars_=covars_,
        gamma_=gamma_,
        viterbi_path_=viterbi_path_,
        warnings=new_warnings,
    )

def regime_labels_from_S(S: int) -> List[str]:
    if S == 1:
        return ["Single"]
    if S == 2:
        return ["Low Vol", "High Vol"]
    if S == 3:
        return ["Low Vol", "Mid Vol", "High Vol"]
    if S == 4:
        return ["Low Vol", "Mid-Low Vol", "Mid-High Vol", "High Vol"]
    if S == 5:
        return ["Very Low", "Low", "Mid", "High", "Extreme"]
    return [f"Regime {i}" for i in range(S)]


# ---------------------------------------------------------
# Core Class: MSGaussianHMM
# ---------------------------------------------------------
class MSGaussianHMM:
    """
    Wrapper multi-start su hmmlearn.GaussianHMM.
    """
    def __init__(
        self,
        S: int,
        n_starts: int = 10,
        cov_type: CovType = "diag",
        shared_mean: bool = False,
        n_iter: int = 300,
        tol: float = 1e-4,
        sticky: float = 0.95,
        occupancy_min: float = 0.02,
        min_covar: float = 1e-6,
        jitter: float = 1e-6,
        eigen_eps: float = 1e-10,
        var_floor: float = 1e-8,
        random_state: int = 42,
        suppress_hmmlearn_warnings: bool = True,
        diversify_starts: bool = True,
        mean_jitter: float = 0.2,
        startprob_alpha: float = 4.0,
        sticky_jitter: float = 0.04,
    ):
        self.S = int(S)
        self.n_starts = int(n_starts)
        self.cov_type = cov_type
        self.shared_mean = bool(shared_mean)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.sticky = float(sticky)
        self.occupancy_min = float(occupancy_min)
        self.min_covar = float(min_covar)
        self.jitter = float(jitter)
        self.eigen_eps = float(eigen_eps)
        self.var_floor = float(var_floor)
        self.suppress_hmmlearn_warnings = bool(suppress_hmmlearn_warnings)

        # Multi-start diversification. The default ("variance-switching" shuffle
        # only) gives little spread across starts, especially for small S, so
        # n_starts is largely wasted. When enabled, start 0 keeps the clean
        # data-informed init (no regression vs a single clean start) while
        # starts >= 1 are progressively diversified to explore distinct basins.
        self.diversify_starts = bool(diversify_starts)
        self.mean_jitter = float(mean_jitter)
        self.startprob_alpha = float(startprob_alpha)
        self.sticky_jitter = float(sticky_jitter)

        self.rng = np.random.RandomState(random_state)

        self.best_result_: Optional[FitResult] = None
        self.all_results_: List[FitResult] = []

    def fit(self, Y: np.ndarray) -> "MSGaussianHMM":
        Y = _ensure_2d(Y)
        seeds = self.rng.randint(0, 2**31 - 1, size=self.n_starts)
        results: List[FitResult] = [
            self._fit_single_chain(Y, int(sd), start_idx=i) for i, sd in enumerate(seeds)
        ]

        valid = [r for r in results if np.isfinite(r.loglik)]
        if not valid:
            raise ValueError("All initialization attempts failed (no finite loglik).")

        self.all_results_ = sorted(valid, key=lambda x: x.loglik, reverse=True)
        self.best_result_ = self.all_results_[0]
        return self

    def _fit_single_chain(self, Y: np.ndarray, seed: int, start_idx: int = 0) -> FitResult:
        Y = _ensure_2d(Y)
        T, k = Y.shape

        rng_local = np.random.RandomState(seed)
        warn_list: List[FitWarning] = []

        # start 0 is always the clean, data-informed init; only later starts
        # get perturbed, so the best-of selection can never do worse than a
        # single clean start.
        diversify = self.diversify_starts and start_idx > 0

        if diversify and self.S > 1:
            startprob0 = rng_local.dirichlet(self.startprob_alpha * np.ones(self.S))
        else:
            startprob0 = np.full(self.S, 1.0 / self.S)

        if diversify:
            lo = max(1.0 / self.S, self.sticky - self.sticky_jitter)
            hi = min(0.999, self.sticky + self.sticky_jitter)
            sticky_eff = float(rng_local.uniform(lo, hi)) if hi > lo else self.sticky
        else:
            sticky_eff = self.sticky
        trans0 = _make_sticky_transmat(self.S, sticky_eff, rng_local)

        if self.shared_mean:
            mu = _get_global_mean(Y)
            means0 = np.repeat(mu, self.S, axis=0)
            covars0 = _init_covars_variance_switching(
                Y, self.S, self.cov_type, rng_local, jitter=self.jitter, var_floor=self.var_floor
            )
            params = "stc"
        else:
            means0 = _init_means_kmeans(Y, self.S, rng_local)
            if diversify and self.mean_jitter > 0:
                col_std = np.std(Y, axis=0, ddof=1)
                col_std = np.where(np.isfinite(col_std) & (col_std > 0), col_std, 1.0)
                means0 = means0 + self.mean_jitter * col_std * rng_local.standard_normal(means0.shape)
            covars0 = _init_covars_variance_switching(
                Y, self.S, self.cov_type, rng_local, jitter=self.jitter, var_floor=self.var_floor
            )
            params = "stmc"

        model = GaussianHMM(
            n_components=self.S,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            tol=self.tol,
            verbose=False,
            init_params="",
            params=params,
            random_state=seed,
            min_covar=self.min_covar,
        )

        model.startprob_ = startprob0
        model.transmat_ = trans0
        model.means_ = means0
        model.covars_ = covars0

        if self.suppress_hmmlearn_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(Y)
                except Exception as e:
                    return self._empty(seed, T, [FitWarning("ERR_FIT", "critical", "Fit crashed.", {"exception": repr(e)})])
        else:
            try:
                model.fit(Y)
            except Exception as e:
                return self._empty(seed, T, [FitWarning("ERR_FIT", "critical", "Fit crashed.", {"exception": repr(e)})])

        converged = bool(getattr(model.monitor_, "converged", False))
        n_it = int(getattr(model.monitor_, "iter", self.n_iter))
        if not converged:
            warn_list.append(FitWarning("WARN_NO_CONVERGENCE", "warning", "Model did not converge.", {"iter": n_it}))

        try:
            ll = float(model.score(Y))
        except Exception as e:
            return self._empty(seed, T, warn_list + [FitWarning("ERR_SCORE", "critical", "score() failed.", {"exception": repr(e)})])

        try:
            _, vpath = model.decode(Y, algorithm="viterbi")
            _, gamma = model.score_samples(Y)
        except Exception as e:
            return self._empty(seed, T, warn_list + [FitWarning("ERR_DECODE", "critical", "decode/score_samples failed.", {"exception": repr(e)})])

        vpath = np.asarray(vpath, dtype=int)
        gamma = np.asarray(gamma, dtype=float)

        occ = gamma.mean(axis=0)
        if np.any(occ < self.occupancy_min):
            warn_list.append(FitWarning(
                "WARN_TINY_OCCUPANCY", "warning",
                "At least one regime has very low occupancy (ghost state / collapse).",
                {"occupancy": occ, "threshold": self.occupancy_min}
            ))

        min_eigs = _min_eigvals(model.covars_, self.cov_type)
        if np.any(min_eigs < self.eigen_eps):
            warn_list.append(FitWarning(
                "WARN_DEGENERATE_COV", "warning",
                "Covariance/variance nearly singular in at least one regime.",
                {"min_eigs": min_eigs, "eigen_eps": self.eigen_eps}
            ))

        warn_list.append(FitWarning(
            "INFO_OFFLINE_DECODING", "info",
            "Viterbi and smoothed posteriors use the full sample (offline).",
            {}
        ))

        n_params = _count_free_params(self.S, k, self.cov_type, self.shared_mean)
        bic_val = _bic(ll, n_params, T)

        return FitResult(
            S=self.S, cov_type=self.cov_type, shared_mean=self.shared_mean, seed=seed,
            converged=converged, n_iter=n_it, loglik=ll, bic=bic_val,
            startprob_=np.asarray(model.startprob_, dtype=float),
            transmat_=np.asarray(model.transmat_, dtype=float),
            means_=np.asarray(model.means_, dtype=float),
            covars_=np.asarray(model.covars_, dtype=float),
            gamma_=gamma,
            viterbi_path_=vpath,
            warnings=warn_list
        )

    def _empty(self, seed: int, T: int, warn_list: List[FitWarning]) -> FitResult:
        empty_gamma = np.full((T, self.S), np.nan)
        empty_vpath = np.full(T, -1, dtype=int)
        return FitResult(
            S=self.S, cov_type=self.cov_type, shared_mean=self.shared_mean, seed=seed,
            converged=False, n_iter=0, loglik=-np.inf, bic=np.inf,
            startprob_=np.array([]), transmat_=np.array([]),
            means_=np.array([]), covars_=np.array([]),
            gamma_=empty_gamma, viterbi_path_=empty_vpath,
            warnings=warn_list
        )


# ---------------------------------------------------------
# User-supplied parameters: warm-start or fixed inference
# ---------------------------------------------------------
@dataclass
class HMMParams:
    """Fully specified Gaussian-HMM parameter set.

    A single canonical container used both to *seed* EM (``fit_from_params``)
    and to run *pure inference* with the parameters held fixed
    (``infer_with_params``). Inputs are normalized and validated in
    ``__post_init__`` so callers can be loose about shapes.

    Canonical storage (after normalization)
    ---------------------------------------
    startprob_ : (S,)        rows/vector sum to 1
    transmat_  : (S, S)      each row sums to 1
    means_     : (S, k)
    covars_    : diag -> (S, k)   (per-state variances, > 0)
                 full -> (S, k, k) (per-state SPD matrices)
    cov_type   : "diag" | "full"
    shared_mean: bool   (informational; affects BIC free-param count only)

    Accepted loose inputs
    ---------------------
    - 1-D ``means_`` of shape (S,) is read as (S, 1) when k == 1.
    - ``covars_`` of shape (S,) is read as (S, 1) variances for k == 1.
    - For cov_type="diag", a (S, k, k) stack is accepted and its diagonal is
      extracted. For cov_type="full", a (S, k) input is expanded to diagonal
      matrices.
    - ``startprob_``/``transmat_`` are renormalized when ``renormalize=True``
      (default); otherwise a non-stochastic input raises.
    """

    startprob_: np.ndarray
    transmat_: np.ndarray
    means_: np.ndarray
    covars_: np.ndarray
    cov_type: CovType = "diag"
    shared_mean: bool = False
    renormalize: bool = True

    def __post_init__(self):
        if self.cov_type not in ("diag", "full"):
            raise ValueError("cov_type must be 'diag' or 'full'.")

        sp = np.asarray(self.startprob_, dtype=float).reshape(-1)
        S = sp.shape[0]
        if S < 1:
            raise ValueError("startprob_ must have at least one state.")

        tm = np.asarray(self.transmat_, dtype=float)
        if tm.shape != (S, S):
            raise ValueError(f"transmat_ must be ({S}, {S}), got {tm.shape}.")

        mu = np.asarray(self.means_, dtype=float)
        if mu.ndim == 1:
            mu = mu.reshape(S, -1) if mu.shape[0] == S else mu.reshape(-1, 1)
        if mu.shape[0] != S:
            raise ValueError(f"means_ must have {S} rows, got shape {mu.shape}.")
        k = mu.shape[1]

        cov = self._normalize_covars(np.asarray(self.covars_, dtype=float), S, k, self.cov_type)

        # Validate / renormalize the stochastic parts.
        sp = self._as_stochastic(sp, axis=0, what="startprob_")
        tm = self._as_stochastic(tm, axis=1, what="transmat_")

        # Validate covariance positivity.
        self._check_cov_positive(cov, self.cov_type)

        if not np.all(np.isfinite(mu)):
            raise ValueError("means_ contains non-finite values.")

        self.startprob_ = sp
        self.transmat_ = tm
        self.means_ = mu
        self.covars_ = cov
        self.shared_mean = bool(self.shared_mean)

    # -- normalization helpers ------------------------------------------------
    @staticmethod
    def _normalize_covars(cov: np.ndarray, S: int, k: int, cov_type: CovType) -> np.ndarray:
        if cov.ndim == 1:
            # (S,) variances -> only valid for k == 1
            if k != 1 or cov.shape[0] != S:
                raise ValueError(f"1-D covars_ of shape {cov.shape} is only valid for S={S}, k=1.")
            cov = cov.reshape(S, 1)

        if cov_type == "diag":
            if cov.ndim == 3:
                if cov.shape != (S, k, k):
                    raise ValueError(f"covars_ {cov.shape} incompatible with (S,k)=({S},{k}).")
                cov = np.stack([np.diag(cov[s]) for s in range(S)], axis=0)
            if cov.shape != (S, k):
                raise ValueError(f"diag covars_ must be ({S}, {k}), got {cov.shape}.")
            return cov

        # full
        if cov.ndim == 2:
            if cov.shape != (S, k):
                raise ValueError(f"covars_ {cov.shape} incompatible with (S,k)=({S},{k}).")
            cov = np.stack([np.diag(cov[s]) for s in range(S)], axis=0)
        if cov.shape != (S, k, k):
            raise ValueError(f"full covars_ must be ({S}, {k}, {k}), got {cov.shape}.")
        return cov

    def _as_stochastic(self, arr: np.ndarray, axis: int, what: str) -> np.ndarray:
        if np.any(arr < -1e-12):
            raise ValueError(f"{what} has negative entries.")
        arr = np.clip(arr, 0.0, None)
        sums = arr.sum(axis=axis, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError(f"{what} has a zero-sum row/vector.")
        if self.renormalize:
            return arr / sums
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError(f"{what} is not stochastic (set renormalize=True to fix).")
        return arr

    @staticmethod
    def _check_cov_positive(cov: np.ndarray, cov_type: CovType) -> None:
        if cov_type == "diag":
            if np.any(cov <= 0) or not np.all(np.isfinite(cov)):
                raise ValueError("diag covars_ must be finite and strictly positive.")
            return
        for s in range(cov.shape[0]):
            C = cov[s]
            if not np.allclose(C, C.T, atol=1e-8):
                raise ValueError(f"full covars_[{s}] is not symmetric.")
            if not np.all(np.isfinite(C)) or np.linalg.eigvalsh(C).min() <= 0:
                raise ValueError(f"full covars_[{s}] is not positive-definite.")

    # -- properties / serialization ------------------------------------------
    @property
    def S(self) -> int:
        return int(self.startprob_.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means_.shape[1])

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict in canonical form (lists, not arrays)."""
        return {
            "cov_type": self.cov_type,
            "shared_mean": bool(self.shared_mean),
            "startprob_": self.startprob_.tolist(),
            "transmat_": self.transmat_.tolist(),
            "means_": self.means_.tolist(),
            "covars_": self.covars_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HMMParams":
        return cls(
            startprob_=np.asarray(d["startprob_"], dtype=float),
            transmat_=np.asarray(d["transmat_"], dtype=float),
            means_=np.asarray(d["means_"], dtype=float),
            covars_=np.asarray(d["covars_"], dtype=float),
            cov_type=d.get("cov_type", "diag"),
            shared_mean=bool(d.get("shared_mean", False)),
        )

    @classmethod
    def from_fitresult(cls, res: FitResult) -> "HMMParams":
        return cls(
            startprob_=res.startprob_,
            transmat_=res.transmat_,
            means_=res.means_,
            covars_=res.covars_,
            cov_type=res.cov_type,
            shared_mean=res.shared_mean,
        )


def _gaussian_hmm_from_params(
    params: HMMParams,
    *,
    n_iter: int,
    tol: float,
    min_covar: float,
    train_params: str,
    random_state: int = 0,
) -> GaussianHMM:
    """Build a hmmlearn GaussianHMM with parameters injected and no auto-init.

    ``train_params`` selects which parameters EM may update ("" = none, i.e.
    held fixed; "stmc" = all). ``init_params=""`` guarantees hmmlearn never
    overwrites the values we set.
    """
    model = GaussianHMM(
        n_components=params.S,
        covariance_type=params.cov_type,
        n_iter=int(n_iter),
        tol=float(tol),
        verbose=False,
        init_params="",
        params=train_params,
        random_state=int(random_state),
        min_covar=float(min_covar),
    )
    model.startprob_ = params.startprob_
    model.transmat_ = params.transmat_
    model.means_ = params.means_
    model.covars_ = params.covars_
    return model


def _result_from_model(
    model: GaussianHMM,
    Y: np.ndarray,
    params: HMMParams,
    *,
    converged: bool,
    n_iter: int,
    extra_warnings: Optional[List[FitWarning]] = None,
) -> FitResult:
    T, k = Y.shape
    ll = float(model.score(Y))
    _, vpath = model.decode(Y, algorithm="viterbi")
    _, gamma = model.score_samples(Y)
    n_params = _count_free_params(params.S, k, params.cov_type, params.shared_mean)
    return FitResult(
        S=params.S, cov_type=params.cov_type, shared_mean=params.shared_mean, seed=-1,
        converged=bool(converged), n_iter=int(n_iter), loglik=ll, bic=_bic(ll, n_params, T),
        startprob_=np.asarray(model.startprob_, dtype=float),
        transmat_=np.asarray(model.transmat_, dtype=float),
        means_=np.asarray(model.means_, dtype=float),
        covars_=np.asarray(model.covars_, dtype=float),
        gamma_=np.asarray(gamma, dtype=float),
        viterbi_path_=np.asarray(vpath, dtype=int),
        warnings=list(extra_warnings or []),
    )


def infer_with_params(
    Y: np.ndarray,
    params: HMMParams,
    *,
    reorder: bool = False,
    order_by: OrderBy = "vol",
    ascending: bool = True,
) -> FitResult:
    """Run inference with a *fixed* parameter set (no training).

    Decodes the most-likely path, smoothed posteriors and log-likelihood under
    the supplied parameters, which are held exactly as given. Use this when the
    parameters are already estimated (e.g. loaded from storage) and you only
    want state inference on new data.
    """
    Y = _ensure_2d(Y)
    if not isinstance(params, HMMParams):
        params = HMMParams.from_dict(params) if isinstance(params, dict) else params
    if Y.shape[1] != params.n_features:
        raise ValueError(f"Y has {Y.shape[1]} features but params expect {params.n_features}.")

    # train_params="" => EM updates nothing; min_covar tiny so we don't perturb
    # the user's covariances.
    model = _gaussian_hmm_from_params(
        params, n_iter=0, tol=0.0, min_covar=1e-12, train_params="",
    )
    res = _result_from_model(
        model, Y, params, converged=True, n_iter=0,
        extra_warnings=[FitWarning("INFO_FIXED_PARAMS", "info",
                                   "Inference ran with user-supplied parameters held fixed.", {})],
    )
    return reorder_fitresult(res, by=order_by, ascending=ascending) if reorder else res


def fit_from_params(
    Y: np.ndarray,
    params: HMMParams,
    *,
    n_iter: int = 300,
    tol: float = 1e-4,
    train_params: str = "stmc",
    min_covar: float = 1e-6,
    suppress_hmmlearn_warnings: bool = True,
    reorder: bool = False,
    order_by: OrderBy = "vol",
    ascending: bool = True,
) -> FitResult:
    """Warm-start EM from a supplied parameter set and return the refined fit.

    The given ``params`` are used as the *initial* values; ``train_params``
    selects which blocks EM is allowed to update ("stmc" = all; "tmc" would
    freeze the means; "" reduces to :func:`infer_with_params`).
    """
    Y = _ensure_2d(Y)
    if not isinstance(params, HMMParams):
        params = HMMParams.from_dict(params) if isinstance(params, dict) else params
    if Y.shape[1] != params.n_features:
        raise ValueError(f"Y has {Y.shape[1]} features but params expect {params.n_features}.")

    model = _gaussian_hmm_from_params(
        params, n_iter=n_iter, tol=tol, min_covar=min_covar, train_params=train_params,
    )

    if suppress_hmmlearn_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Y)
    else:
        model.fit(Y)

    converged = bool(getattr(model.monitor_, "converged", False))
    n_it = int(getattr(model.monitor_, "iter", n_iter))
    warns: List[FitWarning] = []
    if train_params and not converged:
        warns.append(FitWarning("WARN_NO_CONVERGENCE", "warning", "Model did not converge.", {"iter": n_it}))
    res = _result_from_model(model, Y, params, converged=converged, n_iter=n_it, extra_warnings=warns)
    return reorder_fitresult(res, by=order_by, ascending=ascending) if reorder else res


# ---------------------------------------------------------
# Model Selection Logic
# ---------------------------------------------------------
def select_num_regimes(
    Y: np.ndarray,
    S_max: int = 6,
    criterion: Criterion = "bic",
    n_starts: int = 30,
    cov_type: CovType = "diag",
    shared_mean: bool = False,
    n_iter: int = 300,
    tol: float = 1e-4,
    sticky: float = 0.97,
    random_state: int = 123,
    min_occupancy_accept: float = 0.02,
    min_duration_accept: float = 2.0,
    S_min: int = 1,
) -> Dict[str, Any]:
    _validate_S_bounds(S_min, S_max)
    Y = _ensure_2d(Y)
    T, k = Y.shape

    base_rng = np.random.RandomState(random_state)

    per_S: Dict[int, Any] = {}
    best_S: Optional[int] = None
    best_score: float = np.inf

    for S in range(int(S_min), int(S_max) + 1):
        ms = MSGaussianHMM(
            S=S,
            n_starts=n_starts,
            cov_type=cov_type,
            shared_mean=shared_mean,
            n_iter=n_iter,
            tol=tol,
            sticky=sticky,
            random_state=int(base_rng.randint(0, 2**31 - 1)),
        ).fit(Y)

        best = ms.best_result_
        if best is None or not np.isfinite(best.loglik):
            per_S[S] = {"score": np.inf, "rejected": True, "best_result": best, "reason": "no_finite_loglik"}
            continue

        occ = best.gamma_.mean(axis=0)
        dur = _expected_durations(best.transmat_)

        rejected = False
        reasons = []
        if np.any(occ < min_occupancy_accept):
            rejected = True
            reasons.append({"tiny_occupancy": occ})
        if np.any(dur < min_duration_accept):
            rejected = True
            reasons.append({"tiny_duration": dur})

        n_params = _count_free_params(S, k, cov_type, shared_mean)
        score = _criterion_score(criterion, best.loglik, n_params, T)

        per_S[S] = {
            "score": float(score),
            "rejected": bool(rejected),
            "reasons": reasons,
            "best_result": best,
        }

        if (not rejected) and np.isfinite(score) and score < best_score:
            best_score = float(score)
            best_S = int(S)

    if best_S is None:
        best_S = min(per_S.keys(), key=lambda s: per_S[s]["score"])
        best_score = float(per_S[best_S]["score"])

    return {"best_S": best_S, "best_score": best_score, "criterion": criterion, "per_S": per_S}


def fit_with_auto_S(
    Y: np.ndarray,
    S_max: int = 6,
    criterion: Criterion = "bic",
    n_starts: int = 30,
    cov_type: CovType = "diag",
    shared_mean: bool = False,
    n_iter: int = 300,
    tol: float = 1e-4,
    sticky: float = 0.97,
    random_state: int = 123,
    min_occupancy_accept: float = 0.02,
    min_duration_accept: float = 2.0,
    S_min: int = 1,
) -> Dict[str, Any]:
    _validate_S_bounds(S_min, S_max)
    Y = _ensure_2d(Y)

    sel = select_num_regimes(
        Y=Y,
        S_max=S_max,
        S_min=S_min,
        criterion=criterion,
        n_starts=n_starts,
        cov_type=cov_type,
        shared_mean=shared_mean,
        n_iter=n_iter,
        tol=tol,
        sticky=sticky,
        random_state=random_state,
        min_occupancy_accept=min_occupancy_accept,
        min_duration_accept=min_duration_accept,
    )
    best_S = int(sel["best_S"])

    rng = np.random.RandomState(random_state)
    final_seed = int(rng.randint(0, 2**31 - 1))

    final_model = MSGaussianHMM(
        S=best_S,
        n_starts=n_starts,
        cov_type=cov_type,
        shared_mean=shared_mean,
        n_iter=n_iter,
        tol=tol,
        sticky=sticky,
        random_state=final_seed,
    ).fit(Y)

    final_result = final_model.best_result_
    if final_result is None:
        raise RuntimeError("Final fit failed: best_result_ is None.")

    return {
        "best_S": best_S,
        "criterion": criterion,
        "selection": sel,
        "final_model": final_model,
        "final_result": final_result,
    }


def _series_fit_autos(
    y_1d: np.ndarray,
    *,
    S_max: int,
    S_min: int = 1,
    criterion: Criterion,
    n_starts: int,
    cov_type: CovType,
    shared_mean: bool,
    n_iter: int,
    tol: float,
    sticky: float,
    random_state: int,
    min_occupancy_accept: float,
    min_duration_accept: float,
    reorder_by: Optional[OrderBy],
    reorder_ascending: bool,
) -> Dict[str, Any]:
    y = np.asarray(y_1d).reshape(-1, 1)

    out = fit_with_auto_S(
        Y=y,
        S_max=S_max,
        S_min=S_min,
        criterion=criterion,
        n_starts=n_starts,
        cov_type=cov_type,
        shared_mean=shared_mean,
        n_iter=n_iter,
        tol=tol,
        sticky=sticky,
        random_state=random_state,
        min_occupancy_accept=min_occupancy_accept,
        min_duration_accept=min_duration_accept,
    )

    res: FitResult = out["final_result"]
    if reorder_by is not None:
        res = reorder_fitresult(res, by=reorder_by, ascending=reorder_ascending)
        out["final_result"] = res

    S = int(res.S)
    if reorder_by == "vol" and reorder_ascending:
        high_state = S - 1
        labels = regime_labels_from_S(S)
    else:
        vol = _state_vol_measure(res.covars_, res.cov_type)
        high_state = int(np.argmax(vol))
        labels = [f"State {i}" for i in range(S)]

    state = res.viterbi_path_.astype(int)
    highvol = (state == high_state).astype(int)

    gamma = np.asarray(res.gamma_, dtype=float)
    prob_hv = gamma[:, high_state]

    return {
        "selection": out["selection"],
        "best_S": out["best_S"],
        "criterion": out["criterion"],
        "final_model": out["final_model"],
        "final_result": res,
        "state": state,
        "highvol": highvol,
        "prob_hv": prob_hv,
        "gamma": gamma,
        "high_state": high_state,
        "labels": labels,
    }


def fit_autos_Y(
    Y: np.ndarray,
    *,
    fit_mode: _FitMode = "joint",
    col_names: Optional[Sequence[str]] = None,
    S_max: int = 5,
    S_min: int = 1,
    criterion: Criterion = "bic",
    n_starts: int = 30,
    cov_type: CovType = "diag",
    shared_mean: bool = False,
    n_iter: int = 200,
    tol: float = 1e-4,
    sticky: float = 0.95,
    random_state: int = 123,
    min_occupancy_accept: float = 0.02,
    min_duration_accept: float = 2.0,
    reorder_by: Optional[OrderBy] = "vol",
    reorder_ascending: bool = True,
) -> Dict[str, Any]:
    _validate_S_bounds(S_min, S_max)
    Y = _ensure_2d(Y)
    _, k = Y.shape

    if col_names is None:
        col_names = [f"Y{j}" for j in range(k)]
    else:
        col_names = list(col_names)
        if len(col_names) != k:
            raise ValueError("col_names length must match number of columns in Y.")

    rng = np.random.RandomState(random_state)

    if fit_mode == "joint":
        out = fit_with_auto_S(
            Y=Y,
            S_max=S_max,
            S_min=S_min,
            criterion=criterion,
            n_starts=n_starts,
            cov_type=cov_type,
            shared_mean=shared_mean,
            n_iter=n_iter,
            tol=tol,
            sticky=sticky,
            random_state=int(rng.randint(0, 2**31 - 1)),
            min_occupancy_accept=min_occupancy_accept,
            min_duration_accept=min_duration_accept,
        )

        res: FitResult = out["final_result"]
        if reorder_by is not None:
            res = reorder_fitresult(res, by=reorder_by, ascending=reorder_ascending)
            out["final_result"] = res

        S = int(res.S)
        if reorder_by == "vol" and reorder_ascending:
            high_state = S - 1
            labels = regime_labels_from_S(S)
        else:
            vol = _state_vol_measure(res.covars_, res.cov_type)
            high_state = int(np.argmax(vol))
            labels = [f"State {i}" for i in range(S)]

        state = res.viterbi_path_.astype(int)
        highvol = (state == high_state).astype(int)
        prob_hv = res.gamma_[:, high_state]

        gamma = np.asarray(res.gamma_, dtype=float)

        return {
            "fit_mode": fit_mode,
            "col_names": col_names,
            "results": {
                "selection": out["selection"],
                "best_S": out["best_S"],
                "criterion": out["criterion"],
                "final_model": out["final_model"],
                "final_result": res,
                "state": state,
                "highvol": highvol,
                "prob_hv": prob_hv,
                "gamma": gamma,
                "high_state": high_state,
                "labels": labels,
            },
        }

    if fit_mode == "panel":
        results: Dict[str, Any] = {}
        for j, nm in enumerate(col_names):
            results[nm] = _series_fit_autos(
                Y[:, j],
                S_max=S_max,
                S_min=S_min,
                criterion=criterion,
                n_starts=n_starts,
                cov_type="diag",
                shared_mean=shared_mean,
                n_iter=n_iter,
                tol=tol,
                sticky=sticky,
                random_state=int(rng.randint(0, 2**31 - 1)),
                min_occupancy_accept=min_occupancy_accept,
                min_duration_accept=min_duration_accept,
                reorder_by=reorder_by,
                reorder_ascending=reorder_ascending,
            )
        return {"fit_mode": fit_mode, "col_names": col_names, "results": results}

    raise ValueError("fit_mode must be 'joint' or 'panel'.")


# ---------------------------------------------------------
# MSRegimeEngine (Unified Interface)
# ---------------------------------------------------------
class MSRegimeEngine:
    """
    Engine unico: prende sempre un DataFrame e un mode.
    """
    def __init__(
        self,
        *,
        S_max: int = 5,
        S_min: int = 1,
        criterion: Literal["bic", "aic", "hqic"] = "bic",
        n_starts: int = 30,
        shared_mean: bool = False,
        n_iter: int = 200,
        tol: float = 1e-4,
        sticky: float = 0.95,
        min_occupancy_accept: float = 0.02,
        min_duration_accept: float = 2.0,
        reorder_by: Optional[OrderBy] = "vol",
        reorder_ascending: bool = True,
        random_state: int = 123,
        standardize: bool = True,
    ):
        _validate_S_bounds(S_min, S_max)
        self.S_max = int(S_max)
        self.S_min = int(S_min)
        self.criterion = criterion
        self.n_starts = int(n_starts)
        self.shared_mean = bool(shared_mean)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.sticky = float(sticky)
        self.min_occupancy_accept = float(min_occupancy_accept)
        self.min_duration_accept = float(min_duration_accept)
        self.reorder_by = reorder_by
        self.reorder_ascending = bool(reorder_ascending)
        self.random_state = int(random_state)
        self.standardize = bool(standardize)

    def fit(
        self,
        df: Any,  # pandas.DataFrame
        *,
        model: ModelKind = "panel",
        dropna: Literal["any", "all"] = "all",
    ) -> RegimeRun:
        # Single-axis model schema. 'categorical' is rejected here (raises) and
        # routed to fit_categorical_regimes(); the others map to (fit_mode, cov_type).
        fit_mode, cov_type = _resolve_model(model)

        if dropna == "all":
            df = df.dropna(how="all")
        elif dropna == "any":
            df = df.dropna(how="any")
        else:
            raise ValueError("dropna must be 'any' or 'all'.")

        cols = list(df.columns)
        if len(cols) == 0:
            raise ValueError("Empty DataFrame after dropna.")

        rng = np.random.RandomState(self.random_state)

        # Panel mode: standardize each column (zero mean, unit variance)
        # before fitting. hmmlearn's EM is numerically sensitive to Y's
        # absolute scale -- jitter/min_covar/var_floor are all absolute-scale
        # hyperparameters tuned for roughly unit variance, not the ~1e-4
        # variance typical of raw daily log returns -- which is the likely
        # cause of "Model is not converging" warnings on some symbols.
        # State assignments/probabilities are exactly invariant to this
        # affine rescaling (a Gaussian posterior's argmax and relative
        # likelihoods are unchanged under an affine reparametrization), so
        # fit in standardized units and de-standardize means_/covars_ (the
        # only scale-dependent outputs) back to the original units below,
        # per column, before they reach ``meta``. Joint/joint_full mode is
        # deliberately left untouched -- de-standardizing a "full"
        # covariance across differently-scaled columns needs a matrix
        # (not scalar) transform, out of scope here.
        standardize = self.standardize and fit_mode == "panel"
        if standardize:
            col_means = df.values.mean(axis=0)
            col_stds = df.values.std(axis=0, ddof=1)
            col_stds = np.where(np.isfinite(col_stds) & (col_stds > 0), col_stds, 1.0)
            Y_fit = (df.values - col_means) / col_stds
        else:
            Y_fit = df.values

        out = fit_autos_Y(
            Y=Y_fit,
            fit_mode=fit_mode,
            col_names=cols,
            S_max=self.S_max,
            S_min=self.S_min,
            criterion=self.criterion,
            n_starts=self.n_starts,
            cov_type=cov_type,
            shared_mean=self.shared_mean,
            n_iter=self.n_iter,
            tol=self.tol,
            sticky=self.sticky,
            random_state=int(rng.randint(0, 2**31 - 1)),
            min_occupancy_accept=self.min_occupancy_accept,
            min_duration_accept=self.min_duration_accept,
            reorder_by=self.reorder_by,
            reorder_ascending=self.reorder_ascending,
        )

        labels_map: Dict[str, List[str]] = {}
        meta: Dict[str, Any] = {"_MODEL_": model}
        panel_parts = []

        idx = df.index

        if fit_mode == "panel":
            for j, col in enumerate(cols):
                r = out["results"][col]
                res = r["final_result"]
                S = int(res.S)

                # De-standardize means_/covars_ back to the column's own
                # original units -- everything else (bic/loglik/startprob_/
                # transmat_/state assignments/probabilities) is unaffected
                # by the standardization (see the comment above where Y_fit
                # is built), so only these two need the inverse transform.
                if standardize:
                    mean_j, std_j = col_means[j], col_stds[j]
                    means_out = (res.means_ * std_j + mean_j).tolist()
                    covars_out = (res.covars_ * (std_j ** 2)).tolist()
                else:
                    means_out = res.means_.tolist()
                    covars_out = res.covars_.tolist()

                value = pd.Series(df[col].values, index=idx, name=f"{col}_value")
                state = pd.Series(r["state"], index=idx, name=f"{col}_state")
                highvol = pd.Series(r["highvol"], index=idx, name=f"{col}_highvol")
                prob = pd.Series(r["prob_hv"], index=idx, name=f"P_{col}_HV")

                gamma = np.asarray(r["gamma"], dtype=float)
                prob_cols = [f"P_{col}_S{s}" for s in range(S)]
                prob_df = pd.DataFrame(gamma, index=idx, columns=prob_cols)

                panel_parts.append(pd.concat([value, state, highvol, prob, prob_df], axis=1))

                labels_map[col] = r["labels"]
                meta[col] = {
                    "S": int(res.S),
                    "labels": r["labels"],
                    "bic": float(res.bic),
                    "loglik": float(res.loglik),
                    "startprob_": res.startprob_.tolist(),
                    "transmat_": res.transmat_.tolist(),
                    "means_": means_out,
                    "covars_": covars_out,
                }

        else:  # joint (multivariate)
            r = out["results"]
            res = r["final_result"]
            S = int(res.S)

            common_state = pd.Series(r["state"], index=idx)
            common_high = pd.Series(r["highvol"], index=idx)
            common_prob = pd.Series(r["prob_hv"], index=idx)
            gamma = np.asarray(r["gamma"], dtype=float)

            for col in cols:
                value = pd.Series(df[col].values, index=idx, name=f"{col}_value")
                state = common_state.rename(f"{col}_state")
                highvol = common_high.rename(f"{col}_highvol")
                prob = common_prob.rename(f"P_{col}_HV")

                prob_cols = [f"P_{col}_S{s}" for s in range(S)]
                prob_df = pd.DataFrame(gamma, index=idx, columns=prob_cols)

                panel_parts.append(pd.concat([value, state, highvol, prob, prob_df], axis=1))

                labels_map[col] = r["labels"]
                meta[col] = {
                    "S": int(res.S),
                    "labels": r["labels"],
                    "bic": float(res.bic),
                    "loglik": float(res.loglik),
                    "shared_multivariate_state": True,
                    "startprob_": res.startprob_.tolist(),
                    "transmat_": res.transmat_.tolist(),
                    "means_": res.means_.tolist(),
                    "covars_": res.covars_.tolist(),
                }

        panel = pd.concat(panel_parts, axis=1).sort_index()
        return RegimeRun(panel=panel, rows=cols, labels_map=labels_map, meta=meta)


# Canonical alias: the tool layer (lazystats.regimes.tools) and newer code use RegimeEngine.
RegimeEngine = MSRegimeEngine

# =============================================================================
# MAIN EXECUTION BLOCK (Examples)
# =============================================================================
# The demo __main__ examples of the original lazyhmm module (including the
# yfinance-based financial example) were dropped in the lazystats migration:
# direct provider downloads are forbidden here (plan v3.1 -- the hub is the
# only downloader). See lazystats.regimes.datasources.load_from_datahub.

# ============================================================================
# DISCRETE (CATEGORICAL) HMM / MARKOV REGIME SWITCHING
# ============================================================================
# This section adds support for regime models whose observations are categorical
# labels (e.g., per-group regimes 0/1/2...) rather than continuous returns.
#
# Core idea:
#   - Observation at time t is an integer symbol y_t in {0,...,K-1}
#     (or a vector of categorical labels that we internally encode into a symbol).
#   - Hidden state s_t in {0,...,S-1} follows a Markov chain.
#   - Emission P(y_t | s_t) is categorical with parameters emissionprob_[s, k].
#
# The API mirrors the continuous pipeline where possible:
#   - fit_with_auto_S_categorical(...) returns dict with best_S, all_results, final_result.
#   - The returned result has: startprob_, transmat_, emissionprob_, gamma_, viterbi_path_,
#     loglik, bic, converged, n_iter, plus symbol encoding metadata.

from dataclasses import dataclass as _dataclass

@_dataclass
class DiscreteFitResult:
    S: int
    K: int
    seed: int

    converged: bool
    n_iter: int
    loglik: float
    bic: float

    startprob_: np.ndarray        # (S,)
    transmat_: np.ndarray         # (S,S)
    emissionprob_: np.ndarray     # (S,K)

    gamma_: np.ndarray            # (T,S)
    viterbi_path_: np.ndarray     # (T,)

    # If the user provided multivariate categorical observations (T,D),
    # we encode each row to a symbol in {0..K-1}. We keep the mapping.
    symbol_map_: Optional[Dict[int, Tuple[int, ...]]] = None  # symbol -> tuple labels
    symbol_invmap_: Optional[Dict[Tuple[int, ...], int]] = None  # tuple labels -> symbol


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _encode_categorical_obs(Y: np.ndarray) -> Tuple[np.ndarray, int, Optional[Dict[int, Tuple[int, ...]]], Optional[Dict[Tuple[int, ...], int]]]:
    """Encode categorical observations to integer symbols.

    Accepts:
      - Y shape (T,) already integer-like categories
      - Y shape (T, D) integer-like categories (e.g., regimes per subgroup)

    Returns:
      y: (T,) int symbols
      K: number of unique symbols
      symbol_map: symbol -> tuple (for multivariate); or None for univariate
      symbol_invmap: inverse mapping or None
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        y = Y.astype(int)
        uniq = np.unique(y)
        inv = {int(v): i for i, v in enumerate(uniq)}
        y_enc = np.array([inv[int(v)] for v in y], dtype=int)
        symbol_map = {i: (int(v),) for i, v in enumerate(uniq)}
        symbol_invmap = {(int(v),): i for i, v in enumerate(uniq)}
        return y_enc, len(uniq), symbol_map, symbol_invmap

    if Y.ndim == 2:
        T, D = Y.shape
        Y_int = np.ascontiguousarray(Y.astype(np.int32))
        # Fast unique-row encoding via structured view
        row_dt = np.dtype([('', np.int32)] * D)
        row_view = Y_int.view(row_dt).ravel()          # (T,) structured
        uniq_rows, inverse = np.unique(row_view, return_inverse=True)
        y_enc = inverse.astype(int)
        K = len(uniq_rows)
        # Build maps — iterate only K unique rows, not T
        symbol_map: Dict[int, Tuple[int, ...]] = {}
        symbol_invmap: Dict[Tuple[int, ...], int] = {}
        for k_idx in range(K):
            row_tuple = tuple(int(v) for v in uniq_rows[k_idx])
            symbol_map[k_idx] = row_tuple
            symbol_invmap[row_tuple] = k_idx
        return y_enc, K, symbol_map, symbol_invmap

    raise ValueError("Y must be 1D or 2D categorical array")


def _normalize_rows(mat: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    mat = np.maximum(mat, eps)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def _discrete_forward_backward(
    y: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Forward-backward in log space (vectorized).

    Returns:
      loglik
      gamma (T,S)
      xi_sum (S,S) expected transition counts summed over t
    """
    T = y.shape[0]
    S = log_start.shape[0]

    # Precompute per-timestep emission: (T, S)
    log_obs = log_emit[:, y].T  # log_emit is (S,K), index cols by y → (S,T) → transpose

    # --- Forward ---
    log_alpha = np.empty((T, S))
    log_alpha[0] = log_start + log_obs[0]
    for t in range(1, T):
        tmp = log_alpha[t - 1][:, None] + log_trans   # (S, S)
        a_max = tmp.max(axis=0)                        # (S,)
        log_alpha[t] = log_obs[t] + a_max + np.log(np.exp(tmp - a_max).sum(axis=0))

    a_max_final = log_alpha[T - 1].max()
    loglik = float(a_max_final + np.log(np.exp(log_alpha[T - 1] - a_max_final).sum()))

    # --- Backward ---
    log_beta = np.empty((T, S))
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        tmp = log_trans + (log_obs[t + 1] + log_beta[t + 1])[None, :]  # (S, S)
        a_max = tmp.max(axis=1)                                         # (S,)
        log_beta[t] = a_max + np.log(np.exp(tmp - a_max[:, None]).sum(axis=1))

    # --- Gamma ---
    log_gamma = log_alpha + log_beta
    lg_max = log_gamma.max(axis=1, keepdims=True)
    log_gamma_n = log_gamma - lg_max - np.log(np.exp(log_gamma - lg_max).sum(axis=1, keepdims=True))
    gamma = np.exp(log_gamma_n)

    # --- Xi sum (fully vectorized) ---
    #   log_xi[t, i, j] = log_alpha[t,i] + log_trans[i,j]
    #                    + log_obs[t+1,j] + log_beta[t+1,j]
    la = log_alpha[:-1, :, None]                            # (T-1, S, 1)
    lt = log_trans[None, :, :]                              # (1, S, S)
    lb_obs = (log_obs[1:] + log_beta[1:])[:, None, :]      # (T-1, 1, S)
    log_xi_all = la + lt + lb_obs                           # (T-1, S, S)

    # Normalize per timestep then sum
    lx_flat = log_xi_all.reshape(T - 1, -1)                # (T-1, S*S)
    lx_max = lx_flat.max(axis=1)                            # (T-1,)
    log_norm = lx_max + np.log(np.exp(lx_flat - lx_max[:, None]).sum(axis=1))
    xi_sum = np.exp(log_xi_all - log_norm[:, None, None]).sum(axis=0)

    return loglik, gamma, xi_sum


def _discrete_viterbi(
    y: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> np.ndarray:
    T = y.shape[0]
    S = log_start.shape[0]

    # Precompute per-timestep emission
    log_obs = log_emit[:, y].T  # (T, S)

    delta = np.empty((T, S))
    psi = np.empty((T, S), dtype=int)

    delta[0] = log_start + log_obs[0]
    psi[0] = 0

    for t in range(1, T):
        tmp = delta[t - 1][:, None] + log_trans  # (S, S)
        psi[t] = np.argmax(tmp, axis=0)
        delta[t] = log_obs[t] + np.max(tmp, axis=0)

    path = np.empty(T, dtype=int)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def fit_discrete_hmm(
    Y: np.ndarray,
    S: int,
    n_iter: int = 200,
    tol: float = 1e-4,
    sticky: float = 0.0,
    random_state: int = 0,
    laplace: float = 1e-2,
    progress_callback: Optional[Any] = None,
) -> DiscreteFitResult:
    """Fit a discrete (categorical emission) HMM via EM.

    Parameters
    ----------
    Y : array
        Categorical observations. Shape (T,) or (T,D). Values must be integer-like.
    S : int
        Number of hidden regimes.
    sticky : float
        In [0,1). Mix transition matrix toward identity to encourage persistence.
        Implementation: trans = (1-sticky)*trans + sticky*I.
    laplace : float
        Additive smoothing for emission/transition estimates to avoid zeros.
    progress_callback : callable, optional
        Called as progress_callback(iter, loglik, converged) after each EM iteration.
    """
    y, K, sym_map, sym_inv = _encode_categorical_obs(Y)
    T = y.shape[0]

    rng = np.random.default_rng(int(random_state))

    # init params
    startprob = rng.dirichlet(np.ones(S))
    transmat = rng.dirichlet(np.ones(S), size=S)
    emissionprob = rng.dirichlet(np.ones(K), size=S)

    prev_ll = -np.inf
    converged = False

    for it in range(1, n_iter + 1):
        log_start = np.log(np.maximum(startprob, 1e-300))
        log_trans = np.log(np.maximum(transmat, 1e-300))
        log_emit = np.log(np.maximum(emissionprob, 1e-300))

        ll, gamma, xi_sum = _discrete_forward_backward(y, log_start, log_trans, log_emit)

        # M-step
        startprob = gamma[0].copy()

        trans_counts = xi_sum + laplace
        transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        if sticky and sticky > 0:
            transmat = (1.0 - sticky) * transmat + sticky * np.eye(S)
            transmat = _normalize_rows(transmat)

        emit_counts = np.full((S, K), laplace, dtype=float)
        np.add.at(emit_counts, (slice(None), y), gamma.T)
        emissionprob = emit_counts / emit_counts.sum(axis=1, keepdims=True)

        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
            converged = True
            prev_ll = ll
            if progress_callback is not None:
                progress_callback(it, ll, True)
            break
        prev_ll = ll
        if progress_callback is not None:
            progress_callback(it, ll, False)

    # final decode
    log_start = np.log(np.maximum(startprob, 1e-300))
    log_trans = np.log(np.maximum(transmat, 1e-300))
    log_emit = np.log(np.maximum(emissionprob, 1e-300))
    vpath = _discrete_viterbi(y, log_start, log_trans, log_emit)

    # BIC
    # free parameters: (S-1) start + S*(S-1) transitions + S*(K-1) emissions
    k_params = (S - 1) + S * (S - 1) + S * (K - 1)
    bic = -2.0 * prev_ll + k_params * np.log(max(T, 1))

    return DiscreteFitResult(
        S=S,
        K=K,
        seed=int(random_state),
        converged=converged,
        n_iter=it,
        loglik=float(prev_ll),
        bic=float(bic),
        startprob_=startprob,
        transmat_=transmat,
        emissionprob_=emissionprob,
        gamma_=gamma,
        viterbi_path_=vpath,
        symbol_map_=sym_map,
        symbol_invmap_=sym_inv,
    )


def _auto_S_restart_loop(
    *,
    S_max: int,
    S_min: int = 1,
    n_starts: int,
    random_state: int,
    fit_one: Any,                       # callable(S:int, seed:int) -> result (.bic/.loglik/.seed)
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generic BIC model-order selection for the pure-numpy discrete engines.

    Loops S = S_min..S_max, runs ``n_starts`` random restarts per S keeping the best
    by BIC, and assembles the documented ``{best_S, all_results, final_result}``
    shape. Shared by fit_with_auto_S_categorical and fit_with_auto_S_multivar so
    the restart/selection bookkeeping lives in exactly one place.
    """
    rng = np.random.default_rng(int(random_state))
    all_results: List[Dict[str, Any]] = []

    best_overall = None
    best_S = None

    for S in range(int(S_min), int(S_max) + 1):
        best_res_S = None
        best_bic_S = np.inf

        for start_i in range(int(n_starts)):
            seed = int(rng.integers(0, 2**31 - 1))
            res = fit_one(S, seed)
            if res.bic < best_bic_S:
                best_bic_S = res.bic
                best_res_S = res

            if progress_callback is not None:
                progress_callback(S, start_i + 1, n_starts, best_bic_S)

        all_results.append(
            {
                "S": S,
                "best_bic": float(best_res_S.bic),
                "best_loglik": float(best_res_S.loglik),
                "best_seed": int(best_res_S.seed),
                "result": best_res_S,
            }
        )

        if best_overall is None or best_res_S.bic < best_overall.bic:
            best_overall = best_res_S
            best_S = S

    return {
        "best_S": int(best_S),
        "all_results": all_results,
        "final_result": best_overall,
    }


def fit_with_auto_S_categorical(
    Y: np.ndarray,
    S_max: int = 4,
    criterion: str = "bic",
    n_starts: int = 10,
    n_iter: int = 200,
    tol: float = 1e-4,
    sticky: float = 0.0,
    random_state: int = 0,
    laplace: float = 1e-2,
    progress_callback: Optional[Any] = None,
    S_min: int = 1,
) -> Dict[str, Any]:
    """Auto-select S for categorical observations (discrete HMM).

    Parameters
    ----------
    progress_callback : callable, optional
        Called as progress_callback(S, start_idx, n_starts, best_bic_so_far) after
        each random restart completes.

    Returns dict:
      - best_S
      - all_results: list of dicts {S, best_bic, best_loglik, best_seed, result}
      - final_result: DiscreteFitResult for best_S
    """
    _validate_S_bounds(S_min, S_max)
    crit = criterion.lower().strip()
    if crit not in {"bic"}:
        raise ValueError("Only BIC is supported for categorical auto-selection")

    def fit_one(S: int, seed: int) -> DiscreteFitResult:
        return fit_discrete_hmm(
            Y=Y, S=S, n_iter=n_iter, tol=tol,
            sticky=sticky, random_state=seed, laplace=laplace,
        )

    return _auto_S_restart_loop(
        S_max=S_max, S_min=S_min, n_starts=n_starts, random_state=random_state,
        fit_one=fit_one, progress_callback=progress_callback,
    )


# ============================================================================
# MULTIVARIATE INDEPENDENT-EMISSION HMM
# ============================================================================
# Each observation y_t = (y_t1, ..., y_tD) with y_td in {0, ..., C_d - 1}.
# Given hidden state s, emissions are conditionally independent:
#     P(y_t | s) = prod_{d=1}^{D} theta[s, d, y_td]
#
# Parameters: S*sum(C_d), vastly smaller than S*prod(C_d) of full categorical.
# Special case: all C_d = 2 → multivariate Bernoulli HMM.


@_dataclass
class MultiVarFitResult:
    """Result of fitting a multivariate independent-emission HMM."""
    S: int
    D: int
    C: np.ndarray           # (D,) int — categories per dimension

    seed: int
    converged: bool
    n_iter: int
    loglik: float
    bic: float

    startprob_: np.ndarray          # (S,)
    transmat_: np.ndarray           # (S, S)
    emission_theta_: np.ndarray     # (S, D, max_C) — padded with 0 where c >= C_d

    gamma_: np.ndarray              # (T, S)
    viterbi_path_: np.ndarray       # (T,)


def _multivar_log_emission(
    Y: np.ndarray,
    theta: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Compute log P(y_t | s) for all t, s.  Vectorized.

    Parameters
    ----------
    Y : (T, D) int array of observations
    theta : (S, D, max_C) emission probabilities
    C : (D,) int — number of categories per dimension

    Returns
    -------
    log_obs : (T, S) log-emission for each timestep and state
    """
    T, D = Y.shape
    S = theta.shape[0]
    log_theta = np.log(np.maximum(theta, 1e-300))

    # For each dimension d, gather log_theta[s, d, Y[t, d]] for all t, s
    # Use advanced indexing: log_theta[:, d, Y[:, d]] gives (S, T)
    log_obs = np.zeros((T, S), dtype=np.float64)
    for d in range(D):
        # log_theta[:, d, :] is (S, max_C); Y[:, d] is (T,)
        log_obs += log_theta[:, d, Y[:, d]].T  # (S, T).T = (T, S)

    return log_obs


def _multivar_forward_backward(
    log_obs: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Forward-backward for precomputed log-emission matrix.

    Parameters
    ----------
    log_obs : (T, S)
    log_start : (S,)
    log_trans : (S, S)

    Returns
    -------
    loglik, gamma (T, S), xi_sum (S, S)
    """
    T, S = log_obs.shape

    # Forward
    log_alpha = np.empty((T, S))
    log_alpha[0] = log_start + log_obs[0]
    for t in range(1, T):
        tmp = log_alpha[t - 1][:, None] + log_trans
        a_max = tmp.max(axis=0)
        log_alpha[t] = log_obs[t] + a_max + np.log(np.exp(tmp - a_max).sum(axis=0))

    a_max_final = log_alpha[T - 1].max()
    loglik = float(a_max_final + np.log(np.exp(log_alpha[T - 1] - a_max_final).sum()))

    # Backward
    log_beta = np.empty((T, S))
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        tmp = log_trans + (log_obs[t + 1] + log_beta[t + 1])[None, :]
        a_max = tmp.max(axis=1)
        log_beta[t] = a_max + np.log(np.exp(tmp - a_max[:, None]).sum(axis=1))

    # Gamma
    log_gamma = log_alpha + log_beta
    lg_max = log_gamma.max(axis=1, keepdims=True)
    log_gamma_n = log_gamma - lg_max - np.log(
        np.exp(log_gamma - lg_max).sum(axis=1, keepdims=True)
    )
    gamma = np.exp(log_gamma_n)

    # Xi sum (vectorized)
    la = log_alpha[:-1, :, None]
    lt = log_trans[None, :, :]
    lb_obs = (log_obs[1:] + log_beta[1:])[:, None, :]
    log_xi_all = la + lt + lb_obs

    lx_flat = log_xi_all.reshape(T - 1, -1)
    lx_max = lx_flat.max(axis=1)
    log_norm = lx_max + np.log(np.exp(lx_flat - lx_max[:, None]).sum(axis=1))
    xi_sum = np.exp(log_xi_all - log_norm[:, None, None]).sum(axis=0)

    return loglik, gamma, xi_sum


def _multivar_viterbi(
    log_obs: np.ndarray,
    log_start: np.ndarray,
    log_trans: np.ndarray,
) -> np.ndarray:
    """Viterbi decode for precomputed log-emission."""
    T, S = log_obs.shape
    delta = np.empty((T, S))
    psi = np.empty((T, S), dtype=int)

    delta[0] = log_start + log_obs[0]
    psi[0] = 0

    for t in range(1, T):
        tmp = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(tmp, axis=0)
        delta[t] = log_obs[t] + np.max(tmp, axis=0)

    path = np.empty(T, dtype=int)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def fit_multivar_hmm(
    Y: np.ndarray,
    S: int,
    n_iter: int = 200,
    tol: float = 1e-4,
    sticky: float = 0.0,
    random_state: int = 0,
    laplace: float = 1e-2,
    progress_callback: Optional[Any] = None,
) -> MultiVarFitResult:
    """Fit a multivariate independent-emission HMM via EM.

    Parameters
    ----------
    Y : (T, D) int array — each column is a categorical variable
    S : int — number of hidden states
    sticky : float in [0, 1) — encourage self-transitions
    laplace : float — additive smoothing
    progress_callback : callable(iter, loglik, converged), optional
    """
    Y = np.asarray(Y, dtype=int)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    T, D = Y.shape

    # Determine categories per dimension
    C = np.array([int(Y[:, d].max()) + 1 for d in range(D)], dtype=int)
    max_C = int(C.max())

    rng = np.random.default_rng(int(random_state))

    # Init
    startprob = rng.dirichlet(np.ones(S))
    transmat = rng.dirichlet(np.ones(S), size=S)

    # theta[s, d, c] = P(y_d = c | state = s)
    theta = np.empty((S, D, max_C), dtype=np.float64)
    for d in range(D):
        theta[:, d, :C[d]] = rng.dirichlet(np.ones(C[d]), size=S)
        if C[d] < max_C:
            theta[:, d, C[d]:] = 0.0

    prev_ll = -np.inf
    converged = False

    for it in range(1, n_iter + 1):
        log_start = np.log(np.maximum(startprob, 1e-300))
        log_trans = np.log(np.maximum(transmat, 1e-300))

        log_obs = _multivar_log_emission(Y, theta, C)
        ll, gamma, xi_sum = _multivar_forward_backward(log_obs, log_start, log_trans)

        # M-step: startprob
        startprob = gamma[0].copy()

        # M-step: transmat
        trans_counts = xi_sum + laplace
        transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        if sticky and sticky > 0:
            transmat = (1.0 - sticky) * transmat + sticky * np.eye(S)
            transmat = transmat / transmat.sum(axis=1, keepdims=True)

        # M-step: emission theta — vectorized per dimension
        for d in range(D):
            Cd = C[d]
            counts = np.full((S, Cd), laplace, dtype=np.float64)
            # gamma is (T, S); Y[:, d] is (T,)
            np.add.at(counts, (slice(None), Y[:, d]), gamma.T)
            theta[:, d, :Cd] = counts / counts.sum(axis=1, keepdims=True)

        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
            converged = True
            prev_ll = ll
            if progress_callback is not None:
                progress_callback(it, ll, True)
            break
        prev_ll = ll
        if progress_callback is not None:
            progress_callback(it, ll, False)

    # Final decode
    log_start = np.log(np.maximum(startprob, 1e-300))
    log_trans = np.log(np.maximum(transmat, 1e-300))
    log_obs = _multivar_log_emission(Y, theta, C)
    vpath = _multivar_viterbi(log_obs, log_start, log_trans)

    # BIC: free params = (S-1) + S*(S-1) + S * sum(C_d - 1)
    n_params = (S - 1) + S * (S - 1) + S * int(np.sum(C - 1))
    bic = -2.0 * prev_ll + n_params * np.log(max(T, 1))

    return MultiVarFitResult(
        S=S, D=D, C=C,
        seed=int(random_state),
        converged=converged,
        n_iter=it,
        loglik=float(prev_ll),
        bic=float(bic),
        startprob_=startprob,
        transmat_=transmat,
        emission_theta_=theta,
        gamma_=gamma,
        viterbi_path_=vpath,
    )


def fit_with_auto_S_multivar(
    Y: np.ndarray,
    S_max: int = 4,
    criterion: str = "bic",
    n_starts: int = 10,
    n_iter: int = 200,
    tol: float = 1e-4,
    sticky: float = 0.0,
    random_state: int = 0,
    laplace: float = 1e-2,
    progress_callback: Optional[Any] = None,
    S_min: int = 1,
) -> Dict[str, Any]:
    """Auto-select S for multivariate independent-emission HMM.

    Parameters
    ----------
    progress_callback : callable(S, start_idx, n_starts, best_bic), optional

    Returns
    -------
    dict with best_S, all_results, final_result (MultiVarFitResult)
    """
    _validate_S_bounds(S_min, S_max)
    crit = criterion.lower().strip()
    if crit not in {"bic"}:
        raise ValueError("Only BIC is supported for auto-selection")

    def fit_one(S: int, seed: int) -> MultiVarFitResult:
        return fit_multivar_hmm(
            Y=Y, S=S, n_iter=n_iter, tol=tol,
            sticky=sticky, random_state=seed, laplace=laplace,
        )

    return _auto_S_restart_loop(
        S_max=S_max, S_min=S_min, n_starts=n_starts, random_state=random_state,
        fit_one=fit_one, progress_callback=progress_callback,
    )