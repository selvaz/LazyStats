"""Configuration contract for the regime estimation runs.

Separates the *method* — fitting a regime model and comparing fits — from the
*preset*: which instruments, over which windows, with which fitting parameters.
Those are project choices, not statistics.

The universe used to come from ``tickers.yaml`` inside the market data
repository, selected by a ``--priority`` tier. That put one project's instrument
list inside a general-purpose package, and made "which symbols does the daily
regime job cover?" a question you answered by reading another repository's
configuration. It is stated here instead, like every other preset.

**Windows are named, and the comparison is between two of them.** The previous
design could only contrast the whole available history against one shorter
window, with the shorter one hardcoded as "8y" at the call site. The method has
no opinion about eight years: it compares regimes estimated over *different
windows*, and which windows those are is a project's choice. A window with no
``lookback_years`` is the whole available history; naming it makes it something
a comparison can refer to like any other.

There is deliberately **no default preset**. ``load_config`` requires an
explicit path, for the same reason the other runners do: a scheduled job quietly
fitting the wrong universe is worse than one that refuses to start.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The window that imposes no restriction. Its estimates carry no variant tag,
#: which is what keeps them in the same series the migrated history lives in.
FULL_HISTORY = "full"


@dataclass(frozen=True)
class Window:
    """One estimation window.

    Attributes:
        name: How comparisons refer to it.
        lookback_years: Years of history to fit over, or ``None`` for all of it.
        variant: The tag that namespaces this window's series in the depot.
            ``None`` for the full history, so those estimates stay in the
            unqualified series rather than starting a parallel one.
    """

    name: str
    lookback_years: int | None

    @property
    def variant(self) -> str | None:
        return None if self.lookback_years is None else f"{self.lookback_years}y"


@dataclass(frozen=True)
class Comparison:
    """Two windows to contrast, and what to call the result."""

    name: str
    baseline: str
    candidate: str


@dataclass(frozen=True)
class RegimeConfig:
    """Everything a regime run needs that is a project choice, not method."""

    instruments: tuple[str, ...]
    windows: tuple[Window, ...]
    comparisons: tuple[Comparison, ...]
    s_max: int
    n_starts: int
    random_state: int
    retro_days: int

    def window(self, name: str) -> Window:
        for w in self.windows:
            if w.name == name:
                return w
        raise KeyError(name)

    def as_provenance(self) -> dict[str, object]:
        """The parameter block recorded alongside a result."""
        return {
            "s_max": self.s_max,
            "n_starts": self.n_starts,
            "random_state": self.random_state,
            "retro_days": self.retro_days,
            "windows": {w.name: w.lookback_years for w in self.windows},
        }


class ConfigError(ValueError):
    """The configuration is missing, malformed or incomplete.

    Its own type so a caller can tell "you configured this wrong" apart from a
    failure inside the estimation.
    """


def _require(raw: dict[str, Any], key: str, kind: type, path: Path) -> Any:
    if key not in raw:
        raise ConfigError(f"{path.name}: missing required key '{key}'")
    value = raw[key]
    # bool is a subclass of int; n_starts of `true` must not pass as 1.
    if isinstance(value, bool) or not isinstance(value, kind):
        got = type(value).__name__
        raise ConfigError(f"{path.name}: '{key}' must be {kind.__name__}, got {got}")
    return value


def _positive_int(raw: dict[str, Any], key: str, path: Path) -> int:
    value = _require(raw, key, int, path)
    if value <= 0:
        raise ConfigError(f"{path.name}: '{key}' must be positive, got {value}")
    return int(value)


def _instruments(raw: dict[str, Any], path: Path) -> tuple[str, ...]:
    items = _require(raw, "instruments", list, path)
    if not items:
        raise ConfigError(f"{path.name}: 'instruments' is empty; nothing to fit")
    if not all(isinstance(s, str) and s.strip() for s in items):
        raise ConfigError(f"{path.name}: 'instruments' must be non-empty strings")
    # Refuse rather than strip: " SPY" and "SPY" would silently become the same
    # instrument, hiding a typo in the preset.
    untrimmed = [s for s in items if s != s.strip()]
    if untrimmed:
        raise ConfigError(
            f"{path.name}: 'instruments' entries must not have leading or trailing "
            f"whitespace: {untrimmed}"
        )
    # Bare symbols only. The depot keys regime series by bare symbol, and a
    # canonical id would build `regime:ticker:GLD` — a new series beside the
    # real one, with nothing failing. See regimes.series.bare_symbol.
    qualified = [s for s in items if ":" in s]
    if qualified:
        raise ConfigError(
            f"{path.name}: 'instruments' must be bare symbols, not canonical ids: "
            f"{qualified} — write 'GLD', not 'ticker:GLD'"
        )
    if len(set(items)) != len(items):
        dupes = sorted({s for s in items if items.count(s) > 1})
        raise ConfigError(f"{path.name}: 'instruments' contains duplicates: {dupes}")
    return tuple(items)


def _windows(raw: dict[str, Any], path: Path) -> tuple[Window, ...]:
    table = raw.get("windows")
    if not isinstance(table, dict) or not table:
        raise ConfigError(f"{path.name}: expected a non-empty [windows] table")

    windows: list[Window] = []
    for name, spec in table.items():
        where = f"{path.name}: [windows.{name}]"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: expected a table")
        unknown = set(spec) - {"lookback_years"}
        if unknown:
            raise ConfigError(f"{where}: unknown field(s) {sorted(unknown)}")
        years = spec.get("lookback_years")
        if years is not None:
            if isinstance(years, bool) or not isinstance(years, int) or years <= 0:
                raise ConfigError(f"{where}: 'lookback_years' must be a positive integer")
        windows.append(Window(name=name, lookback_years=years))

    unrestricted = [w.name for w in windows if w.lookback_years is None]
    if len(unrestricted) > 1:
        raise ConfigError(
            f"{path.name}: {unrestricted} all fit the whole history, so they would "
            f"share one unqualified series and overwrite each other"
        )
    # Two windows of the same length would produce the same variant tag and
    # upsert into each other's series.
    tags = [w.variant for w in windows if w.variant]
    if len(set(tags)) != len(tags):
        raise ConfigError(f"{path.name}: two windows share a lookback and would collide")
    return tuple(windows)


def _comparisons(raw: dict[str, Any], windows: tuple[Window, ...],
                 path: Path) -> tuple[Comparison, ...]:
    table = raw.get("comparisons", {})
    if not isinstance(table, dict):
        raise ConfigError(f"{path.name}: [comparisons] must be a table")

    known = {w.name for w in windows}
    out: list[Comparison] = []
    for name, spec in table.items():
        where = f"{path.name}: [comparisons.{name}]"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: expected a table")
        missing = [k for k in ("baseline", "candidate") if k not in spec]
        if missing:
            raise ConfigError(f"{where}: missing {missing}")
        unknown = set(spec) - {"baseline", "candidate"}
        if unknown:
            raise ConfigError(f"{where}: unknown field(s) {sorted(unknown)}")
        for role in ("baseline", "candidate"):
            if spec[role] not in known:
                raise ConfigError(
                    f"{where}: '{role}' names {spec[role]!r}, which is not a declared "
                    f"window; [windows] has {sorted(known)}"
                )
        if spec["baseline"] == spec["candidate"]:
            raise ConfigError(f"{where}: comparing {spec['baseline']!r} with itself")
        out.append(Comparison(name=name, baseline=spec["baseline"], candidate=spec["candidate"]))
    return tuple(out)


def load_config(path: str | Path) -> RegimeConfig:
    """Load and validate a regime run configuration.

    Args:
        path: Explicit path to a TOML file. There is no default and no search:
            the caller states which preset it is running.

    Raises:
        ConfigError: The file is absent, unparseable, or a key is missing, of
            the wrong type, or out of range. Every message names the file and
            the key, because this is read by whoever is fixing a failed run.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p.name}: not valid TOML: {exc}") from exc

    windows = _windows(raw, p)
    return RegimeConfig(
        instruments=_instruments(raw, p),
        windows=windows,
        comparisons=_comparisons(raw, windows, p),
        s_max=_positive_int(raw, "s_max", p),
        n_starts=_positive_int(raw, "n_starts", p),
        random_state=_require(raw, "random_state", int, p),
        retro_days=_positive_int(raw, "retro_days", p),
    )


__all__ = [
    "FULL_HISTORY",
    "Comparison",
    "ConfigError",
    "RegimeConfig",
    "Window",
    "load_config",
]
