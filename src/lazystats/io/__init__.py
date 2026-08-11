"""I/O seams. Importing this package pulls nothing heavy: every integration
(market-data-hub, pandas) is imported lazily inside the function that needs
it, so ``lazystats`` stays a pure, dependency-free library.

The depot exports are exposed lazily for the same reason, via PEP 562:
importing ``lazystats.io.datahub`` to read returns should not also load the
result depot. Existing imports of ``ResultDepot`` and the canonical path
resolver keep working; the module is loaded on first attribute access.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type checkers and IDEs only; no runtime import
    from lazystats.io.depot import ResultDepot, resolve_result_depot_path

__all__ = ["ResultDepot", "resolve_result_depot_path"]

_LAZY = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from lazystats.io import depot

        return getattr(depot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY)
