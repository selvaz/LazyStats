"""I/O seams. Importing this package pulls nothing heavy: every integration
(market-data-hub, pandas) is imported lazily inside the function that needs
it, so ``lazystats`` stays a pure, dependency-free library."""

from lazystats.io.depot import ResultDepot, resolve_result_depot_path

__all__ = ["ResultDepot", "resolve_result_depot_path"]
