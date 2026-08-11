"""DataSpace adapter for this repository's analysis result depot.

Makes LazyStats registrable in a :class:`lazydataspace.DataSpace` so a
workflow spanning several repositories can verify every source's readiness
together, before its first write.

Deliberately thin: no second path resolver and no second read API. The path
comes from :func:`lazystats.io.depot.resolve_result_depot_path`, and callers
reach results through :class:`lazystats.io.depot.ResultDepot` exactly as
they do today. Registering this Source changes nothing about how the repo
works standalone.

``lazydataspace`` is an optional dependency (``pip install
lazystats[lazydataspace]``). Nothing else in this package imports this
module, so the repo installs and runs without it.

Example:
    from lazydataspace import DataSpace
    from lazystats.dataspace_source import StatsSource

    space = DataSpace(StatsSource())
    space.require_ready()
    depot = space.source("stats").open()
"""

from __future__ import annotations

import os
import sqlite3

from lazydataspace import Health, SourceInfo

from lazystats.io.depot import ResultDepot, resolve_result_depot_path

#: What this endpoint offers, mirroring what the depot actually stores:
#: provenance-carrying analysis results, plus the per-date history of a
#: recurring ("stable") series.
CAPABILITIES = (
    "stats.results",
    "stats.series",
)

#: Presence of this table distinguishes "a readable SQLite file" from
#: "actually this repository's result depot".
_SENTINEL_TABLE = "analysis_results"


class StatsSource:
    """This repository's result depot, as a DataSpace ``Source``.

    Satisfies the ``lazydataspace.Source`` protocol structurally — no base
    class to inherit.

    Args:
        db_path: Explicit depot path. Omit to use this repo's own
            resolution order (``LAZYSTATS_RESULT_DEPOT_DB``).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "stats"

    @property
    def owner(self) -> str:
        return "lazystats"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return CAPABILITIES

    def describe(self) -> SourceInfo:
        """Return the non-sensitive self-description.

        Carries no path: ``SourceInfo`` has no field for one, and the
        description is written to be safe in a log.
        """
        return SourceInfo(
            name=self.name,
            owner=self.owner,
            capabilities=self.capabilities,
            description=(
                "Analysis result depot with full provenance: who produced a "
                "result, over which instruments, with which parameters and "
                "inputs, plus per-date history for recurring series. Read "
                "via lazystats.io.depot.ResultDepot."
            ),
        )

    def health(self) -> Health:
        """Resolve the depot path, open it read-only and confirm it is ours.

        A real check: it resolves, opens and queries. Three distinct
        failures are reported rather than collapsed into one: nothing
        configured, configured but absent, and readable but without the
        sentinel table.

        Deliberately does **not** construct a :class:`ResultDepot`: that
        constructor runs the schema script and a migration, so it *creates*
        the database when the file is missing — right for a writer, wrong
        for a readiness probe, which would report ready and hand the
        workflow an empty depot. Opens with ``mode=ro`` instead.

        Failure details name the configuration knob but never its value:
        this report is logged, and SQLite errors quote the full path.
        """
        try:
            path = resolve_result_depot_path(self._db_path)
        except Exception as exc:
            return Health(ready=False, detail=f"path resolution raised {type(exc).__name__}")

        if not path:
            return Health(
                ready=False,
                detail="no result depot configured (set LAZYSTATS_RESULT_DEPOT_DB)",
            )

        if path != ":memory:" and not os.path.exists(path):
            return Health(
                ready=False,
                detail="result depot does not exist (check LAZYSTATS_RESULT_DEPOT_DB)",
            )

        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (_SENTINEL_TABLE,),
                ).fetchone()
            finally:
                con.close()
        except Exception as exc:
            # Type only: SQLite errors quote the full file path.
            return Health(ready=False, detail=f"cannot open result depot: {type(exc).__name__}")

        if row is None:
            return Health(
                ready=False,
                detail=f"database is readable but has no {_SENTINEL_TABLE} table (wrong file?)",
            )
        return Health(ready=True)

    def open(self) -> ResultDepot:
        """Open the configured depot.

        The caller owns the returned object and must close it — this
        adapter holds no connection of its own, so registering a Source
        never opens a file.

        Raises:
            RuntimeError: No depot is configured. Explicit rather than
                silently returning an in-memory depot that would accept
                writes and lose them.
        """
        path = resolve_result_depot_path(self._db_path)
        if not path:
            raise RuntimeError("no result depot configured (set LAZYSTATS_RESULT_DEPOT_DB)")
        return ResultDepot(path)


__all__ = ["CAPABILITIES", "StatsSource"]
