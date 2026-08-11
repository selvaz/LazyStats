"""The DataSpace adapter for this repository's result depot.

Skipped entirely when ``lazydataspace`` is not installed: it is an optional
extra, and the repo must keep working standalone without it.
"""

from __future__ import annotations

import sqlite3

import pytest

lazydataspace = pytest.importorskip("lazydataspace", reason="optional [lazydataspace] extra")

from lazydataspace import DataSpace, Health, Source, SourceInfo  # noqa: E402

from lazystats.dataspace_source import StatsSource  # noqa: E402
from lazystats.io.depot import ResultDepot, resolve_result_depot_path  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    """The resolver reads LAZYSTATS_RESULT_DEPOT_DB; this machine has it set.

    Without this, the "nothing configured" tests would pass against the
    developer's real depot instead of an unconfigured resolver.
    """
    monkeypatch.delenv("LAZYSTATS_RESULT_DEPOT_DB", raising=False)


@pytest.fixture
def real_depot(tmp_path):
    """A depot created by the repo's own ResultDepot, not a hand-rolled schema."""
    path = str(tmp_path / "depot.db")
    depot = ResultDepot(path)
    depot.close()
    return path


class TestResolver:
    def test_explicit_wins(self):
        assert resolve_result_depot_path("/tmp/x.db") == "/tmp/x.db"

    def test_falls_back_to_the_env_var(self, monkeypatch):
        monkeypatch.setenv("LAZYSTATS_RESULT_DEPOT_DB", "/tmp/from-env.db")
        assert resolve_result_depot_path() == "/tmp/from-env.db"

    def test_returns_none_when_unconfigured(self):
        """None, not a default path: the caller decides what unconfigured means."""
        assert resolve_result_depot_path() is None


class TestProtocolConformance:
    def test_satisfies_the_source_protocol(self, real_depot):
        assert isinstance(StatsSource(real_depot), Source)

    def test_identity(self, real_depot):
        source = StatsSource(real_depot)
        assert source.name == "stats"
        assert source.owner == "lazystats"

    def test_registrable_in_a_dataspace(self, real_depot):
        space = DataSpace(StatsSource(real_depot))
        assert space.list() == ["stats"]


class TestDescribe:
    def test_returns_source_info(self, real_depot):
        info = StatsSource(real_depot).describe()
        assert isinstance(info, SourceInfo)
        assert "stats.results" in info.capabilities

    def test_description_does_not_leak_the_path(self, real_depot, tmp_path):
        import re

        info = StatsSource(real_depot).describe()
        assert real_depot not in info.description
        assert str(tmp_path) not in info.description
        assert not re.search(r"[A-Za-z]:[\\/]", info.description), "no absolute path"


class TestHealth:
    def test_ready_against_a_real_depot(self, real_depot):
        health = StatsSource(real_depot).health()
        assert isinstance(health, Health)
        assert health.ready is True

    def test_unready_when_nothing_is_configured(self):
        health = StatsSource().health()
        assert health.ready is False
        assert "LAZYSTATS_RESULT_DEPOT_DB" in health.detail

    def test_unready_when_the_file_is_absent(self, tmp_path):
        health = StatsSource(str(tmp_path / "missing.db")).health()
        assert health.ready is False
        assert "does not exist" in health.detail

    def test_absent_depot_is_not_created_by_the_check(self, tmp_path):
        """ResultDepot's constructor would CREATE the file — the reason
        health() opens with mode=ro instead of instantiating it."""
        missing = tmp_path / "missing.db"
        StatsSource(str(missing)).health()
        assert not missing.exists()

    def test_unready_when_the_file_is_not_a_database(self, tmp_path):
        junk = tmp_path / "not-a-db.db"
        junk.write_text("this is not sqlite", encoding="utf-8")
        health = StatsSource(str(junk)).health()
        assert health.ready is False
        assert "cannot open" in health.detail

    def test_unready_when_pointed_at_the_wrong_database(self, tmp_path):
        other = tmp_path / "other.db"
        con = sqlite3.connect(str(other))
        con.execute("CREATE TABLE something_else (x INTEGER)")
        con.commit()
        con.close()
        health = StatsSource(str(other)).health()
        assert health.ready is False
        assert "analysis_results" in health.detail

    def test_failure_detail_never_contains_the_path(self, tmp_path):
        junk = tmp_path / "secret-location.db"
        junk.write_text("junk", encoding="utf-8")
        detail = StatsSource(str(junk)).health().detail
        assert str(junk) not in detail
        assert "secret-location" not in detail


class TestOpen:
    def test_open_returns_a_usable_depot(self, real_depot):
        depot = StatsSource(real_depot).open()
        try:
            assert isinstance(depot, ResultDepot)
        finally:
            depot.close()

    def test_open_without_configuration_raises(self):
        """Explicit, rather than silently returning an in-memory depot that
        would accept writes and lose them."""
        with pytest.raises(RuntimeError, match="LAZYSTATS_RESULT_DEPOT_DB"):
            StatsSource().open()

    def test_registering_a_source_opens_nothing(self, tmp_path):
        """A Source is a description, not a connection."""
        missing = tmp_path / "untouched.db"
        DataSpace(StatsSource(str(missing)))
        assert not missing.exists()


class TestReadinessGate:
    def test_gate_passes_with_a_real_depot(self, real_depot):
        DataSpace(StatsSource(real_depot)).require_ready()

    def test_gate_fails_before_a_workflow_writes(self, tmp_path):
        space = DataSpace(StatsSource(str(tmp_path / "missing.db")))
        with pytest.raises(lazydataspace.SourceNotReadyError) as exc:
            space.require_ready()
        assert "stats" in str(exc.value)


class TestStandaloneIndependence:
    def test_the_package_does_not_import_the_adapter(self):
        """Importing lazystats must not require lazydataspace."""
        import ast
        import pathlib

        import lazystats

        package_dir = pathlib.Path(lazystats.__file__).parent
        importers = []
        for module in package_dir.rglob("*.py"):
            if module.name == "dataspace_source.py":
                continue
            tree = ast.parse(module.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    root = node.module.split(".")[0]
                    if root == "lazydataspace" or node.module.endswith("dataspace_source"):
                        importers.append(module.name)
                elif isinstance(node, ast.Import):
                    if any(a.name.split(".")[0] == "lazydataspace" for a in node.names):
                        importers.append(module.name)
        assert not importers, f"these modules would make lazydataspace mandatory: {importers}"
