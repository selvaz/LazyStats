"""Purity boundary (plan v3.1 Fase 6): lazystats never imports lazybridge,
lazytools, duckdb or HTTP clients — anywhere, not even lazily — and its core
imports nothing heavy at all. io/local.py must never be reachable from an LLM
profile: this repo enforces that it exports no ToolProvider."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "lazystats"

FORBIDDEN_ANYWHERE = {"lazybridge", "lazytools", "duckdb", "requests", "httpx",
                      "urllib3", "yfinance"}
# Heavy-but-legitimate lazy imports, allowed only inside function bodies:
LAZY_ONLY = {"pandas", "market_data_hub"}


def _imports(tree: ast.AST, *, top_level_only: bool) -> set[str]:
    nodes = tree.body if top_level_only else list(ast.walk(tree))  # type: ignore[attr-defined]
    mods: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module)
    return mods


def test_no_bridge_or_warehouse_imports_anywhere() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for mod in _imports(tree, top_level_only=False):
            if mod.split(".")[0] in FORBIDDEN_ANYWHERE:
                offenders.append(f"{path.name}: {mod}")
    assert not offenders, f"lazystats must stay pure; found: {offenders}"


def test_heavy_integrations_are_lazy_only() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for mod in _imports(tree, top_level_only=True):
            if mod.split(".")[0] in LAZY_ONLY:
                offenders.append(f"{path.name}: {mod}")
    assert not offenders, f"must be imported lazily inside functions: {offenders}"


def test_import_lazystats_pulls_no_third_party_modules() -> None:
    """`import lazystats` on a bare interpreter must succeed and drag in no
    third-party package at all (stdlib-only core)."""
    import subprocess

    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import lazystats\n"
        "import lazystats.io.depot\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "third_party = {m for m in new if m not in sys.stdlib_module_names\n"
        "               and not m.startswith('lazystats') and not m.startswith('_')}\n"
        "assert not third_party, f'unexpected deps: {third_party}'\n"
        "print('pure OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "pure OK" in proc.stdout


def test_local_loaders_are_notebook_only_marked() -> None:
    """io/local.py results self-identify as notebook-only, and the module
    exposes no ToolProvider (nothing with _is_lazy_tool_provider)."""
    import lazystats.io.local as local

    assert "NEVER exposed to LLM profiles" in (local.__doc__ or "")
    for name in dir(local):
        obj = getattr(local, name)
        assert not getattr(obj, "_is_lazy_tool_provider", False), name
