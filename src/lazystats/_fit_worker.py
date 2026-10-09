"""Minimal spawn entry point; scientific imports happen inside the fit budget."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def fit_worker(result_path: str, instrument: str, kwargs: dict[str, Any],
               fit_function: Callable[..., dict[str, Any]] | None) -> None:
    record: dict[str, Any]
    try:
        if fit_function is None:
            from lazystats.regimes.estimation import fit_symbol

            fit_function = fit_symbol
        record = {"result": fit_function(instrument, **kwargs)}
    except Exception as exc:
        record = {"error": f"{type(exc).__name__}: {exc}"}
    Path(result_path).write_text(json.dumps(record), encoding="utf-8")
