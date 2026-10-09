"""Lightweight spawn-importable fit doubles; no scientific or pytest imports."""
import os
from pathlib import Path


def fake_fit(symbol, *, marker=None, **kwargs):
    if marker:
        Path(marker).write_text(str(os.getpid()), encoding="utf-8")
    if symbol == "HANG":
        while True:
            pass  # CPU-bound, rather than an interruptible sleep
    if symbol == "ERROR":
        raise ValueError("bad synthetic returns")
    if symbol == "CRASH":
        os._exit(17)
    return {
        "symbol": symbol, "dates": ["2026-10-01"], "chart": None,
        "readings": [{"state": 0, "n_states": 1, "is_high_vol": False,
                      "prob_high_vol": 0.0, "state_probs": [1.0]}],
        "diagnostics": {"n_states": 1, "labels": ["Single State"], "states": []},
    }


def large_fit(symbol, **kwargs):
    return {"symbol": symbol, "chart": "x" * 2_000_000}
