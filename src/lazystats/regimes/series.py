"""Identity of regime series in the shared result depot.

Two things decide where a regime estimate lands, and both are easy to get
wrong in a way nothing detects.

**The symbol must be bare.** Series keys in the depot read ``regime:GLD`` --
the format a 405,313-row migration already wrote. ``io.datahub.load_returns``
labels its columns canonically, as ``ticker:GLD``, so a pipeline that takes the
instrument straight from the dataset would build ``regime:ticker:GLD``. Nothing
would fail: ``save_stable_point`` would happily write a brand-new series beside
the old one, every history would restart at zero, and the 800,000 points already
there would simply be orphaned. :func:`bare_symbol` is the boundary.

**The database must be named, not guessed.** One depot is shared by every
deployment, and it has no per-database isolation of its own. A run against a
test or staging market database must therefore never supersede production's
vintages -- so a key is namespaced by the identity of the market database it
was computed from, unless that database *is* production.

The original implementation derived production from a hardcoded fallback path,
precisely because comparing against the *resolved* path would make a staging
deployment -- which sets its own ``MARKET_DATA_DB`` and never passes ``--db`` --
compare equal to itself and silently bypass the namespace. Here both paths are
passed in instead: the caller states which database it read and which one counts
as production. Same protection, no hidden dependency on an environment variable
being what someone assumed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: Domain prefix ``io.datahub`` puts on canonical instrument ids.
_TICKER_PREFIX = "ticker:"


def bare_symbol(instrument: str) -> str:
    """The plain symbol behind an instrument id.

    ``"ticker:GLD"`` and ``"GLD"`` both give ``"GLD"``. Only the ticker domain
    is supported by the loader, so any other prefix is a caller error rather
    than something to strip and hope about.

    Raises:
        ValueError: The id carries a domain this function must not silently
            discard -- dropping it would produce a key that looks right and
            points at another instrument's history.
    """
    text = instrument.strip()
    if not text:
        raise ValueError("instrument id is empty")
    if ":" not in text:
        return text
    domain, _, symbol = text.partition(":")
    if f"{domain}:" != _TICKER_PREFIX:
        raise ValueError(
            f"unsupported instrument domain {domain!r} in {instrument!r}; "
            f"regime series are keyed by bare ticker symbols"
        )
    if not symbol:
        raise ValueError(f"instrument id has no symbol: {instrument!r}")
    return symbol


def series_key(
    instrument: str,
    *,
    market_db: str | Path,
    production_db: str | Path,
    variant: str | None = None,
) -> str:
    """The depot series key for one symbol's regime estimate.

    Args:
        instrument: Symbol or canonical id; both forms are accepted and the
            key always uses the bare symbol.
        market_db: The market database this estimate was computed from.
        production_db: The database that counts as production. When the two
            match, the key is unqualified and supersedes the existing history;
            otherwise it is namespaced and cannot touch production's series.
        variant: Distinguishes a different fit of the *same* symbol against the
            *same* database -- a window restricted to the last eight years, say
            -- so it never upserts into the full-history series.

    Returns:
        ``regime:<symbol>``, plus ``@<db-id>`` when the market database is not
        production, plus ``:<variant>`` when a variant is named.
    """
    symbol = bare_symbol(instrument)
    if Path(market_db).resolve() == Path(production_db).resolve():
        base = f"regime:{symbol}"
    else:
        db_id = hashlib.sha256(str(Path(market_db).resolve()).encode("utf-8")).hexdigest()[:12]
        base = f"regime:{symbol}@{db_id}"
    return f"{base}:{variant}" if variant else base


__all__ = ["bare_symbol", "series_key"]
