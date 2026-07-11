# LazyStats

Pure statistical library of the LazyBridge ecosystem (plan v3.1, Fase 6).
Stdlib-only core — **zero hard dependencies**, no lazybridge, no LLM wrapping
(that lives in [LazyTools](https://github.com/selvaz/LazyTools)'s
`statistical_analysis`, whose behaviour these functions replicate and whose
golden tests this repo preserves).

```
src/lazystats/
  core/     return_volatility / return_correlation / return_outliers + helpers
  models/   ReturnDataset (shape-compatible with the LazyTools wrapper)
  io/
    datahub.py   default data path: market-data-hub (lazy import)
    depot.py     SQLite result depot with mandatory provenance
    local.py     notebook-only CSV/DataFrame loaders — NEVER LLM tools
```

## Usage

```python
from lazystats import return_volatility
from lazystats.io.datahub import load_returns

ds = load_returns("SPY,TLT", start="2024-01-01", frequency="W")
print(return_volatility(ds, frequency="W")["volatility"])
```

## Roadmap (plan §7 Step 6)

1. ✅ Migrate the LazyTools statistics (golden-test parity).
2. ⬜ Migrate LazyHMM's engines (`core/`), absorbing its SQLite depot.
3. ⬜ Freeze LazyRay; migrate only after numeric + depot equivalence.
4. ⬜ Deprecate the source repos after one coexistence release.

## Install

```
pip install "lazystats @ git+https://github.com/selvaz/LazyStats.git"
```
