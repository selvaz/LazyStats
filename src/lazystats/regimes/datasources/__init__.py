# -*- coding: utf-8 -*-
"""
lazystats.regimes.datasources — external data-source loaders for the LazyHMM tool layer
=============================================================================
Loaders in this subpackage pull a returns matrix from an external provider and
store it in the LazyHMM in-process/SQLite depot under a ``data_key``, using the
exact payload shape that ``lazystats.regimes.tools.load_time_series`` produces.  This lets
``fit_regimes(data_key=...)`` / ``RegimeEngine`` consume the data with zero glue.

Currently provides:
  * load_from_datahub — pull log-returns from the ``market-data-hub`` package.
"""
from __future__ import annotations

from .datahub import load_from_datahub

__all__ = ["load_from_datahub"]
