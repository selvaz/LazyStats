"""The one place a regime chart becomes bytes.

The daily report embeds a plot per symbol. Drawing one needs the *fitted model*,
which exists only inside the step that fitted it and cannot cross a step
boundary — the pipeline threads identifiers between steps, never data, and a
model object is not JSON in any case.

So the chart is drawn while the model is alive and leaves as a PNG. That is the
narrowest thing that can carry a plot out of the fit: a few tens of kilobytes,
already in the form the report will embed, with nothing left holding the model.

Isolated in its own module so :mod:`~lazystats.regimes.report` stays a pure
function of data and can be tested without matplotlib ever being imported.
"""
from __future__ import annotations

import io
from typing import Any

#: How many observations a year the daily fit has. The engine's default assumes
#: weekly data, and a chart asking for five years of it would draw one — the
#: original carried this same correction at the call site.
DAILY_POINTS_PER_YEAR = 252


def chart_png(run: Any, symbol: str, *, last_years: int = 5, dpi: int = 110) -> bytes:
    """One symbol's price series with its regimes shaded, as a PNG.

    Args:
        run: The fitted result the engine returned, still alive.
        symbol: Which series to draw, as the fit named it.
        last_years: How much history to show.
        dpi: Raster resolution. The report embeds the result, so this is what
            decides the page's size on disk.

    Returns:
        The encoded image.
    """
    # Imported here, not at module scope: matplotlib belongs to the `regimes`
    # extra, and choosing a backend is a global side effect that must not happen
    # merely because something imported this package.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run.plot_series_with_regimes(symbol, last_years=last_years,
                                 points_per_year=DAILY_POINTS_PER_YEAR)
    figure = plt.gcf()
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        # Not closing leaks a figure per symbol: a run over a large universe
        # would hold every one of them until the process ended.
        plt.close(figure)
    return buffer.getvalue()


__all__ = ["DAILY_POINTS_PER_YEAR", "chart_png"]
