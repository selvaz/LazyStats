"""Drawing a regime chart, against a real fitted model.

The one part of the report that cannot be checked from records alone: it needs
the model, and the model exists only where the engine ran. No hub is involved —
the series is synthetic — so this runs wherever the ``regimes`` extra is
installed, which is where the drawing could break.

What it guards is not how the picture looks. It is that a picture comes out at
all, that it is a PNG, and that drawing ninety of them does not leave ninety
figures open.
"""
from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("hmmlearn")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lazystats.regimes import MSRegimeEngine  # noqa: E402
from lazystats.regimes.charts import chart_png  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def run():
    """A two-regime series: quiet for a stretch, then turbulent."""
    rng = np.random.default_rng(11)
    values = np.concatenate([rng.normal(0.0004, 0.004, 600),
                             rng.normal(-0.001, 0.02, 300),
                             rng.normal(0.0004, 0.004, 400)])
    index = pd.bdate_range("2020-01-01", periods=len(values))
    frame = pd.DataFrame({"GLD": values}, index=index)
    return MSRegimeEngine(S_max=2, n_starts=3, random_state=123).fit(frame)


def test_a_chart_comes_out_as_a_png(run):
    png = chart_png(run, "GLD")
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 1000


def test_drawing_leaves_no_figure_open(run):
    """A figure per symbol, never closed, is a run over a large universe holding
    all of them until the process ends."""
    import matplotlib.pyplot as plt

    plt.close("all")
    for _ in range(3):
        chart_png(run, "GLD")
    assert plt.get_fignums() == []


def test_the_resolution_is_the_caller_s_to_choose(run):
    """It decides the size of the page on disk, and the page is emailed."""
    assert len(chart_png(run, "GLD", dpi=40)) < len(chart_png(run, "GLD", dpi=140))


def test_an_unknown_symbol_raises_rather_than_drawing_nothing(run):
    """A blank chart under the right heading is worse than no report."""
    with pytest.raises(KeyError, match="NOT_IN_THIS_FIT"):
        chart_png(run, "NOT_IN_THIS_FIT")
