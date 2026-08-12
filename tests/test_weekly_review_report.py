from __future__ import annotations

from lazystats.weekly_review_report import render_html


def row_with_untrusted_text(text: str) -> dict:
    return {
        "result_id": text,
        "produced_by": text,
        "cadence": "stable",
        "created_at": "2026-08-12T00:00:00Z",
        "payload": {
            "week_start": "2026-08-01",
            "week_end": "2026-08-08",
            "n_daily_items": 1,
            "synthesis": {
                "narrative": text,
                "new_trends": [text],
                "regime_confirmations": [],
                "new_risks": [],
            },
            "verifications": [{
                "instrument": text,
                "anomaly_type": "return_outlier",
                "date": "2026-08-08",
                "verdict": "confirmed",
                "note": text,
            }],
        },
    }


def test_untrusted_text_cannot_close_the_script_or_inject_markup() -> None:
    attack = '</script><img src=x onerror="alert(1)">'
    html = render_html(row_with_untrusted_text(attack))
    assert attack not in html
    assert "\\u003c/script\\u003e" in html
    assert "\\u003cimg" in html


def test_untrusted_values_are_rendered_as_text_nodes() -> None:
    html = render_html(row_with_untrusted_text("ordinary text"))
    assert ".innerHTML" not in html
    assert "textContent" in html
    assert "createTextNode" in html
