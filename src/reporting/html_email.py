from __future__ import annotations

import pandas as pd


def render_html_report(signals: pd.DataFrame) -> str:
    """Build a lightweight HTML report for daily decisions."""
    table = signals.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    return f"<html><body><h2>Daily Trading Signals</h2>{table}</body></html>"


def build_email_payload(subject: str, html_body: str, recipients: list[str]) -> dict[str, object]:
    """Create an email payload consumed by external notifier."""
    return {"subject": subject, "html": html_body, "recipients": recipients}
