"""Adaptive-controller dashboard generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def generate_dashboard(run_dir: Path, events: List[Dict[str, Any]], summary: Dict[str, Any]) -> Path:
    if not events:
        raise RuntimeError("No events to plot.")

    try:
        import pandas as pd
        import plotly.express as px
    except Exception as e:
        raise RuntimeError("Dashboard requires optional dependencies: pandas + plotly.") from e

    df = pd.DataFrame(events)
    out_path = Path(run_dir) / "dashboard.html"

    fig_score = px.line(
        df,
        x="t",
        y=["raw_div", "avg_score"],
        title="Adaptive Controller Signals",
    )

    fig_scale = px.line(
        df,
        x="t",
        y=["scale_used", "next_scale"],
        color="status",
        title="Adaptive Controller Intervention Schedule",
    )

    status_counts = summary.get("status_counts", {})
    status_df = pd.DataFrame(
        [{"status": k, "count": int(v)} for k, v in status_counts.items()]
    )
    fig_status = px.bar(status_df, x="status", y="count", title="Status Counts")

    html = (
        "<html><head><meta charset='utf-8'><title>Adaptive Controller Dashboard</title></head><body>"
        "<h1>Adaptive Controller Run Dashboard</h1>"
        f"<p>Run dir: {run_dir}</p>"
        + fig_score.to_html(include_plotlyjs="cdn", full_html=False)
        + fig_scale.to_html(include_plotlyjs=False, full_html=False)
        + fig_status.to_html(include_plotlyjs=False, full_html=False)
        + "</body></html>"
    )

    out_path.write_text(html, encoding="utf-8")
    return out_path
