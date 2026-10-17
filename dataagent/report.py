"""Turn findings into a Markdown or a self-contained HTML report (charts embedded)."""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

BADGE = {"Verified": ("✅", "#15803d"), "Partly verified": ("⚠️", "#b45309"), "Rejected": ("❌", "#b91c1c")}


def markdown(session, findings, model: str) -> str:
    lines = [f"# Data analysis report: {Path(session.source).name}",
             f"_Generated {datetime.now():%d %b %Y %H:%M} by the Verified Data Analyst Agent ({model})._", "",
             "## Dataset", session.description, "", "### Cleaning steps"] + [f"- {x}" for x in session.cleaning_log]
    counts = {k: sum(f.label == k for f in findings) for k in BADGE}
    lines += ["", "## Trust summary", " | ".join(f"{BADGE[k][0]} {k}: {v}" for k, v in counts.items()), ""]
    for i, f in enumerate(findings, 1):
        lines += [f"## {i}. {f.question}", f"**{BADGE[f.label][0]} {f.label}**", "", f.summary, ""]
        for msg in f.issues + f.warnings:
            lines.append(f"> {msg}")
        for img in f.images:
            lines.append(f"![chart]({img})")
        lines += ["", "<details><summary>Code and raw output</summary>", "", "```python", f.code, "```", "```",
                  f.output.strip(), "```", "</details>", ""]
    return "\n".join(lines)


def html_report(session, findings, model: str) -> str:
    def img_tag(p):
        data = base64.b64encode(Path(p).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border:1px solid #ddd;border-radius:6px">'
    parts = [f"<h1>Data analysis report: {html.escape(Path(session.source).name)}</h1>",
             f"<p style='color:#666'>Generated {datetime.now():%d %b %Y %H:%M} by the Verified Data Analyst Agent "
             f"({html.escape(model)}).</p>", "<h2>Dataset</h2>", f"<p>{html.escape(session.description)}</p>",
             "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in session.cleaning_log) + "</ul>"]
    for i, f in enumerate(findings, 1):
        icon, color = BADGE[f.label]
        notes = "".join(f"<li>{html.escape(m)}</li>" for m in f.issues + f.warnings)
        parts.append(f"<h2>{i}. {html.escape(f.question)}</h2>"
                     f"<p><b style='color:{color}'>{icon} {f.label}</b></p><p>{html.escape(f.summary)}</p>"
                     + (f"<ul style='color:#92400e'>{notes}</ul>" if notes else "")
                     + "".join(img_tag(p) for p in f.images if Path(p).exists())
                     + f"<details><summary>Code and raw output</summary><pre>{html.escape(f.code)}</pre>"
                       f"<pre>{html.escape(f.output)}</pre></details>")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Data analysis report</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
            "pre{background:#f4f4f5;padding:.8rem;overflow:auto;font-size:.85rem}</style></head><body>"
            + "".join(parts) + "</body></html>")
