from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from .utils import mean

METRICS = [
    "overall_score", "reliability_score", "trustworthiness_score", "decision_accuracy",
    "claim_precision", "provenance_coverage", "graph_reference_recall", "traceability_score",
    "unsupported_claim_rate", "latency_seconds",
]


def generate_report(result_dir: Path, rows: list[dict], traces: list[dict], manifest: dict) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    table_rows = []
    for method, items in grouped.items():
        cells = "".join(f"<td>{mean(float(item[m]) for item in items):.3f}</td>" for m in METRICS)
        table_rows.append(f"<tr><th>{html.escape(method)}</th>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(m)}</th>" for m in METRICS)
    sample = next((trace for trace in traces if trace["method"] == "ClaimGraph"), traces[0] if traces else {})
    sample_json = html.escape(json.dumps(sample, indent=2, ensure_ascii=False))
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>ClaimGraph Evaluation</title>
<style>body{{font-family:system-ui;max-width:1300px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:right}}th:first-child{{text-align:left}}pre{{background:#f6f8fa;padding:16px;overflow:auto}}code{{font-family:ui-monospace}}</style></head>
<body><h1>ClaimGraph Evaluation Report</h1>
<p>Graph-grounded claim verification benchmark. Results in offline mode are deterministic smoke-test outputs, not paper claims.</p>
<h2>Run manifest</h2><pre>{html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))}</pre>
<h2>Aggregate metrics</h2><table><thead><tr><th>Method</th>{headers}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Example ClaimGraph trace</h2><pre>{sample_json}</pre></body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")
