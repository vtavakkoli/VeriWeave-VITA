from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from .utils import mean

METRICS = [
    "overall_score",
    "answer_quality",
    "verification_quality",
    "provenance_quality",
    "decision_accuracy",
    "review_routing_accuracy",
    "claim_precision",
    "unsupported_claim_rate",
    "citation_precision",
    "graph_reference_recall",
    "traceability_score",
    "counterfactual_stability",
    "evidence_cut_robustness",
    "decision_space_width",
    "vita_invariance",
    "closure_convergence",
    "residual_risk_mass",
    "synergistic_blind_spots",
    "vita_tested_subsets",
    "argumentation_conflict_free",
    "temporal_stability",
    "attention_selected_mass",
    "attention_covered_coalition_probability",
    "attention_residual_mass",
    "attention_entropy",
    "attention_effective_sample_size",
    "attention_risk_tail_mass",
    "method_attention_selected_mass",
    "method_attention_covered_coalition_probability",
    "method_attention_residual_mass",
    "method_attention_selected_clauses",
    "decision_changing_blind_spots",
    "robustness_quality",
    "latency_seconds",
]


def generate_report(result_dir: Path, rows: list[dict], traces: list[dict], manifest: dict, statistics: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    table_rows = []
    for method in manifest.get("methods", sorted(grouped)):
        items = grouped.get(method, [])
        cells = "".join(f"<td>{mean(float(item[metric]) for item in items):.3f}</td>" for metric in METRICS)
        table_rows.append(f"<tr><th>{html.escape(method)}</th>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(metric)}</th>" for metric in METRICS)

    sample = next((trace for trace in traces if trace["method"] == "VeriWeave-VITA-BPA"), traces[0] if traces else {})
    sample_json = html.escape(json.dumps(sample, indent=2, ensure_ascii=False))
    stats_json = html.escape(json.dumps(statistics, indent=2, ensure_ascii=False))
    manifest_json = html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))

    document = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>VeriWeave-VITA-BPA evaluation</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1450px;margin:40px auto;padding:0 20px;line-height:1.45}}
h1,h2{{letter-spacing:-.02em}} table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}}
th,td{{border:1px solid #d0d7de;padding:8px;text-align:right;white-space:nowrap}}th:first-child{{text-align:left;position:sticky;left:0;background:#fff}}
pre{{background:#f6f8fa;padding:16px;overflow:auto;border-radius:8px}}.notice{{border-left:4px solid #57606a;padding:10px 14px;background:#f6f8fa}}
</style></head>
<body><h1>VeriWeave-VITA-BPA evaluation report</h1>
<p class='notice'>All methods are audited by the same post-hoc verifier, singleton horizon stress test, coalitional VITA audit, Boltzmann Policy Attention stress test, argumentation check, and temporal replay. Inspired graph baselines are transparent standard-library approximations, not official reproductions of Microsoft GraphRAG, HippoRAG, or G-Retriever.</p>
<h2>Run manifest</h2><pre>{manifest_json}</pre>
<h2>Aggregate metrics</h2><table><thead><tr><th>Method</th>{headers}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Paired statistics</h2><pre>{stats_json}</pre>
<h2>Example VeriWeave-VITA-BPA trace</h2><pre>{sample_json}</pre>
</body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")
