from __future__ import annotations

import json
from pathlib import Path

from .utils import mean


def write_statistics(result_dir: Path, rows: list[dict], target: str = "ClaimGraph") -> list[dict]:
    target_rows = {row["task_id"]: row for row in rows if row["method"] == target}
    methods = sorted({row["method"] for row in rows if row["method"] != target})
    output = []
    for method in methods:
        baseline = {row["task_id"]: row for row in rows if row["method"] == method}
        pairs = [(target_rows[key]["overall_score"], baseline[key]["overall_score"]) for key in target_rows.keys() & baseline.keys()]
        diffs = [a - b for a, b in pairs]
        output.append({
            "target": target,
            "baseline": method,
            "n": len(pairs),
            "mean_difference": round(mean(diffs), 6),
            "target_wins": sum(1 for d in diffs if d > 0),
            "baseline_wins": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
        })
    (result_dir / "statistics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
