from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .utils import mean


def _bootstrap_ci(values: list[float], *, seed: int = 7, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(resample))
    means.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return means[low_index], means[high_index]


def _two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    cumulative = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * cumulative)


def write_statistics(result_dir: Path, rows: list[dict], target: str = "VeriWeave-VITA-BPA", seed: int = 7) -> list[dict]:
    target_rows = {row["task_id"]: row for row in rows if row["method"] == target}
    methods = sorted({row["method"] for row in rows if row["method"] != target})
    output = []
    for method in methods:
        baseline = {row["task_id"]: row for row in rows if row["method"] == method}
        pairs = [
            (float(target_rows[key]["overall_score"]), float(baseline[key]["overall_score"]))
            for key in sorted(target_rows.keys() & baseline.keys())
        ]
        differences = [left - right for left, right in pairs]
        wins = sum(1 for value in differences if value > 0)
        losses = sum(1 for value in differences if value < 0)
        ties = sum(1 for value in differences if value == 0)
        ci_low, ci_high = _bootstrap_ci(differences, seed=seed)
        output.append({
            "target": target,
            "baseline": method,
            "n": len(pairs),
            "mean_difference": round(mean(differences), 6),
            "bootstrap_95_ci": [round(ci_low, 6), round(ci_high, 6)],
            "target_wins": wins,
            "baseline_wins": losses,
            "ties": ties,
            "two_sided_sign_test_p": round(_two_sided_sign_p(wins, losses), 10),
        })
    (result_dir / "statistics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output
