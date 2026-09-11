from __future__ import annotations

import math
import random
from typing import Mapping, Sequence


def paired_bootstrap_delta(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """Paired bootstrap CI for candidate-minus-baseline mean query recall."""

    common = sorted(set(baseline) & set(candidate))
    if not common:
        return {
            "paired_queries": 0,
            "mean_delta": 0.0,
            "ci_95": [0.0, 0.0],
            "probability_gt_zero": 0.0,
        }
    deltas = [candidate[qid] - baseline[qid] for qid in common]
    mean_delta = sum(deltas) / len(deltas)
    if samples <= 0:
        return {
            "paired_queries": len(common),
            "mean_delta": mean_delta,
            "ci_95": [mean_delta, mean_delta],
            "probability_gt_zero": float(mean_delta > 0),
        }
    rng = random.Random(seed)
    boot: list[float] = []
    n = len(deltas)
    for _ in range(samples):
        boot.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    lo = boot[int(0.025 * (samples - 1))]
    hi = boot[int(0.975 * (samples - 1))]
    return {
        "paired_queries": len(common),
        "mean_delta": mean_delta,
        "ci_95": [lo, hi],
        "probability_gt_zero": sum(value > 0 for value in boot) / samples,
    }


def exact_mcnemar(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
) -> dict[str, float | int]:
    """Exact two-sided McNemar/binomial test for paired all-evidence success."""

    common = sorted(set(baseline) & set(candidate))
    improved = sum((not baseline[qid]) and candidate[qid] for qid in common)
    regressed = sum(baseline[qid] and (not candidate[qid]) for qid in common)
    discordant = improved + regressed
    if not discordant:
        p_value = 1.0
    else:
        tail = min(improved, regressed)
        probability = sum(
            math.comb(discordant, k) for k in range(tail + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        "paired_queries": len(common),
        "candidate_improved": improved,
        "candidate_regressed": regressed,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
    }


def compare_query_outcomes(
    baseline_rows: Sequence[dict[str, object]],
    candidate_rows: Sequence[dict[str, object]],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    baseline_recall = {str(row["id"]): float(row["recall"]) for row in baseline_rows}
    candidate_recall = {str(row["id"]): float(row["recall"]) for row in candidate_rows}
    baseline_all = {str(row["id"]): bool(row["all_evidence"]) for row in baseline_rows}
    candidate_all = {str(row["id"]): bool(row["all_evidence"]) for row in candidate_rows}
    return {
        "macro_recall_delta": paired_bootstrap_delta(
            baseline_recall,
            candidate_recall,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "all_evidence_success": exact_mcnemar(baseline_all, candidate_all),
    }
