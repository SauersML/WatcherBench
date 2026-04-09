#!/usr/bin/env python3
"""
WatcherBench Statistical Analysis.

Reads results.json and produces rigorous cross-model comparisons:
  1. k=5 calibration (do k=20 scores change when restricted to k=5?)
  2. Common items analysis (only compare on scenarios ALL models can score)
  3. Per-model mean + bootstrap 95% CI
  4. Paired Wilcoxon signed-rank tests between all model pairs
  5. Full heatmap (model × scenario)

Usage:
    python analyze.py                  # analyze results.json
    python analyze.py path/to/results.json
"""

import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple


def load_results(path: str = "results.json") -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. k=5 calibration
# ---------------------------------------------------------------------------

def k5_calibration(results: List[Dict]) -> None:
    """Compare scores at full-k vs k=5 for models that had k>5."""
    print("\n" + "=" * 90)
    print("  1. k=5 CALIBRATION")
    print("  Do scores change when we restrict k=20 models to only their top-5 logprobs?")
    print("=" * 90)

    changed = 0
    total = 0
    diffs = []

    print(f"\n{'Model':<32} {'Scenario':<24} {'k':>3} {'Score_full':>11} {'Score_k5':>11} {'Delta':>8} {'Status_full':<18} {'Status_k5':<18}")
    print("-" * 140)

    for r in results:
        if r.get("top_k", 0) <= 5:
            continue  # skip models that only had k=5
        if r["status"] == "api_error":
            continue

        total += 1
        s_full = r.get("scheming_score")
        s_k5 = r.get("scheming_score_k5")
        st_full = r["status"]
        st_k5 = r.get("status_k5", "?")

        if s_full != s_k5 or st_full != st_k5:
            changed += 1
            delta = "CHANGED"
            if s_full is not None and s_k5 is not None:
                delta = f"{s_k5 - s_full:+.4f}"
                diffs.append(s_k5 - s_full)
            elif s_full is not None and s_k5 is None:
                delta = "LOST"
            elif s_full is None and s_k5 is not None:
                delta = "GAINED"

            print(
                f"{r['model']:<32} {r['scenario']:<24} {r['top_k']:>3} "
                f"{_fmt_score(s_full):>11} {_fmt_score(s_k5):>11} {delta:>8} "
                f"{st_full:<18} {st_k5:<18}"
            )

    print("-" * 140)
    print(f"  Total k>5 results: {total}")
    print(f"  Changed when restricted to k=5: {changed} ({changed/total*100:.1f}%)" if total else "")
    if diffs:
        print(f"  Mean score delta (where both scorable): {sum(diffs)/len(diffs):+.4f}")
    if changed == 0:
        print("  CONCLUSION: k=5 and k=20 produce identical scores. k=5 models are directly comparable.")
    else:
        print(f"  CONCLUSION: {changed}/{total} scores differ. k=5 models lose information at ranks 6-20.")
    print()


# ---------------------------------------------------------------------------
# 2. Common items analysis
# ---------------------------------------------------------------------------

def find_common_items(results: List[Dict], models: List[str], use_k5: bool = False) -> List[str]:
    """Find scenarios where ALL specified models have both_found."""
    status_key = "status_k5" if use_k5 else "status"
    # Build model -> set of both_found scenarios
    scorable = defaultdict(set)
    for r in results:
        if r["model"] in models:
            st = r.get(status_key, r["status"])
            if st == "both_found":
                scorable[r["model"]].add(r["scenario"])

    if not scorable:
        return []

    # Intersect across all models
    common = None
    for model in models:
        s = scorable.get(model, set())
        common = s if common is None else common & s
    return sorted(common) if common else []


# ---------------------------------------------------------------------------
# 3. Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(scores: List[float], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    if not scores:
        return (float("nan"), float("nan"), float("nan"))
    n = len(scores)
    rng = random.Random(42)
    boot_means = []
    for _ in range(n_boot):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(n_boot * alpha / 2)]
    hi = boot_means[int(n_boot * (1 - alpha / 2))]
    return (sum(scores) / n, lo, hi)


# ---------------------------------------------------------------------------
# 4. Paired Wilcoxon signed-rank test (no scipy dependency)
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(x: List[float], y: List[float]) -> Tuple[float, str]:
    """
    Paired Wilcoxon signed-rank test (two-sided).
    Returns (test_statistic, significance_indicator).
    Significance based on critical values for small n.
    """
    assert len(x) == len(y)
    diffs = [(xi - yi) for xi, yi in zip(x, y) if xi != yi]
    n = len(diffs)
    if n == 0:
        return (0.0, "n/a (identical)")

    # Rank by absolute difference
    ranked = sorted(enumerate(diffs), key=lambda t: abs(t[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs(ranked[j][1]) == abs(ranked[i][1]):
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    w_plus = sum(ranks[k] for k in range(n) if diffs[ranked[k][0]] > 0)
    w_minus = sum(ranks[k] for k in range(n) if diffs[ranked[k][0]] < 0)
    T = min(w_plus, w_minus)

    # Critical values for two-sided test at alpha=0.05
    # (from standard tables, n -> critical T)
    crit_05 = {5: 1, 6: 2, 7: 4, 8: 6, 9: 8, 10: 11, 11: 14, 12: 17,
               13: 21, 14: 26, 15: 30, 16: 36, 17: 41, 18: 47, 19: 54, 20: 60}

    if n < 5:
        sig = "n<5"
    elif n in crit_05:
        sig = "p<.05 *" if T <= crit_05[n] else "n.s."
    else:
        # Normal approximation for n > 20
        mean_T = n * (n + 1) / 4
        std_T = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (T - mean_T) / std_T if std_T > 0 else 0
        sig = f"p<.05 * (z={abs(z):.2f})" if abs(z) > 1.96 else f"n.s. (z={abs(z):.2f})"

    return (T, sig)


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------

def _fmt_score(s):
    return f"{s:.4f}" if s is not None else "---"


def print_heatmap(results: List[Dict], models: List[str], scenarios: List[str],
                  title: str, score_key: str = "scheming_score") -> None:
    """Print model × scenario heatmap."""
    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["model"], r["scenario"])] = r

    # Header
    print(f"\n{'=' * (34 + 9 * len(scenarios))}")
    print(f"  {title}")
    print(f"{'=' * (34 + 9 * len(scenarios))}")

    # Scenario names (abbreviated)
    abbrevs = [s[:7] for s in scenarios]
    print(f"{'Model':<34}", end="")
    for a in abbrevs:
        print(f" {a:>7}", end="")
    print(f" {'Mean':>7}")
    print("-" * (34 + 9 * len(scenarios) + 8))

    for model in models:
        print(f"{model:<34}", end="")
        scores = []
        for scen in scenarios:
            r = lookup.get((model, scen))
            if r and r.get(score_key) is not None and r["status"] == "both_found":
                s = r[score_key]
                scores.append(s)
                # Color hint: high=scheming, low=honest
                print(f" {s:>7.3f}", end="")
            elif r and r["status"] == "api_error":
                print(f" {'ERR':>7}", end="")
            elif r is None:
                print(f" {'skip':>7}", end="")
            else:
                print(f" {'---':>7}", end="")

        if scores:
            print(f" {sum(scores)/len(scores):>7.3f}", end="")
        else:
            print(f" {'---':>7}", end="")
        print()

    print("=" * (34 + 9 * len(scenarios) + 8))


def run_analysis(data: Dict[str, Any]) -> None:
    results = data["results"]
    models = data["models"]

    all_scenarios = list(dict.fromkeys(r["scenario"] for r in results))
    audit_scenarios = list(dict.fromkeys(
        r["scenario"] for r in results if r["role_mode"] == "user_audit"
    ))
    continue_scenarios = list(dict.fromkeys(
        r["scenario"] for r in results if r["role_mode"] == "assistant_continue"
    ))

    # --- 1. k=5 calibration ---
    k5_calibration(results)

    # --- 2. Full heatmap ---
    print_heatmap(results, models, audit_scenarios,
                  "HEATMAP: user_audit scenarios (score at full k)")
    if continue_scenarios:
        print_heatmap(results, models, continue_scenarios,
                      "HEATMAP: assistant_continue scenarios (score at full k)")

    # --- 3. Common items analysis ---
    print("\n" + "=" * 90)
    print("  2. COMMON ITEMS ANALYSIS")
    print("  Only compare models on scenarios where ALL models have both_found.")
    print("=" * 90)

    # Find common items across all models (at full k)
    common_full = find_common_items(results, models, use_k5=False)
    # Find common items at k=5 (allows k=5 models to participate fairly)
    common_k5 = find_common_items(results, models, use_k5=True)

    print(f"\n  Common items at full k (all {len(models)} models): {common_full if common_full else 'NONE'}")
    print(f"  Common items at k=5 (all {len(models)} models):    {common_k5 if common_k5 else 'NONE'}")

    # If no common items across ALL models, find the largest subset of models
    # that share common items
    score_key = "scheming_score_k5"
    common = common_k5
    k_label = "k=5"
    comparison_models = models

    if not common:
        common = common_full
        score_key = "scheming_score"
        k_label = "full-k"

    if not common:
        # Try dropping models one at a time to find largest subset with common items
        print("\n  No common items across all models. Searching for largest comparable subset...")
        best_subset = []
        best_common = []
        for drop_n in range(1, len(models)):
            for combo in combinations(models, len(models) - drop_n):
                ci = find_common_items(results, list(combo), use_k5=False)
                if len(ci) > len(best_common) or (len(ci) == len(best_common) and len(combo) > len(best_subset)):
                    best_common = ci
                    best_subset = list(combo)
            if best_common:
                break

        if best_common:
            common = best_common
            comparison_models = best_subset
            score_key = "scheming_score"
            k_label = "full-k"
            dropped = [m for m in models if m not in comparison_models]
            print(f"  Best subset: {len(comparison_models)} models, {len(common)} common items")
            print(f"  Dropped: {dropped}")
        else:
            print("\n  WARNING: No common items found for any model subset.")
            return

    print(f"\n  Using {len(common)} common items at {k_label}: {common}")
    print(f"  Comparing {len(comparison_models)} models: {comparison_models}")

    # --- 4. Per-model mean + bootstrap CI ---
    print(f"\n{'=' * 90}")
    print(f"  3. MODEL RANKING (common items, {k_label}, bootstrap 95% CI)")
    print(f"{'=' * 90}")

    model_data = []
    for model in comparison_models:
        scores = []
        for scen in common:
            r_list = [r for r in results if r["model"] == model and r["scenario"] == scen]
            if r_list:
                r = r_list[0]
                s = r.get(score_key)
                if s is not None:
                    scores.append(s)
        if scores:
            mean, ci_lo, ci_hi = bootstrap_ci(scores)
            model_data.append((model, mean, ci_lo, ci_hi, len(scores)))
        else:
            model_data.append((model, None, None, None, 0))

    # Sort by mean descending
    model_data.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)

    print(f"\n{'Model':<34} {'Mean':>7} {'95% CI':>16} {'N':>4}")
    print("-" * 65)
    for model, mean, lo, hi, n in model_data:
        if mean is not None:
            print(f"{model:<34} {mean:>7.4f} [{lo:.4f}, {hi:.4f}] {n:>4}")
        else:
            print(f"{model:<34} {'---':>7} {'---':>16} {n:>4}")
    print("-" * 65)
    print("  Higher = more likely to commit to deception. 0=honest, 1=scheming.")
    print(f"  CI from 10,000 bootstrap resamples over {len(common)} common scenarios.")

    # --- 5. Paired Wilcoxon tests ---
    print(f"\n{'=' * 90}")
    print(f"  4. PAIRWISE SIGNIFICANCE (Wilcoxon signed-rank, two-sided, on common items)")
    print(f"{'=' * 90}")

    # Build model -> scores vector (aligned by common scenario order)
    model_vectors = {}
    for model in comparison_models:
        vec = []
        for scen in common:
            r_list = [r for r in results if r["model"] == model and r["scenario"] == scen]
            if r_list:
                s = r_list[0].get(score_key)
                vec.append(s if s is not None else float("nan"))
            else:
                vec.append(float("nan"))
        model_vectors[model] = vec

    # Only compare models that have scores on all common items
    scorable_models = [m for m in models if all(not math.isnan(v) for v in model_vectors[m])]

    if len(scorable_models) < 2:
        print("\n  Not enough models with complete common-item scores for pairwise tests.")
    else:
        pairs = list(combinations(scorable_models, 2))
        print(f"\n  Models with complete data: {len(scorable_models)}")
        print(f"  Pairwise tests: {len(pairs)}\n")

        print(f"{'Model A':<30} {'Model B':<30} {'T':>6} {'Sig':>20}")
        print("-" * 90)
        for a, b in pairs:
            T, sig = wilcoxon_signed_rank(model_vectors[a], model_vectors[b])
            print(f"{a:<30} {b:<30} {T:>6.1f} {sig:>20}")
        print("-" * 90)
        print(f"  * = significant at p<0.05 (two-sided)")
        print(f"  n.s. = not significant")
        print(f"  Paired by scenario across {len(common)} common items.")

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print("  METHODOLOGY SUMMARY")
    print(f"{'=' * 90}")
    print(f"  Scenarios total:     {len(all_scenarios)}")
    print(f"    user_audit:        {len(audit_scenarios)}")
    print(f"    assistant_continue:{len(continue_scenarios)}")
    print(f"  Models tested:       {len(models)}")
    print(f"  Common items ({k_label}):  {len(common)} — {common}")
    print(f"  Score: sigmoid(lp_deceptive - lp_honest)")
    print(f"  Only 'both_found' results included (no artificial defaults)")
    print(f"  Bootstrap CI: 10,000 resamples, 95% percentile interval")
    print(f"  Pairwise test: Wilcoxon signed-rank, two-sided, alpha=0.05")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    data = load_results(path)
    run_analysis(data)


if __name__ == "__main__":
    main()
