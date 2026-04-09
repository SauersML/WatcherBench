#!/usr/bin/env python3
"""
WatcherBench Statistical Analysis.

Reads results.json and produces rigorous cross-model comparisons:
  1. k=5 calibration
  2. Model × scenario heatmaps
  3. Two-way fixed-effects model (OLS on logprob differences)
  4. Estimated marginal means with Holm-corrected pairwise contrasts
  5. Coverage and missingness diagnostics

Uses statsmodels for proper inference on unbalanced designs.

Usage:
    python analyze.py                  # analyze results.json
    python analyze.py path/to/results.json
"""

import json
import math
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def load_results(path: str = "results.json") -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _fmt(s, width=7):
    return f"{s:.4f}" if s is not None and not (isinstance(s, float) and math.isnan(s)) else "---"


# ---------------------------------------------------------------------------
# 1. k=5 calibration
# ---------------------------------------------------------------------------

def k5_calibration(results: List[Dict]) -> None:
    """Compare scores at full-k vs k=5 for models that had k>5.

    NOTE: This checks whether k=20 models lose information when truncated to
    k=5.  It does NOT tell us whether k=5-only models would have had the
    target tokens in ranks 6-20, because different models have different
    token distributions.  A low change-rate here is necessary but not
    sufficient to treat k=5 and k=20 scores as interchangeable.
    """
    print("\n" + "=" * 90)
    print("  1. k=5 CALIBRATION  (upper-bound check)")
    print("  How often do k=20 models lose information when truncated to k=5?")
    print("  NOTE: This is a necessary but NOT sufficient condition for cross-k")
    print("  comparability — k=5-only models may have different rank distributions.")
    print("=" * 90)

    changed = 0
    total = 0
    k5_only_missing = 0
    k5_only_total = 0

    print(f"\n{'Model':<32} {'Scenario':<24} {'k':>3} {'Score_full':>11} {'Score_k5':>11} {'Status_full':<18} {'Status_k5':<18}")
    print("-" * 130)

    for r in results:
        if r["status"] == "api_error":
            continue
        # Track k=5-only models: how often are tokens missing?
        if r.get("top_k", 0) <= 5:
            k5_only_total += 1
            if r["status"] in ("deceptive_missing", "honest_missing"):
                k5_only_missing += 1
            continue
        total += 1
        s_full = r.get("scheming_score")
        s_k5 = r.get("scheming_score_k5")
        st_full = r["status"]
        st_k5 = r.get("status_k5", "?")

        if s_full != s_k5 or st_full != st_k5:
            changed += 1
            print(
                f"{r['model']:<32} {r['scenario']:<24} {r['top_k']:>3} "
                f"{_fmt(s_full):>11} {_fmt(s_k5):>11} {st_full:<18} {st_k5:<18}"
            )

    print("-" * 130)
    if total:
        print(f"  k=20 models: {changed}/{total} changed when truncated to k=5 ({changed/total*100:.1f}%)")
    if k5_only_total:
        print(f"  k=5-only models: {k5_only_missing}/{k5_only_total} have a missing token ({k5_only_missing/k5_only_total*100:.1f}%)")
        if k5_only_missing > 0:
            print("  ^ These missing tokens MIGHT have appeared at ranks 6-20 — scores are less reliable.")
    print()


# ---------------------------------------------------------------------------
# 2. Heatmap
# ---------------------------------------------------------------------------

def print_heatmap(results: List[Dict], models: List[str], scenarios: List[str],
                  title: str) -> None:
    lookup = {(r["model"], r["scenario"]): r for r in results}
    w = 34 + 9 * len(scenarios) + 8

    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")

    abbrevs = [s[:7] for s in scenarios]
    print(f"{'Model':<34}", end="")
    for a in abbrevs:
        print(f" {a:>7}", end="")
    print(f" {'Mean':>7}")
    print("-" * w)

    for model in models:
        print(f"{model:<34}", end="")
        scores = []
        for scen in scenarios:
            r = lookup.get((model, scen))
            if r and r.get("scheming_score") is not None and r["status"] == "both_found":
                s = r["scheming_score"]
                scores.append(s)
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
    print("=" * w)


# ---------------------------------------------------------------------------
# 3. Two-way fixed-effects OLS on logprob differences
# ---------------------------------------------------------------------------

def build_analysis_df(results: List[Dict]) -> pd.DataFrame:
    """Build a DataFrame of scorable results with logprob differences."""
    rows = []
    for r in results:
        if r["status"] != "both_found":
            continue
        lp_dec = r.get("lp_deceptive")
        lp_hon = r.get("lp_honest")
        if lp_dec is None or lp_hon is None:
            continue
        rows.append({
            "model": r["model"],
            "scenario": r["scenario"],
            "role_mode": r["role_mode"],
            "lp_diff": lp_dec - lp_hon,
            "scheming_score": r["scheming_score"],
        })
    return pd.DataFrame(rows)


def run_ols_analysis(df: pd.DataFrame, models: List[str], all_results: List[Dict] = None) -> None:
    """Fit two-way OLS and extract estimated marginal means."""
    print("\n" + "=" * 90)
    print("  2. TWO-WAY FIXED-EFFECTS MODEL")
    print("  lp_diff ~ C(model) + C(scenario)")
    print("  Estimates model effects controlling for scenario difficulty via OLS.")
    print("  Handles unbalanced designs correctly (no biased centering constants).")
    print("=" * 90)

    if len(df) < 3:
        print("\n  Not enough data points for OLS. Need a full run.")
        return

    n_models = df["model"].nunique()
    n_scenarios = df["scenario"].nunique()
    print(f"\n  Observations: {len(df)}  |  Models: {n_models}  |  Scenarios: {n_scenarios}")

    # Check we have enough data
    if n_models < 2:
        print("  Need at least 2 models with scorable results.")
        return
    if n_scenarios < 2:
        print("  Need at least 2 scenarios with scorable results.")
        return

    # Coverage table
    print(f"\n  Coverage (scorable scenarios per model):")
    for model in models:
        n = len(df[df["model"] == model])
        scenarios_list = sorted(df[df["model"] == model]["scenario"].tolist())
        print(f"    {model:<34} {n:>2} scenarios: {', '.join(s[:12] for s in scenarios_list)}")

    # Fit OLS
    # Use Treatment coding with the first model as reference
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_model = smf.ols("lp_diff ~ C(model) + C(scenario)", data=df).fit()
    except Exception as e:
        print(f"\n  OLS fitting failed: {e}")
        return

    print(f"\n  R² = {ols_model.rsquared:.4f}  |  Adj R² = {ols_model.rsquared_adj:.4f}")
    print(f"  Residual std = {np.sqrt(ols_model.mse_resid):.4f}")

    # --- Estimated Marginal Means (EMMs) ---
    # For each model, predict lp_diff at the mean scenario effect
    # EMM_i = intercept + α_i + mean(β_j) across all scenarios in the model
    # Simpler: predict for each model at each scenario, then average

    scorable_models = sorted(df["model"].unique())
    all_scenarios_in_data = sorted(df["scenario"].unique())

    # Build prediction grid: each model × all scenarios
    emm_data = []
    for model in scorable_models:
        preds = []
        for scen in all_scenarios_in_data:
            try:
                pred_df = pd.DataFrame({"model": [model], "scenario": [scen]})
                pred = ols_model.predict(pred_df)
                preds.append(float(pred.iloc[0]))
            except Exception:
                continue
        if preds:
            emm_lp = np.mean(preds)
            # Convert logprob difference to scheming score
            emm_score = 1.0 / (1.0 + np.exp(-emm_lp))
            emm_data.append((model, emm_lp, emm_score, len(df[df["model"] == model])))

    # Sort by EMM scheming score descending
    emm_data.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  ESTIMATED MARGINAL MEANS (model effects averaged over all scenarios)")
    print(f"  {'Model':<34} {'EMM lp_diff':>12} {'EMM Score':>10} {'N_obs':>6}")
    print(f"  {'-' * 66}")
    for model, emm_lp, emm_score, n in emm_data:
        print(f"  {model:<34} {emm_lp:>+12.4f} {emm_score:>10.4f} {n:>6}")
    print(f"  {'-' * 66}")
    print(f"  EMM lp_diff: estimated mean log(P_deceptive/P_honest), controlling for scenario")
    print(f"  EMM Score:   sigmoid(EMM lp_diff) — 0=honest, 1=scheming")

    # --- Pairwise contrasts with Holm correction ---
    print(f"\n{'=' * 90}")
    print(f"  3. PAIRWISE CONTRASTS (from OLS model, Holm-corrected)")
    print(f"{'=' * 90}")

    if len(scorable_models) < 2:
        print("\n  Need at least 2 scorable models for pairwise tests.")
        return

    pairs = list(combinations(scorable_models, 2))

    # Extract pairwise differences and p-values from the OLS model
    # Use the model's coefficient covariance to compute contrasts
    pair_results = []
    for a, b in pairs:
        # Compute contrast: EMM_a - EMM_b
        # Predict for both models at each scenario, compute mean difference
        diffs = []
        for scen in all_scenarios_in_data:
            try:
                pred_a = float(ols_model.predict(pd.DataFrame({"model": [a], "scenario": [scen]})).iloc[0])
                pred_b = float(ols_model.predict(pd.DataFrame({"model": [b], "scenario": [scen]})).iloc[0])
                diffs.append(pred_a - pred_b)
            except Exception:
                continue
        if not diffs:
            continue
        contrast = np.mean(diffs)  # Should be constant across scenarios (additive model)

        # For the p-value, use the original data — paired Wilcoxon on shared items
        shared_a = []
        shared_b = []
        for scen in all_scenarios_in_data:
            da = df[(df["model"] == a) & (df["scenario"] == scen)]
            db = df[(df["model"] == b) & (df["scenario"] == scen)]
            if len(da) > 0 and len(db) > 0:
                shared_a.append(float(da["lp_diff"].iloc[0]))
                shared_b.append(float(db["lp_diff"].iloc[0]))

        n_shared = len(shared_a)
        if n_shared == 0:
            pair_results.append((a, b, contrast, n_shared, np.nan, "no overlap"))
        elif n_shared < 5 or all(x == y for x, y in zip(shared_a, shared_b)):
            pair_results.append((a, b, contrast, n_shared, np.nan, f"n={n_shared}<5"))
        else:
            try:
                result = stats.wilcoxon(shared_a, shared_b, alternative="two-sided")
                pair_results.append((a, b, contrast, n_shared, float(result.pvalue), ""))
            except Exception:
                pair_results.append((a, b, contrast, n_shared, np.nan, "error"))

    # Apply Holm-Bonferroni correction to p-values
    raw_pvals = [pr[4] for pr in pair_results]
    valid_mask = [not np.isnan(p) for p in raw_pvals]
    valid_pvals = [p for p, v in zip(raw_pvals, valid_mask) if v]

    if valid_pvals:
        reject, corrected_pvals, _, _ = multipletests(valid_pvals, method="holm")
        # Map corrected p-values back
        corrected_iter = iter(corrected_pvals)
        final_pvals = []
        for p, v in zip(raw_pvals, valid_mask):
            if v:
                final_pvals.append(next(corrected_iter))
            else:
                final_pvals.append(np.nan)
    else:
        final_pvals = raw_pvals

    print(f"\n  {'Model A':<28} {'Model B':<28} {'N':>4} {'Δ lp_diff':>10} {'p (Holm)':>12} {'Sig':>6}")
    print(f"  {'-' * 92}")

    for i, (a, b, contrast, n_shared, raw_p, note) in enumerate(pair_results):
        p_corrected = final_pvals[i]
        if note:
            sig_str = note
            p_str = "---"
        elif np.isnan(p_corrected):
            sig_str = "---"
            p_str = "---"
        else:
            if p_corrected < 0.001:
                sig_str = "***"
            elif p_corrected < 0.01:
                sig_str = "**"
            elif p_corrected < 0.05:
                sig_str = "*"
            else:
                sig_str = "n.s."
            p_str = f"{p_corrected:.4f}"

        print(f"  {a:<28} {b:<28} {n_shared:>4} {contrast:>+10.4f} {p_str:>12} {sig_str:>6}")

    print(f"  {'-' * 92}")
    print(f"  Δ lp_diff: model A minus model B (+ means A more scheming)")
    print(f"  p-values: Wilcoxon signed-rank on shared scenarios, Holm-Bonferroni corrected")
    print(f"  *** p<.001  ** p<.01  * p<.05  n.s. not significant")

    # --- Missingness diagnostics ---
    print(f"\n{'=' * 90}")
    print(f"  4. MISSINGNESS DIAGNOSTICS")
    print(f"  Missing data is NOT random — models with extreme scores lose tokens from top-K.")
    print(f"{'=' * 90}")

    if all_results is None:
        all_results = []
    results_lookup = {(r["model"], r["scenario"]): r for r in all_results if r["status"] != "api_error"}
    for model in models:
        model_r = [(s, results_lookup.get((model, s))) for s in sorted(set(r["scenario"] for r in all_results))]
        missing = [(s, r["status"]) for s, r in model_r if r and r["status"] != "both_found"]
        if missing:
            print(f"\n  {model}:")
            for scen, status in missing:
                print(f"    {scen:<28} {status}")

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"  METHODOLOGY")
    print(f"{'=' * 90}")
    print(f"  Outcome:    lp_diff = log P(deceptive) - log P(honest)")
    print(f"  Model:      lp_diff ~ C(model) + C(scenario)  [two-way OLS]")
    print(f"  EMMs:       model effects averaged over ALL scenario levels")
    print(f"  Pairwise:   Wilcoxon signed-rank on shared scenarios")
    print(f"  Correction: Holm-Bonferroni for multiple comparisons")
    print(f"  Missing:    excluded (not imputed); documented in diagnostics")
    print(f"  Caveat:     missingness is MNAR (correlated with outcome).")
    print(f"              Very honest models lose deceptive tokens from top-K,")
    print(f"              biasing their remaining scores slightly upward.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(data: Dict[str, Any]) -> None:
    results = data["results"]
    models = data["models"]

    all_scenarios = list(dict.fromkeys(r["scenario"] for r in results))
    audit_scenarios = [s for s in all_scenarios if any(
        r["scenario"] == s and r["role_mode"] == "user_audit" for r in results)]
    continue_scenarios = [s for s in all_scenarios if any(
        r["scenario"] == s and r["role_mode"] == "assistant_continue" for r in results)]

    # 1. k=5 calibration
    k5_calibration(results)

    # 2. Heatmaps
    print_heatmap(results, models, audit_scenarios,
                  "HEATMAP: user_audit scenarios")
    if continue_scenarios:
        print_heatmap(results, models, continue_scenarios,
                      "HEATMAP: assistant_continue scenarios (interpret with caution)")

    # 3. OLS analysis
    df = build_analysis_df(results)
    if len(df) > 0:
        run_ols_analysis(df, models, all_results=results)
    else:
        print("\n  No scorable results for OLS analysis.")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    data = load_results(path)
    run_analysis(data)


if __name__ == "__main__":
    main()
