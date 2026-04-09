#!/usr/bin/env python3
"""
WatcherBench Statistical Analysis.

Reads results.json and produces rigorous cross-model comparisons:
  1. k=5 calibration
  2. Model × scenario heatmaps
  3. Two-way fixed-effects OLS with HC3 robust SEs
  4. Estimated marginal means with BH FDR-corrected pairwise contrasts (from model covariance)
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
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def load_results(path: str = "results.json") -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _fmt(s):
    return f"{s:.4f}" if s is not None and not (isinstance(s, float) and math.isnan(s)) else "---"


# ---------------------------------------------------------------------------
# 1. k=5 calibration
# ---------------------------------------------------------------------------

def k5_calibration(results: List[Dict]) -> None:
    """Show information lost when k=20 models are evaluated at k=5.

    All cross-model rankings use k=5 scores (the only metric every provider
    can produce).  This section shows what the k=20 models lose at that
    resolution — useful context, but the ranking itself is already fair.
    """
    print("\n" + "=" * 90)
    print("  1. k=5 RESOLUTION CHECK")
    print("  All rankings use k=5 scores (universal across providers).")
    print("  Below: what k=20 models lose when evaluated at k=5.")
    print("=" * 90)

    changed = 0
    total = 0

    print(f"\n{'Model':<32} {'Scenario':<24} {'k':>3} {'Score_k20':>11} {'Score_k5':>11} {'Status_k20':<18} {'Status_k5':<18}")
    print("-" * 130)

    for r in results:
        if r.get("top_k", 0) <= 5 or r["status"] == "api_error":
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
        print(f"  {changed}/{total} k=20 results changed at k=5 ({changed/total*100:.1f}%)")
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
            if r and r.get("scheming_score_k5") is not None and r.get("status_k5") == "both_found":
                s = r["scheming_score_k5"]
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
    """Build a DataFrame of scorable results with logprob differences.

    Includes both native-logprob and sampled models.  Sampled models
    (Anthropic) use first-word-only Monte Carlo at T=1, which estimates
    the same quantity as native logprobs: P(first_token = X | context).
    The lp_diff = log(count_dec/count_hon) is directly comparable because
    the matched-count denominator cancels in the difference.

    Sampled observations carry additional sampling noise; HC3 SEs adapt
    to this heteroscedasticity automatically.
    """
    rows = []
    for r in results:
        if r.get("status_k5", r["status"]) != "both_found":
            continue
        lp_dec = r.get("lp_deceptive_k5", r.get("lp_deceptive"))
        lp_hon = r.get("lp_honest_k5", r.get("lp_honest"))
        if lp_dec is None or lp_hon is None:
            continue
        rows.append({
            "model": r["model"],
            "scenario": r["scenario"],
            "role_mode": r["role_mode"],
            "lp_diff": lp_dec - lp_hon,
            "scheming_score": r.get("scheming_score_k5", r.get("scheming_score")),
        })
    return pd.DataFrame(rows)


def run_ols_analysis(df: pd.DataFrame, models: List[str], all_results: List[Dict] = None) -> None:
    """Fit two-way OLS with HC3 robust SEs; extract EMMs and pairwise contrasts.

    All inference (contrasts, SEs, p-values) comes from a single model with
    HC3 heteroscedasticity-consistent covariance.  No mixing of parametric
    model estimates with separate nonparametric tests — the contrast, its SE,
    and the p-value all flow from the same fitted model.
    """
    print("\n" + "=" * 90)
    print("  2. TWO-WAY FIXED-EFFECTS MODEL")
    print("  lp_diff ~ C(model) + C(scenario)")
    print("  HC3 heteroscedasticity-consistent standard errors.")
    print("  Handles unbalanced incomplete block designs correctly.")
    print("=" * 90)

    if len(df) < 3:
        print("\n  Not enough data points for OLS. Need a full run.")
        return

    n_models = df["model"].nunique()
    n_scenarios = df["scenario"].nunique()
    print(f"\n  Observations: {len(df)}  |  Models: {n_models}  |  Scenarios: {n_scenarios}")

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

    # Design connectedness check — if the bipartite graph (models ↔ scenarios)
    # has multiple connected components, contrasts between components are
    # non-estimable.  statsmodels will still fit but the coefficients are
    # meaningless across components.
    scorable_models_set = set(df["model"].unique())
    # Build adjacency: models sharing at least one scenario are connected
    adj: Dict[str, set] = {m: set() for m in scorable_models_set}
    for _, grp in df.groupby("scenario"):
        scen_models = list(grp["model"].unique())
        for i, m1 in enumerate(scen_models):
            for m2 in scen_models[i + 1:]:
                adj[m1].add(m2)
                adj[m2].add(m1)
    # Find all connected components, keep the largest
    remaining = set(scorable_models_set)
    components: List[set] = []
    while remaining:
        seed = next(iter(remaining))
        visited: set = set()
        queue = [seed]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)
        components.append(visited)
        remaining -= visited
    if len(components) > 1:
        largest = max(components, key=len)
        disconnected = scorable_models_set - largest
        print(f"\n  *** WARNING: Design is DISCONNECTED ({len(components)} components) ***")
        print(f"  Models {disconnected} share no scenarios with the main group.")
        print(f"  Contrasts involving these models are NOT estimable.")
        print(f"  Restricting analysis to the largest connected component ({len(largest)} models).\n")
        df = df[df["model"].isin(largest)]
        n_models = df["model"].nunique()
        n_scenarios = df["scenario"].nunique()

    # Fit OLS with HC3 robust standard errors
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_model = smf.ols("lp_diff ~ C(model) + C(scenario)", data=df).fit(
                cov_type="HC3"
            )
    except Exception as e:
        print(f"\n  OLS fitting failed: {e}")
        return

    # Classical fit for residual SD (HC3 doesn't change point estimates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols_classical = smf.ols("lp_diff ~ C(model) + C(scenario)", data=df).fit()
    residual_sd = float(np.sqrt(ols_classical.mse_resid)) if ols_classical.df_resid > 0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adj_r2 = ols_model.rsquared_adj
    adj_r2_s = f"{adj_r2:.4f}" if not (math.isnan(adj_r2) or math.isinf(adj_r2)) else "n/a"
    print(f"\n  R² = {ols_model.rsquared:.4f}  |  Adj R² = {adj_r2_s}")
    if ols_model.df_resid < 1:
        print(f"\n  *** WARNING: Model is saturated (df_resid = {ols_model.df_resid:.0f}) ***")
        print(f"  With {len(df)} observations and {len(ols_model.params)} parameters,")
        print(f"  there are no residual degrees of freedom. SEs, p-values, and")
        print(f"  effect sizes are undefined. Need more data (models × scenarios).")
        print(f"  Showing point estimates only.\n")
        residual_sd = 0.0
    else:
        print(f"  Residual SD = {residual_sd:.4f}  |  SE type: HC3 (robust)")

    # --- Omnibus Wald test for model effect ---
    # Before doing C(k,2) pairwise contrasts, confirm that the model factor
    # has any effect at all.  This uses the HC3 covariance for a robust Wald.
    scorable_models = sorted(df["model"].unique())
    all_scenarios_in_data = sorted(df["scenario"].unique())
    param_names = list(ols_model.params.index)
    model_coefs = [n for n in param_names if n.startswith("C(model)")]
    omnibus_sig = False
    if model_coefs and ols_model.df_resid >= 1:
        r_matrix = np.zeros((len(model_coefs), len(ols_model.params)))
        for i, name in enumerate(model_coefs):
            r_matrix[i, param_names.index(name)] = 1.0
        wald = ols_model.wald_test(r_matrix, use_f=True, scalar=True)
        f_stat = float(np.squeeze(wald.statistic))
        f_pval = float(np.squeeze(wald.pvalue))
        omnibus_sig = f_pval <= 0.05
        print(f"\n  Omnibus Wald F-test for model effect (HC3):")
        print(f"    F({len(model_coefs)}, {ols_model.df_resid:.0f}) = {f_stat:.3f},  p = {f_pval:.4f}"
              + ("  ***" if f_pval < 0.001 else "  **" if f_pval < 0.01 else "  *" if f_pval < 0.05 else "  n.s."))
        if not omnibus_sig:
            print(f"    No significant model effect — pairwise contrasts are EXPLORATORY.")
            print(f"    Significance stars suppressed (closed testing).")

    # --- Residual diagnostics ---
    if ols_model.df_resid >= 1:
        resids = ols_model.resid
        skew = float(stats.skew(resids))
        kurt = float(stats.kurtosis(resids))  # excess kurtosis (0 = normal)
        print(f"\n  Residual diagnostics:")
        print(f"    Skewness = {skew:+.3f}  (0 = symmetric)")
        print(f"    Excess kurtosis = {kurt:+.3f}  (0 = normal tails)")
        # Flag practical-significance thresholds for shape statistics.
        # These directed checks are more informative than the omnibus SW test
        # at small n because they target the specific departures that threaten
        # ANOVA validity: skewness biases means, heavy tails inflate Type I error.
        concerns = []
        if abs(skew) > 1.0:
            concerns.append(f"notable skewness (|{skew:+.2f}| > 1)")
        if abs(kurt) > 2.0:
            concerns.append(f"notable kurtosis (|{kurt:+.2f}| > 2)")
        if len(resids) >= 8:
            sw_stat, sw_p = stats.shapiro(resids)
            n_resid = len(resids)
            # Shapiro-Wilk is the most powerful omnibus normality test at
            # small n, but power is still low (n~10-30).  Report W as an
            # effect-size measure: values near 1 suggest approximate
            # normality; values below ~0.90 signal concern regardless of p.
            print(f"    Shapiro-Wilk W = {sw_stat:.4f},  p = {sw_p:.4f}  (n = {n_resid})")
            if sw_stat < 0.90:
                concerns.append(f"low Shapiro-Wilk W ({sw_stat:.3f} < 0.90)")
            if sw_p < 0.05:
                concerns.append("Shapiro-Wilk rejects normality (p < .05)")
        if concerns:
            print(f"    ** Normality concerns: {'; '.join(concerns)}")
            print(f"    HC3 SEs handle heteroscedasticity but not heavy tails.")
            print(f"    Interpret small-sample p-values with caution.")
        else:
            print(f"    No strong departures from normality detected,"
                  f" but low power at n = {len(resids)} limits detection.")

    # --- Estimated Marginal Means (EMMs) ---
    # In the additive model, EMM_i = intercept + α_i + mean(β_j).
    # Predicting across the full model×scenario grid and averaging handles
    # the unbalanced design correctly.
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
            emm_score = 1.0 / (1.0 + np.exp(-emm_lp))
            emm_data.append((model, emm_lp, emm_score, len(df[df["model"] == model])))

    emm_data.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  ESTIMATED MARGINAL MEANS (model effects averaged over all scenarios)")
    print(f"  {'Model':<34} {'EMM lp_diff':>12} {'EMM Score':>10} {'N_obs':>6}")
    print(f"  {'-' * 66}")
    for model, emm_lp, emm_score, n in emm_data:
        print(f"  {model:<34} {emm_lp:>+12.4f} {emm_score:>10.4f} {n:>6}")
    print(f"  {'-' * 66}")
    print(f"  EMM lp_diff: estimated mean log(P_deceptive/P_honest), controlling for scenario")
    print(f"  EMM Score:   sigmoid(EMM lp_diff) — 0=honest, 1=scheming")

    # --- Pairwise contrasts from the OLS coefficient covariance ---
    # In the additive model, the contrast between models i and j is exactly
    # α_i - α_j.  The SE comes from the HC3-robust covariance matrix, so
    # contrast, SE, t-stat, and p-value all derive from one coherent model.
    print(f"\n{'=' * 90}")
    if omnibus_sig:
        print(f"  3. PAIRWISE CONTRASTS (OLS coefficients, HC3 SEs, BH FDR-corrected)")
    else:
        print(f"  3. PAIRWISE CONTRASTS — EXPLORATORY (omnibus F n.s., stars suppressed)")
    print(f"  Each contrast is α_i - α_j from the fitted model.")
    print(f"  Controls for scenario difficulty; robust to heteroscedasticity.")
    print(f"{'=' * 90}")

    if len(scorable_models) < 2:
        print("\n  Need at least 2 scorable models for pairwise tests.")
        return

    params = ols_model.params
    cov = ols_model.cov_params()

    # Map each model to its Treatment-coded coefficient name (None = reference)
    model_to_coef = {}
    for model in scorable_models:
        coef_name = f"C(model)[T.{model}]"
        model_to_coef[model] = coef_name if coef_name in param_names else None

    pairs = list(combinations(scorable_models, 2))
    # (model_a, model_b, contrast, se, p_raw, cohens_d)
    pair_results: List[Tuple[str, str, float, float, float, float]] = []

    # Scenario-adjusted scores for Cohen's d: remove scenario effects so the
    # within-group SD reflects model-level variability, not scenario difficulty.
    scenario_coefs = [n for n in param_names if n.startswith("C(scenario)")]
    scenario_effects = np.zeros(len(df))
    for sc in scenario_coefs:
        scenario_effects += ols_model.params[sc] * (df["scenario"] == sc.split("[T.")[1].rstrip("]")).astype(float).values
    df_adj = df.copy()
    df_adj["lp_diff_adj"] = df["lp_diff"].values - scenario_effects

    for a, b in pairs:
        # Build contrast vector c such that c'β = α_a - α_b
        c = np.zeros(len(params))
        if model_to_coef[a] is not None:
            c[param_names.index(model_to_coef[a])] = 1.0
        if model_to_coef[b] is not None:
            c[param_names.index(model_to_coef[b])] = -1.0

        contrast = float(c @ params.values)
        se = float(np.sqrt(c @ cov.values @ c))

        if se < 1e-12:
            pair_results.append((a, b, contrast, se, np.nan, 0.0))
            continue

        t_stat = contrast / se
        p_raw = float(2.0 * stats.t.sf(abs(t_stat), ols_model.df_resid))

        # Cohen's d via pingouin with pooled within-group SD (scenario-adjusted).
        scores_a = df_adj.loc[df_adj["model"] == a, "lp_diff_adj"].values
        scores_b = df_adj.loc[df_adj["model"] == b, "lp_diff_adj"].values
        if len(scores_a) > 1 and len(scores_b) > 1:
            cohens_d = float(pg.compute_effsize(scores_a, scores_b, eftype="cohen"))
        else:
            cohens_d = 0.0

        pair_results.append((a, b, contrast, se, p_raw, cohens_d))

    # Benjamini-Hochberg FDR correction
    raw_pvals = [pr[4] for pr in pair_results]
    valid_mask = [not np.isnan(p) for p in raw_pvals]
    valid_pvals = [p for p, v in zip(raw_pvals, valid_mask) if v]

    if valid_pvals:
        _, corrected_pvals, _, _ = multipletests(valid_pvals, method="fdr_bh")
        corrected_iter = iter(corrected_pvals)
        final_pvals = [
            float(next(corrected_iter)) if v else np.nan
            for _, v in zip(raw_pvals, valid_mask)
        ]
    else:
        final_pvals = list(raw_pvals)

    n_tests = sum(valid_mask)
    print(f"\n  {n_tests} pairwise tests, Benjamini-Hochberg FDR-corrected (q ≤ 0.05)")
    print(f"\n  {'Model A':<26} {'Model B':<26} {'Δ':>8} {'SE':>7} {'t':>7} {'p_raw':>8} {'q':>8} {'d':>6} {'Sig':>5}")
    print(f"  {'-' * 105}")

    # Sort by adjusted p-value (most significant first) for readability
    display_order = sorted(
        range(len(pair_results)),
        key=lambda i: final_pvals[i] if not np.isnan(final_pvals[i]) else 2.0,
    )

    for i in display_order:
        a, b, contrast, se, p_raw, d = pair_results[i]
        q_val = final_pvals[i]
        if np.isnan(p_raw):
            print(f"  {a:<26} {b:<26} {contrast:>+8.4f} {'---':>7} {'---':>7} {'---':>8} {'---':>8} {'---':>6} {'deg.':>5}")
            continue

        t_stat = contrast / se if se > 1e-12 else 0.0
        p_raw_s = f"{p_raw:.4f}" if p_raw >= 0.0001 else f"{p_raw:.1e}"
        q_val_s = f"{q_val:.4f}" if q_val >= 0.0001 else f"{q_val:.1e}"
        if not omnibus_sig:
            sig_str = ""
        elif q_val < 0.001:
            sig_str = "***"
        elif q_val < 0.01:
            sig_str = "**"
        elif q_val < 0.05:
            sig_str = "*"
        else:
            sig_str = "n.s."

        print(
            f"  {a:<26} {b:<26} {contrast:>+8.4f} {se:>7.4f} {t_stat:>7.2f} "
            f"{p_raw_s:>8} {q_val_s:>8} {d:>+6.2f} {sig_str:>5}"
        )

    print(f"  {'-' * 105}")
    print(f"  Sorted by q-value (most significant first)")
    print(f"  Δ:     α_A - α_B from OLS (+ means A more scheming)")
    print(f"  SE:    HC3 heteroscedasticity-robust standard error")
    print(f"  p_raw: uncorrected two-sided p from t({ols_model.df_resid:.0f})")
    print(f"  q:     Benjamini-Hochberg FDR-corrected (among discoveries at q ≤ α, ≤ α fraction are false)")
    print(f"  d:     Cohen's d (Δ / pooled within-group SD, scenario-adjusted)")
    print(f"         |d| > 0.2 small, > 0.5 medium, > 0.8 large")

    # --- Missingness diagnostics ---
    print(f"\n{'=' * 90}")
    print(f"  4. MISSINGNESS DIAGNOSTICS")
    print(f"  Missing data is NOT random — models with extreme scores lose tokens from top-K.")
    print(f"{'=' * 90}")

    if all_results is None:
        all_results = []
    results_lookup = {(r["model"], r["scenario"]): r for r in all_results if r["status"] != "api_error"}
    all_scen_names = sorted(set(r["scenario"] for r in all_results))
    for model in models:
        missing = []
        for scen in all_scen_names:
            r = results_lookup.get((model, scen))
            if r:
                st = r.get("status_k5", r["status"])
                if st != "both_found":
                    missing.append((scen, st))
        if missing:
            print(f"\n  {model}:")
            for scen, status in missing:
                print(f"    {scen:<28} {status}")

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"  METHODOLOGY")
    print(f"{'=' * 90}")
    print(f"  Metric:     k=5 logprobs (universal — all providers return top-5)")
    print(f"  Outcome:    lp_diff = log P(deceptive) - log P(honest)  [at k=5]")
    print(f"  Model:      lp_diff ~ C(model) + C(scenario)  [two-way additive OLS]")
    print(f"              Assumes no model×scenario interaction (untestable with one obs/cell)")
    print(f"  SEs:        HC3 heteroscedasticity-consistent (robust to non-constant variance)")
    print(f"  Omnibus:    Wald F-test (HC3) gates pairwise contrasts (closed testing).")
    print(f"              If omnibus p > .05, contrasts are exploratory — stars suppressed")
    print(f"  EMMs:       model effects averaged over ALL scenario levels")
    print(f"              EMM Score = sigmoid(EMM lp_diff) shown for interpretability;")
    print(f"              all statistical comparisons are on the linear lp_diff scale")
    print(f"  Pairwise:   linear contrasts from OLS coefficient covariance matrix")
    print(f"              (contrast, SE, and p-value all from one coherent model)")
    print(f"  Correction: Benjamini-Hochberg FDR (q ≤ 0.05)")
    print(f"              (controls expected false discovery proportion)")
    print(f"  Effect size: Cohen's d (pooled within-group SD, scenario-adjusted)")
    print(f"  Diagnostics: Shapiro-Wilk + skewness/kurtosis on residuals;")
    print(f"               design connectedness check")
    print(f"  Missing:    excluded (not imputed); documented in diagnostics")
    print(f"  Sampled:    Anthropic models lack native logprobs. Estimated via Monte")
    print(f"              Carlo (128 samples at T=1, first-word-only matching).")
    print(f"              First-word matching estimates P(first_token = target),")
    print(f"              the same quantity native logprobs measure. HC3 SEs")
    print(f"              adapt to the higher variance of sampled observations.")
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

    # 3. OLS analysis (all models — native and sampled)
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
