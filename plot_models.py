#!/usr/bin/env python3
"""Per-model P(deceptive) bar chart with BCa cluster-bootstrap CIs.

Methodology:
  variant   : p_var  = sigmoid(lp_diff) = P(dec | engaged)
              w_var  = P(dec) + P(hon)  (engagement)
  cell      : p_cell = engagement-weighted mean of p_var across variants
              W_cell = mean(w_var)      (representative cell engagement)
  model     : p_model = weighted mean of p_cell across scenarios (W_cell)
  CI        : BCa cluster bootstrap over base scenarios (20k iters),
              clipped to [0, 1].
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from analyze import build_analysis_df

RNG = np.random.default_rng(42)
N_BOOT = 20_000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def model_estimate(p_cell: np.ndarray, w_cell: np.ndarray) -> float:
    s = w_cell.sum()
    if s <= 0:
        return float(np.mean(p_cell))
    return float(np.average(p_cell, weights=w_cell))


def bca_ci(p_cell: np.ndarray, w_cell: np.ndarray, alpha: float = 0.05):
    n = len(p_cell)
    theta_hat = model_estimate(p_cell, w_cell)

    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        boots[b] = model_estimate(p_cell[idx], w_cell[idx])
    boots.sort()

    prop_below = float(np.mean(boots < theta_hat))
    eps = 1.0 / (N_BOOT * 10)
    prop_below = min(max(prop_below, eps), 1.0 - eps)
    z0 = norm.ppf(prop_below)

    jack = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        jack[i] = model_estimate(p_cell[mask], w_cell[mask])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den > 0 else 0.0

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)
    a1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)

    lo = float(max(0.0, np.quantile(boots, a1)))
    hi = float(min(1.0, np.quantile(boots, a2)))
    return theta_hat, lo, hi


# --- Data pipeline ---------------------------------------------------------
results = json.load(open(Path(__file__).parent / "results.json"))["results"]
df_var, _ = build_analysis_df(results)

df_var = df_var.copy()
df_var["p_var"] = sigmoid(df_var["lp_diff"].to_numpy())

# Cell-level (model × base_scenario) probability
cell_rows = []
for (model, base), grp in df_var.groupby(["model", "base_scenario"]):
    w = grp["engagement"].to_numpy()
    p = grp["p_var"].to_numpy()
    if w.sum() > 0:
        p_cell = float(np.average(p, weights=w))
    else:
        p_cell = float(np.mean(p))
    W_cell = float(np.mean(w))
    cell_rows.append({"model": model, "base_scenario": base, "p_cell": p_cell, "W_cell": W_cell})
df_cells = pd.DataFrame(cell_rows)

# Per-model aggregate
rows = []
for model, mdf in df_cells.groupby("model"):
    if len(mdf) < 2:
        continue
    p = mdf["p_cell"].to_numpy()
    w = mdf["W_cell"].to_numpy()
    theta, lo, hi = bca_ci(p, w)
    rows.append((model, theta, lo, hi, len(p)))

rows.sort(key=lambda r: r[1])  # most honest first
labels = [r[0].split("/")[-1] for r in rows]
theta = np.array([r[1] for r in rows])
lo = np.array([r[2] for r in rows])
hi = np.array([r[3] for r in rows])
ns = [r[4] for r in rows]
errs = np.vstack([theta - lo, hi - theta])

# --- Plot ------------------------------------------------------------------
INK = "#1a2332"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#1a2332"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(11, 6.2))

y_pos = np.arange(len(labels))

# Faint horizontal guideline per model row
for yi in y_pos:
    ax.plot([0, 1], [yi, yi], color=GRID, linewidth=0.8, zorder=1)

# CI whiskers — clean thin horizontal line with small caps
for yi, (l, u) in enumerate(zip(lo, hi)):
    ax.plot([l, u], [yi, yi], color=ACCENT, linewidth=2.2, solid_capstyle="butt", zorder=3)
    cap_h = 0.16
    for x in (l, u):
        ax.plot([x, x], [yi - cap_h, yi + cap_h], color=ACCENT, linewidth=2.2, solid_capstyle="butt", zorder=3)

# Point estimates — large filled dots, single accent color
ax.scatter(
    theta, y_pos,
    s=340,
    c="white",
    edgecolor=ACCENT,
    linewidths=2.6,
    zorder=4,
)
ax.scatter(
    theta, y_pos,
    s=140,
    c=ACCENT,
    zorder=5,
)

# Vertical reference at 0.5 (indifferent)
ax.axvline(0.5, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.7, len(labels) - 0.3)
ax.invert_yaxis()

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=14)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=12)
ax.set_xlabel("P(deceptive)", fontsize=14, labelpad=12, color=INK)

# Minimal spines — bottom only
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(INK)
ax.spines["bottom"].set_linewidth(1.2)

ax.tick_params(axis="y", length=0, pad=12)
ax.tick_params(axis="x", length=4, width=1.0, color=INK, pad=6)

plt.tight_layout()
out = Path(__file__).parent / "model_honesty.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
for r in rows:
    print(f"  {r[0]:<32} P={r[1]:.4f}  [{r[2]:.4f}, {r[3]:.4f}]  n={r[4]}")
