"""
Publication-grade plotting utilities for groundwater data analysis outputs.

This script reads the CSVs you listed and produces:
- Transition heatmaps (2018→2019, 2019→2020)
- Covariate shift diagnostics (PSI/KS/mean shift, auto-detected)
- Feature–target correlation bar plot
- High-risk vs low-risk feature pattern boxplots (or grouped bars, auto)
- Outlier rate plots (overall and/or by year, auto)
- Class separability plot (auto based on available columns)
- Feature statistics table export + optional figure

Author: (you)
Run: python -m experiments.data_analysis
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths (edit if needed)
# -----------------------------
BASE = Path(r"C:\Users\Ilani\OneDrive\Desktop\water\Decision-Oriented-Forecasting-of-Groundwater-Usability-for-Irrigation-and-Livestock-Management\outputs\data_analysis")

PATHS = {
    "class_separability": BASE / "class_separability.csv",
    "covariate_shift": BASE / "covariate_shift_analysis.csv",
    "feature_statistics": BASE / "feature_statistics.csv",
    "feature_target_corr": BASE / "feature_target_correlations.csv",
    "high_risk_patterns": BASE / "high_risk_feature_patterns.csv",
    "outlier_analysis": BASE / "outlier_analysis.csv",
    "tm_18_19": BASE / "transition_matrix_2018_2019.csv",
    "tm_19_20": BASE / "transition_matrix_2019_2020.csv",
}

OUTDIR = BASE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Global plot style (JEM-friendly)
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 900,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": False,
})

HIGH_RISK_PREFIX = ("C4",)  # all C4*
HIGH_RISK_EXACT = {"C3S3", "C3S4"}  # add if present


# -----------------------------
# Helpers
# -----------------------------
def _safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p)
    # Standardize column names: lower + strip
    df.columns = [c.strip() for c in df.columns]
    return df


def _save_fig(fig: plt.Figure, name: str) -> None:
    png = OUTDIR / f"{name}.png"
    pdf = OUTDIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _is_class_label(s: str) -> bool:
    return bool(re.match(r"^C[1-4]S[1-4]$", str(s).strip()))


def _high_risk_mask(labels: List[str]) -> List[bool]:
    out = []
    for lab in labels:
        if lab in HIGH_RISK_EXACT:
            out.append(True)
        elif any(lab.startswith(pref) for pref in HIGH_RISK_PREFIX):
            out.append(True)
        else:
            out.append(False)
    return out


def _try_find(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive match
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _normalize_rows(mat: pd.DataFrame) -> pd.DataFrame:
    m = mat.copy()
    row_sums = m.sum(axis=1).replace(0, np.nan)
    return m.div(row_sums, axis=0).fillna(0.0)


def _plot_heatmap(ax, data: np.ndarray, xlabels: List[str], ylabels: List[str],
                  title: str, vmin: float = 0.0, vmax: float = 1.0, cmap: str = "viridis",
                  annotate: bool = True) -> None:
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if val >= 0.05:  # annotate only meaningful probabilities
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability")


# -----------------------------
# 1) Transition heatmaps
# -----------------------------
def plot_transition_heatmaps(path_18_19: Path, path_19_20: Path, normalize_rows: bool = True) -> None:
    tm1 = _safe_read_csv(path_18_19)
    tm2 = _safe_read_csv(path_19_20)

    # Assume first column might be an index column (class labels)
    # If there's an unnamed column, set it as index
    for tm in (tm1, tm2):
        if tm.columns[0].lower().startswith("unnamed") or tm.columns[0].lower() in ("from", "class", "classification"):
            tm.set_index(tm.columns[0], inplace=True)

    # Ensure indices/columns are class labels
    if not all(_is_class_label(i) for i in tm1.index.astype(str)):
        # try to use first col as index anyway
        pass

    # Align to common order (optional but nice)
    classes = sorted(set(tm1.index.astype(str)).union(set(tm1.columns.astype(str))))
    # keep only columns in tm
    tm1 = tm1.reindex(index=classes, columns=classes).fillna(0.0)
    tm2 = tm2.reindex(index=classes, columns=classes).fillna(0.0)

    if normalize_rows:
        tm1n = _normalize_rows(tm1)
        tm2n = _normalize_rows(tm2)
        suffix = "row_norm"
    else:
        tm1n, tm2n = tm1, tm2
        suffix = "raw"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_heatmap(axes[0], tm1n.values, list(tm1n.columns), list(tm1n.index),
                  "Transition Probabilities: 2018 → 2019", cmap="Blues", annotate=True)
    _plot_heatmap(axes[1], tm2n.values, list(tm2n.columns), list(tm2n.index),
                  "Transition Probabilities: 2019 → 2020", cmap="Blues", annotate=True)

    # Highlight high-risk classes with rectangles (both axes)
    for ax, tm in zip(axes, [tm1n, tm2n]):
        labels = list(tm.index.astype(str))
        risk = _high_risk_mask(labels)
        for i, is_risk in enumerate(risk):
            if is_risk:
                # Draw a rectangle around the y-label row
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(labels), 1,
                                           fill=False, edgecolor="red", linewidth=1.5))

    fig.suptitle("Inter-Annual Transitions in Irrigation Usability Classes (C#S#)", y=1.02)
    _save_fig(fig, f"transition_heatmaps_{suffix}")


# -----------------------------
# 2) Feature–target correlation barplot
# -----------------------------
def plot_feature_target_correlations(path: Path, top_k: int = 20) -> None:
    df = _safe_read_csv(path)

    # Detect columns
    feature_col = _try_find(df, ["feature", "Feature", "variable", "Variable", "name"])
    corr_col = _try_find(df, ["correlation", "Correlation", "corr", "Corr", "pearson_r", "r", "rho"])
    if feature_col is None or corr_col is None:
        raise ValueError(f"Could not detect feature/correlation columns in {path}. Columns: {df.columns.tolist()}")

    d = df[[feature_col, corr_col]].copy()
    d[corr_col] = pd.to_numeric(d[corr_col], errors="coerce")
    d = d.dropna().sort_values(by=corr_col, key=lambda s: s.abs(), ascending=False)

    d = d.head(top_k)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(d))))
    ax.barh(d[feature_col][::-1], d[corr_col][::-1])
    ax.set_xlabel("Correlation with target (sign preserved)")
    ax.set_title("Feature–Target Association (Top variables)")
    ax.axvline(0, linewidth=1)

    _save_fig(fig, "feature_target_correlations_top")


# -----------------------------
# 3) High-risk feature patterns
# -----------------------------
def plot_high_risk_patterns(path: Path, max_features: int = 8) -> None:
    df = _safe_read_csv(path)

    # Try to detect whether the file is "long format" (feature, group, value)
    feature_col = _try_find(df, ["feature", "Feature", "variable", "Variable"])
    group_col = _try_find(df, ["group", "Group", "risk_group", "risk", "label"])
    value_col = _try_find(df, ["value", "Value", "mean", "Mean", "median", "Median"])

    # Case A: Long format => boxplots per feature and group
    if feature_col and group_col and value_col and df[value_col].dtype != object:
        features = df[feature_col].unique().tolist()[:max_features]
        sub = df[df[feature_col].isin(features)].copy()

        fig, ax = plt.subplots(figsize=(10, 5))
        # Build grouped boxplots manually (matplotlib)
        groups = sub[group_col].unique().tolist()
        positions = []
        data = []
        labels = []
        pos = 1
        for f in features:
            for g in groups:
                vals = sub[(sub[feature_col] == f) & (sub[group_col] == g)][value_col].dropna().values
                data.append(vals)
                positions.append(pos)
                labels.append(f"{f}\n{g}")
                pos += 1
            pos += 1  # gap between features

        ax.boxplot(data, positions=positions, showfliers=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("High-risk vs Low-risk Hydrochemical Patterns")
        ax.set_ylabel(value_col)

        _save_fig(fig, "high_risk_feature_patterns_boxplot")
        return

    # Case B: Wide summary format (e.g., columns like low_mean, high_mean)
    # Try to find columns containing "high" and "low"
    cols = [c.lower() for c in df.columns]
    has_high = any("high" in c for c in cols)
    has_low = any("low" in c for c in cols)

    if feature_col and has_high and has_low:
        # Choose a numeric high/low pair
        high_candidates = [c for c in df.columns if "high" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
        low_candidates = [c for c in df.columns if "low" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
        if not high_candidates or not low_candidates:
            raise ValueError("Could not find numeric high/low columns in high_risk_feature_patterns.csv")

        high_col = high_candidates[0]
        low_col = low_candidates[0]

        d = df[[feature_col, low_col, high_col]].copy()
        d = d.head(max_features)

        x = np.arange(len(d))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, d[low_col], width, label="Low-risk")
        ax.bar(x + width/2, d[high_col], width, label="High-risk")
        ax.set_xticks(x)
        ax.set_xticklabels(d[feature_col], rotation=45, ha="right")
        ax.set_title("Hydrochemical Contrasts: High-risk vs Low-risk")
        ax.set_ylabel("Value (summary)")
        ax.legend()

        _save_fig(fig, "high_risk_feature_patterns_grouped_bars")
        return

    raise ValueError(f"Unrecognized format for {path}. Columns: {df.columns.tolist()}")


# -----------------------------
# 4) Covariate shift diagnostics
# -----------------------------
def plot_covariate_shift(path: Path, top_k: int = 20) -> None:
    df = _safe_read_csv(path)

    feature_col = _try_find(df, ["feature", "Feature", "variable", "Variable"])
    if feature_col is None:
        # some files may use first column as feature
        feature_col = df.columns[0]

    # Detect common shift metrics
    metric_candidates = []
    for c in df.columns:
        cl = c.lower()
        if cl == feature_col.lower():
            continue
        if any(k in cl for k in ["psi", "ks", "wasserstein", "emd", "mean_shift", "delta_mean", "shift", "cohen"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                metric_candidates.append(c)

    # Fallback: take all numeric columns except feature
    if not metric_candidates:
        metric_candidates = [c for c in df.columns if c != feature_col and pd.api.types.is_numeric_dtype(df[c])]

    if not metric_candidates:
        raise ValueError(f"No numeric covariate shift metrics found in {path}. Columns: {df.columns.tolist()}")

    # Choose the "main" metric for sorting (prefer PSI)
    sort_col = None
    for pref in ["psi", "ks", "wasserstein", "emd"]:
        for c in metric_candidates:
            if pref in c.lower():
                sort_col = c
                break
        if sort_col:
            break
    if sort_col is None:
        sort_col = metric_candidates[0]

    d = df[[feature_col] + metric_candidates].copy()
    d[sort_col] = pd.to_numeric(d[sort_col], errors="coerce")
    d = d.dropna(subset=[sort_col]).sort_values(by=sort_col, ascending=False).head(top_k)

    # Plot each metric in its own panel (clean for paper)
    n = len(metric_candidates)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metric_candidates):
        vals = pd.to_numeric(d[m], errors="coerce")
        ax.bar(d[feature_col], vals)
        ax.set_ylabel(m)
        ax.axhline(0, linewidth=1)
        ax.grid(False)

    axes[-1].set_xticks(range(len(d[feature_col])))
    axes[-1].set_xticklabels(d[feature_col], rotation=45, ha="right")
    fig.suptitle("Covariate Shift Diagnostics (Top-shifting features)", y=1.01)

    _save_fig(fig, "covariate_shift_diagnostics")


# -----------------------------
# 5) Outlier analysis
# -----------------------------
def plot_outlier_analysis(path: Path, top_k: int = 20) -> None:
    df = _safe_read_csv(path)

    feature_col = _try_find(df, ["feature", "Feature", "variable", "Variable"])
    year_col = _try_find(df, ["year", "Year"])
    rate_col = _try_find(df, ["outlier_rate", "OutlierRate", "rate", "Rate", "pct_outliers", "percent_outliers"])
    count_col = _try_find(df, ["outlier_count", "OutlierCount", "count", "Count"])

    if feature_col is None:
        feature_col = df.columns[0]

    # If we have year-wise rates, make a grouped bar plot
    if year_col and rate_col:
        d = df[[feature_col, year_col, rate_col]].copy()
        d[rate_col] = pd.to_numeric(d[rate_col], errors="coerce")
        d = d.dropna()

        # Keep top_k features by average rate
        top_feats = (d.groupby(feature_col)[rate_col].mean().sort_values(ascending=False).head(top_k).index.tolist())
        d = d[d[feature_col].isin(top_feats)]

        years = sorted(d[year_col].unique().tolist())
        feats = top_feats

        # pivot to wide for plotting
        wide = d.pivot_table(index=feature_col, columns=year_col, values=rate_col, aggfunc="mean").reindex(feats)

        x = np.arange(len(feats))
        width = 0.8 / max(1, len(years))

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, yr in enumerate(years):
            if yr in wide.columns:
                ax.bar(x + i * width - 0.4 + width / 2, wide[yr].values, width, label=str(int(yr)))
        ax.set_xticks(x)
        ax.set_xticklabels(feats, rotation=45, ha="right")
        ax.set_ylabel("Outlier rate")
        ax.set_title("Outlier Rates by Feature and Year")
        ax.legend(title="Year")

        _save_fig(fig, "outlier_analysis_by_year")
        return

    # Case B: No year column — simple bar of outlier rate per feature
    val_col = rate_col or count_col
    if val_col is None:
        # fallback: first numeric column that is not feature
        for c in df.columns:
            if c != feature_col and pd.api.types.is_numeric_dtype(df[c]):
                val_col = c
                break
    if val_col is None:
        raise ValueError(f"No numeric column found for outlier plot in {path}. Columns: {df.columns.tolist()}")

    d = df[[feature_col, val_col]].copy()
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna().sort_values(by=val_col, ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(d)), d[val_col].values)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d[feature_col].values, rotation=45, ha="right")
    ax.set_ylabel(val_col)
    ax.set_title("Outlier Analysis by Feature")

    _save_fig(fig, "outlier_analysis")


# -----------------------------
# 6) Class separability
# -----------------------------
def plot_class_separability(path: Path, top_k: int = 20) -> None:
    df = _safe_read_csv(path)

    feature_col = _try_find(df, ["feature", "Feature", "variable", "Variable"])
    stat_col = _try_find(df, ["f_statistic", "F_statistic", "f_stat", "statistic", "score"])
    pval_col = _try_find(df, ["pvalue", "p_value", "p", "Pvalue"])

    if feature_col is None:
        feature_col = df.columns[0]
    if stat_col is None:
        # fallback: first numeric column
        for c in df.columns:
            if c != feature_col and pd.api.types.is_numeric_dtype(df[c]):
                stat_col = c
                break
    if stat_col is None:
        raise ValueError(f"No numeric statistic column in {path}. Columns: {df.columns.tolist()}")

    d = df.copy()
    d[stat_col] = pd.to_numeric(d[stat_col], errors="coerce")
    d = d.dropna(subset=[stat_col]).sort_values(by=stat_col, ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(d))))
    bars = ax.barh(d[feature_col][::-1], d[stat_col][::-1])

    # Highlight significant features if p-value column available
    if pval_col and pval_col in d.columns:
        pvals = pd.to_numeric(d[pval_col][::-1], errors="coerce")
        for bar, pv in zip(bars, pvals):
            if pd.notna(pv) and pv < 0.05:
                bar.set_color("steelblue")
            else:
                bar.set_color("lightgray")

    ax.set_xlabel(f"{stat_col} (ANOVA)")
    ax.set_title("Class Separability by Feature")

    _save_fig(fig, "class_separability")


# -----------------------------
# 7) Feature statistics table figure
# -----------------------------
def plot_feature_statistics(path: Path) -> None:
    df = _safe_read_csv(path)

    # Select a meaningful subset of columns for the table
    display_cols = []
    for candidates in [
        ["feature", "Feature", "variable"],
        ["mean", "Mean"],
        ["std", "Std", "SD"],
        ["min", "Min"],
        ["median", "Median", "q50"],
        ["max", "Max"],
        ["skewness", "Skewness", "skew"],
        ["kurtosis", "Kurtosis", "kurt"],
        ["missing_pct", "missing_pct", "pct_missing"],
    ]:
        found = _try_find(df, candidates)
        if found:
            display_cols.append(found)

    if len(display_cols) < 2:
        display_cols = df.columns.tolist()

    d = df[display_cols].copy()

    # Round numeric columns for cleaner display
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].round(3)

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(display_cols)), 0.4 * len(d) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=d.values,
        colLabels=d.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    ax.set_title("Feature Statistics Summary", fontsize=13, pad=20)

    _save_fig(fig, "feature_statistics_table")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print(f"Reading CSVs from: {BASE}")
    print(f"Saving figures to: {OUTDIR}\n")

    tasks = [
        ("Transition heatmaps", lambda: plot_transition_heatmaps(PATHS["tm_18_19"], PATHS["tm_19_20"])),
        ("Feature-target correlations", lambda: plot_feature_target_correlations(PATHS["feature_target_corr"])),
        ("High-risk feature patterns", lambda: plot_high_risk_patterns(PATHS["high_risk_patterns"])),
        ("Covariate shift diagnostics", lambda: plot_covariate_shift(PATHS["covariate_shift"])),
        ("Outlier analysis", lambda: plot_outlier_analysis(PATHS["outlier_analysis"])),
        ("Class separability", lambda: plot_class_separability(PATHS["class_separability"])),
        ("Feature statistics", lambda: plot_feature_statistics(PATHS["feature_statistics"])),
    ]

    for name, fn in tasks:
        try:
            print(f"[*] {name} ...")
            fn()
            print(f"    Done.\n")
        except Exception as e:
            print(f"    FAILED: {e}\n")

    print("All done. Check:", OUTDIR)


if __name__ == "__main__":
    main()
