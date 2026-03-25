"""
plot_ablation.py
================
Section 3 – Dimensionality-reduction (DR) method ablation.

Research question: How much does the choice of DR preprocessing affect
downstream classifier performance, and which models are most sensitive?

Plots produced
--------------
1. dr_ablation_lines.png
   Connected line plot: x = DR method, y = metric; one polyline per model,
   faceted by metric.  Reveals DR-sensitivity and rank stability.

2. dr_ablation_bars.png
   Grouped bar chart for four key metrics: bars = models, fill = DR method.
   Direct quantity comparison per model–DR combination.

3. dr_model_heatmap.png
   Grid of heatmaps (DR method × model) – one sub-panel per metric.
   Best single overview of the 2D ablation space.

4. dr_stability.png
   Bar chart of the standard deviation of each metric *across DR methods*
   for each model.  High SD → performance is DR-sensitive; low SD → robust.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_tile,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_fill_gradient2,
    scale_y_continuous,
    theme,
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DR_COLORS,
    DR_LABEL_ORDER,
    DR_METHODS,
    METRIC_COLS,
    METRIC_LABELS,
    METRICS,
    MODEL_COLORS,
    MODEL_LABELS,
    get_combined_test_df,
    load_all_results,
    save_plot,
    theme_publication,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs", "ablation")

KEY_METRICS = {
    "balanced_accuracy":     "Balanced Acc.",
    "auc_roc":               "AUC-ROC",
    "ap_recall_sensitivity": "AP Sensitivity",
    "f1_ap":                 "F1 (AP)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: melt test frame to long format
# ─────────────────────────────────────────────────────────────────────────────

def _melt_metrics(df_all: pd.DataFrame,
                  metric_dict: dict | None = None) -> pd.DataFrame:
    md = metric_dict or METRICS
    rows = []
    for _, row in df_all.iterrows():
        for mc, ml in md.items():
            rows.append(
                dict(
                    model_label=row["model_label"],
                    dr_label=row["dr_label"],
                    metric=ml,
                    value=float(row[mc]),
                    cell_label=f"{row[mc]:.3f}",
                )
            )
    df_long = pd.DataFrame(rows)
    df_long["dr_label"] = pd.Categorical(df_long["dr_label"],
                                         categories=DR_LABEL_ORDER)
    df_long["metric"] = pd.Categorical(df_long["metric"],
                                       categories=list(md.values()))
    return df_long


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Line Plot Across DR Methods
# ─────────────────────────────────────────────────────────────────────────────

def plot_dr_line(df_all: pd.DataFrame) -> None:
    """
    Connected lines: x = DR method (ordered), y = metric value.
    One polyline per model.  Faceted by metric.
    Slope of each line segment quantifies how much a model is affected by the
    DR choice.  Flat lines = robust; steep = sensitive.
    """
    df_long = _melt_metrics(df_all)

    p = (
        ggplot(df_long,
               aes(x="dr_label", y="value",
                   color="model_label", group="model_label"))
        + geom_line(size=0.95, alpha=0.85)
        + geom_point(size=2.6, alpha=0.92)
        + facet_wrap("~metric", ncol=4, scales="free_y")
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_y_continuous(
            labels=lambda l: [f"{v:.2f}" for v in l],
        )
        + labs(
            title="DR Method Ablation — Metric Trends per Model",
            subtitle=(
                "Each line traces a model's performance across DR methods  |  "
                "slope magnitude = DR sensitivity"
            ),
            x="Dimensionality Reduction Method",
            y="Test Score",
        )
        + theme_publication(base_size=9)
        + theme(
            axis_text_x=element_text(angle=38, ha="right", size=7.5),
            figure_size=(17, 9),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "dr_ablation_lines.png"),
              width=17, height=9)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Grouped Bar Chart — Key Metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_dr_bars(df_all: pd.DataFrame) -> None:
    """
    For four key metrics: bars grouped by model, coloured by DR method.
    Allows direct quantity read-off within each model–metric cell.
    """
    df_long = _melt_metrics(df_all, KEY_METRICS)

    # Order models by mean balanced accuracy (descending)
    model_order = (
        df_all.groupby("model_label")["balanced_accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df_long["model_label"] = pd.Categorical(df_long["model_label"],
                                            categories=model_order)

    p = (
        ggplot(df_long,
               aes(x="model_label", y="value", fill="dr_label"))
        + geom_col(position="dodge", width=0.78, alpha=0.87)
        + facet_wrap("~metric", ncol=2, scales="fixed")
        + scale_fill_manual(values=DR_COLORS, name="DR Method")
        + scale_y_continuous(
            limits=(0, 1.06),
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
        )
        + labs(
            title="DR Method Comparison — Key Metrics",
            subtitle="Grouped by model; each colour = one DR method",
            x=None,
            y="Test Score",
        )
        + theme_publication(base_size=9)
        + theme(
            axis_text_x=element_text(angle=38, ha="right", size=8),
            figure_size=(14, 9),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "dr_ablation_bars.png"),
              width=14, height=9)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DR × Model Heatmap Grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_dr_heatmap(df_all: pd.DataFrame) -> None:
    """
    Heatmap grid: rows = DR method, columns = model.
    One sub-panel per metric.  Best single view of the full 2-D ablation space.
    """
    df_long = _melt_metrics(df_all)

    # Order models by mean balanced accuracy (best left)
    model_order = (
        df_all.groupby("model_label")["balanced_accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df_long["model_label"] = pd.Categorical(df_long["model_label"],
                                            categories=model_order)
    df_long["dr_label"]    = pd.Categorical(df_long["dr_label"],
                                            categories=DR_LABEL_ORDER[::-1])

    p = (
        ggplot(df_long,
               aes(x="model_label", y="dr_label", fill="value"))
        + geom_tile(color="white", size=0.55)
        + geom_text(aes(label="cell_label"), size=7, color="#212529")
        + facet_wrap("~metric", ncol=3)
        + scale_fill_gradient2(
            low="#C62828",
            mid="#FFF9C4",
            high="#1B5E20",
            midpoint=0.5,
            limits=(0, 1),
            name="Score",
        )
        + labs(
            title="DR Method × Model — Full Ablation Heatmap",
            subtitle="Each cell = test score  |  faceted by metric  |  green = better",
            x=None,
            y="DR Method",
        )
        + theme_publication(base_size=9)
        + theme(
            axis_text_x=element_text(angle=38, ha="right", size=7.5),
            figure_size=(18, 12),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "dr_model_heatmap.png"),
              width=18, height=12)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Stability Analysis — Std Across DR Methods
# ─────────────────────────────────────────────────────────────────────────────

def plot_dr_stability(df_all: pd.DataFrame) -> None:
    """
    For each model × metric, compute the standard deviation of the test score
    across all DR methods.  High SD → model is highly sensitive to DR choice.
    Models are ordered from most to least stable (lowest mean SD first).
    """
    stability = (
        df_all.groupby("model_label")[METRIC_COLS]
        .std()
        .reset_index()
    )

    rows = []
    for _, row in stability.iterrows():
        for mc, ml in METRICS.items():
            rows.append(
                dict(
                    model_label=row["model_label"],
                    metric=ml,
                    std=float(row[mc]),
                )
            )
    df_long = pd.DataFrame(rows)
    df_long["metric"] = pd.Categorical(df_long["metric"],
                                       categories=METRIC_LABELS)

    # Order: most stable (lowest mean SD across metrics) at bottom → top
    model_mean_std = (
        stability.set_index("model_label")[METRIC_COLS]
        .mean(axis=1)
        .sort_values(ascending=True)
    )
    df_long["model_label"] = pd.Categorical(
        df_long["model_label"], categories=model_mean_std.index.tolist()
    )

    p = (
        ggplot(df_long,
               aes(x="model_label", y="std", fill="model_label"))
        + geom_col(alpha=0.85, show_legend=False)
        + facet_wrap("~metric", ncol=4, scales="free_y")
        + scale_fill_manual(values=MODEL_COLORS)
        + labs(
            title="Model Stability Across DR Methods",
            subtitle=(
                "SD of test metric across 5 DR methods  |  "
                "higher = more sensitive to DR choice  |  models ordered least → most sensitive"
            ),
            x=None,
            y="Std Dev (across DR methods)",
        )
        + theme_publication(base_size=9)
        + theme(
            axis_text_x=element_text(angle=40, ha="right", size=7.5),
            figure_size=(16, 8),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "dr_stability.png"),
              width=16, height=8)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(all_data: dict) -> None:
    df_all = get_combined_test_df(all_data)
    print("[ablation] DR ablation line plot …")
    plot_dr_line(df_all)
    print("[ablation] DR ablation bar chart …")
    plot_dr_bars(df_all)
    print("[ablation] DR × Model heatmap …")
    plot_dr_heatmap(df_all)
    print("[ablation] Stability analysis …")
    plot_dr_stability(df_all)


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    run_all(load_all_results(RESULTS_DIR))