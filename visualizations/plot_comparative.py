"""
plot_comparative.py
===================
Section 2 – Cross-metric comparative analysis.

Rationale: In imbalanced clinical classification (AP prevalence ≈ 5.6 %),
single-metric evaluation is dangerous — a model can score 94 % accuracy by
simply predicting PD every time.  These plots expose *relationships between
metrics* so the analyst can choose the right operating point and detect
metric-gaming behaviour.

Plots produced
--------------
1. sensitivity_vs_specificity.png
   Diagnostic ROC operating-point scatter.

2. precision_recall_scatter.png
   Precision–recall trade-off for the minority (AP) class.

3. auc_vs_balanced_accuracy.png
   Threshold-independent (AUC) vs threshold-dependent (balanced accuracy).

4. metric_correlation_heatmap.png
   Pearson correlation matrix across all model × DR combinations.

5. parallel_coordinates.png
   Polyline per model × DR combination; each vertical axis is one metric
   (min–max normalised for visual comparability).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    element_text,
    facet_wrap,
    geom_abline,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    geom_tile,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient2,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DR_LABEL_ORDER,
    DR_METHODS,
    DR_SHAPES,
    METRIC_COLS,
    METRIC_LABELS,
    METRICS,
    MODEL_COLORS,
    get_combined_test_df,
    load_all_results,
    save_plot,
    theme_publication,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs", "comparative")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sensitivity vs Specificity
# ─────────────────────────────────────────────────────────────────────────────

def plot_sens_spec(df_all: pd.DataFrame) -> None:
    """
    Classic diagnostic performance chart: each point is one model × DR
    combination.  The ideal corner (high sensitivity, high specificity) is
    top-right.  Dashed reference lines mark clinically useful thresholds.
    """
    p = (
        ggplot(
            df_all,
            aes(
                x="pd_specificity",
                y="ap_recall_sensitivity",
                color="model_label",
                shape="dr_label",
            ),
        )
        + geom_vline(xintercept=0.80, linetype="dashed",
                     color="#ADB5BD", size=0.65)
        + geom_hline(yintercept=0.65, linetype="dashed",
                     color="#ADB5BD", size=0.65)
        + annotate("text", x=0.815, y=0.25,
                   label="Spec. ≥ 0.80", size=8, color="#6C757D",
                   angle=90, ha="left")
        + annotate("text", x=0.55, y=0.665,
                   label="Sens. ≥ 0.65", size=8, color="#6C757D",
                   ha="left")
        + geom_point(size=4.5, alpha=0.88)
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_shape_manual(values=DR_SHAPES, name="DR Method")
        + scale_x_continuous(limits=(0.50, 1.02),
                             breaks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        + scale_y_continuous(limits=(0.0, 1.05),
                             breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        + labs(
            title="Sensitivity vs Specificity — Diagnostic Operating Points",
            subtitle=(
                "Each point = one model × DR method  |  "
                "top-right = ideal  |  AP = atypical parkinsonism"
            ),
            x="PD Specificity (True Negative Rate for PD)",
            y="AP Sensitivity (True Positive Rate for AP)",
        )
        + theme_publication()
        + theme(figure_size=(11, 7.5))
    )
    save_plot(p, os.path.join(OUTDIR, "sensitivity_vs_specificity.png"),
              width=11, height=7.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Precision – Recall Scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_recall(df_all: pd.DataFrame) -> None:
    """
    Precision vs recall for the minority AP class.  F1 iso-curves are
    implicitly visible as the harmonic mean of the two axes.
    The dotted line represents precision = recall (F1 is maximised on it).
    """
    p = (
        ggplot(
            df_all,
            aes(
                x="ap_recall_sensitivity",
                y="ap_precision",
                color="model_label",
                shape="dr_label",
            ),
        )
        + geom_abline(slope=-1, intercept=1,
                      linetype="dotted", color="#CED4DA", size=0.75)
        + annotate("text", x=0.85, y=0.20,
                   label="F1 iso-curve", size=8, color="#ADB5BD",
                   angle=-45)
        + geom_point(size=4.5, alpha=0.88)
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_shape_manual(values=DR_SHAPES, name="DR Method")
        + scale_x_continuous(limits=(0.0, 1.05),
                             breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        + scale_y_continuous(limits=(0.0, 1.05),
                             breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        + labs(
            title="Precision–Recall Trade-off — Atypical Parkinsonism (AP) Class",
            subtitle=(
                "Each point = one model × DR method  |  "
                "top-right = ideal  |  class imbalance ≈ 5.6 %"
            ),
            x="AP Recall (Sensitivity)",
            y="AP Precision",
        )
        + theme_publication()
        + theme(figure_size=(11, 7.5))
    )
    save_plot(p, os.path.join(OUTDIR, "precision_recall_scatter.png"),
              width=11, height=7.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  AUC-ROC vs Balanced Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_auc_vs_balacc(df_all: pd.DataFrame) -> None:
    """
    AUC-ROC (threshold-independent) vs balanced accuracy (threshold-dependent).
    Points above the diagonal → model gains more in rank-discrimination than
    in calibrated class-balanced accuracy; points below → overfitted threshold.
    """
    p = (
        ggplot(
            df_all,
            aes(
                x="balanced_accuracy",
                y="auc_roc",
                color="model_label",
                shape="dr_label",
            ),
        )
        + geom_abline(slope=1, intercept=0,
                      linetype="dashed", color="#CED4DA", size=0.7)
        + annotate("text", x=0.595, y=0.606,
                   label="AUC = Bal.Acc", size=8, color="#ADB5BD",
                   angle=38, ha="left")
        + geom_point(size=4.5, alpha=0.88)
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_shape_manual(values=DR_SHAPES, name="DR Method")
        + scale_x_continuous(limits=(0.57, 0.77),
                             breaks=[0.58, 0.62, 0.66, 0.70, 0.74])
        + scale_y_continuous(limits=(0.70, 0.85),
                             breaks=[0.70, 0.73, 0.76, 0.79, 0.82, 0.85])
        + labs(
            title="AUC-ROC vs Balanced Accuracy",
            subtitle=(
                "Above diagonal → better rank-discrimination than calibrated decision  |  "
                "each point = model × DR"
            ),
            x="Balanced Accuracy",
            y="AUC-ROC",
        )
        + theme_publication()
        + theme(figure_size=(10, 7))
    )
    save_plot(p, os.path.join(OUTDIR, "auc_vs_balanced_accuracy.png"),
              width=10, height=7)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metric Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_correlation(df_all: pd.DataFrame) -> None:
    """
    Pearson correlation matrix computed across all model × DR combinations
    (n = 8 models × 5 DR methods = 40 points per metric pair).
    Reveals redundant vs complementary metrics.
    """
    corr = df_all[METRIC_COLS].corr(method="pearson")
    corr.index   = [METRICS[m] for m in corr.index]
    corr.columns = [METRICS[m] for m in corr.columns]

    corr_long = (
        corr.reset_index()
        .melt(id_vars="index", var_name="metric_y", value_name="correlation")
        .rename(columns={"index": "metric_x"})
    )
    corr_long["cell_label"] = corr_long["correlation"].round(2).astype(str)

    corr_long["metric_x"] = pd.Categorical(
        corr_long["metric_x"], categories=METRIC_LABELS
    )
    corr_long["metric_y"] = pd.Categorical(
        corr_long["metric_y"], categories=METRIC_LABELS[::-1]
    )

    p = (
        ggplot(corr_long, aes(x="metric_x", y="metric_y",
                              fill="correlation"))
        + geom_tile(color="white", size=0.7)
        + geom_text(aes(label="cell_label"), size=9, color="#212529")
        + scale_fill_gradient2(
            low="#1565C0",
            mid="white",
            high="#B71C1C",
            midpoint=0.0,
            limits=(-1, 1),
            name="Pearson r",
        )
        + labs(
            title="Metric Correlation Matrix",
            subtitle=(
                f"Pearson r  |  n = {len(df_all)} experiments "
                "(8 models × 5 DR methods)"
            ),
            x=None,
            y=None,
        )
        + theme_publication(base_size=10)
        + theme(
            axis_text_x=element_text(angle=35, ha="right"),
            figure_size=(9.5, 7.5),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "metric_correlation_heatmap.png"),
              width=9.5, height=7.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Parallel Coordinates
# ─────────────────────────────────────────────────────────────────────────────

def plot_parallel_coordinates(df_all: pd.DataFrame) -> None:
    """
    Each polyline is one model × DR combination.  Each vertical "axis" is
    one metric, min–max normalised across all experiments so axes are
    comparable.  Lines are coloured by model; transparency reveals clusters.
    """
    df_norm = df_all.copy()
    for mc in METRIC_COLS:
        lo, hi = df_norm[mc].min(), df_norm[mc].max()
        denom   = hi - lo if hi > lo else 1.0
        df_norm[mc] = (df_norm[mc] - lo) / denom

    rows = []
    for _, row in df_norm.iterrows():
        uid = f"{row['model_label']}|{row['dr_label']}"
        for i, (mc, ml) in enumerate(METRICS.items()):
            rows.append(
                dict(
                    uid=uid,
                    model_label=row["model_label"],
                    dr_label=row["dr_label"],
                    metric_idx=float(i),
                    metric=ml,
                    norm_value=float(row[mc]),
                )
            )
    df_long = pd.DataFrame(rows)
    df_long["metric"] = pd.Categorical(df_long["metric"],
                                       categories=METRIC_LABELS)

    p = (
        ggplot(df_long,
               aes(x="metric_idx", y="norm_value",
                   group="uid", color="model_label"))
        + geom_line(alpha=0.42, size=0.75)
        + scale_x_continuous(
            breaks=list(range(len(METRIC_LABELS))),
            labels=METRIC_LABELS,
        )
        + scale_y_continuous(
            limits=(0.0, 1.0),
            breaks=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + labs(
            title="Parallel Coordinates — All Metrics, Models & DR Methods",
            subtitle=(
                "Each line = one model × DR method  |  "
                "axes min–max normalised for visual comparability"
            ),
            x=None,
            y="Normalised Score",
        )
        + theme_publication()
        + theme(
            axis_text_x=element_text(angle=20, ha="right"),
            figure_size=(14, 6),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "parallel_coordinates.png"),
              width=14, height=6)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(all_data: dict) -> None:
    df_all = get_combined_test_df(all_data)
    print("[comparative] Sensitivity vs Specificity …")
    plot_sens_spec(df_all)
    print("[comparative] Precision–Recall …")
    plot_precision_recall(df_all)
    print("[comparative] AUC vs Balanced Accuracy …")
    plot_auc_vs_balacc(df_all)
    print("[comparative] Metric correlation heatmap …")
    plot_metric_correlation(df_all)
    print("[comparative] Parallel coordinates …")
    plot_parallel_coordinates(df_all)


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    run_all(load_all_results(RESULTS_DIR))