"""
plot_ranking.py
===============
Section 4 – Model ranking and generalisation analysis.

Purpose: Aggregate evidence across all metrics and DR methods into a single
evidence-based ranking that can guide clinical deployment decisions.  Inspired
by Roffo et al. (2016) "Infinite Feature Selection" and the benchmarking
methodology in Wainberg et al. (2016) "Deep learning in biomedicine".

Plots produced
--------------
1. rank_heatmap.png
   Average ordinal rank per model × metric (aggregated over DR methods).
   Green = rank 1; warm red = rank 8.

2. borda_count.png
   Borda count aggregate: sum of (n_models − rank) points across all
   metrics and DR methods.  Higher total = better overall performer.

3. generalization_gap.png
   Scatter of 5-fold CV mean vs held-out test score for four key metrics.
   Points above the diagonal = positive transfer; below = CV overfitting.

4. bump_chart.png
   Bump chart: x = metric, y = ordinal rank (1 = best), one polyline per model.
   Rank averaged over all DR methods.  Crossing lines reveal metric-dependent
   rank inversions.
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
    geom_col,
    geom_line,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    position_nudge,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_reverse,
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
    MODEL_LABELS,
    get_combined_test_df,
    load_all_results,
    save_plot,
    theme_publication,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs", "ranking")

KEY_METRICS = {
    "balanced_accuracy":     "Balanced Acc.",
    "auc_roc":               "AUC-ROC",
    "ap_recall_sensitivity": "AP Sensitivity",
    "f1_ap":                 "F1 (AP)",
}

N_MODELS = len(MODEL_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute per-DR per-metric ranks, then average
# ─────────────────────────────────────────────────────────────────────────────

def _compute_avg_ranks(df_all: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns [model_label, metric, avg_rank]."""
    rank_rows = []
    for dr_label in df_all["dr_label"].unique():
        sub = df_all[df_all["dr_label"] == dr_label].copy()
        for mc, ml in METRICS.items():
            sub_rank = sub[mc].rank(ascending=False, method="min")
            for model_label, rank in zip(sub["model_label"], sub_rank):
                rank_rows.append(
                    dict(model_label=model_label, metric=ml, rank=float(rank))
                )
    df_ranks = pd.DataFrame(rank_rows)
    avg_ranks = (
        df_ranks.groupby(["model_label", "metric"])["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "avg_rank"})
    )
    return avg_ranks


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Rank Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_heatmap(df_all: pd.DataFrame) -> None:
    """
    Average ordinal rank per model (row) × metric (column).
    Ranks averaged across all DR methods (1 = best, 8 = worst).
    Green palette: dark green = top rank.
    """
    avg_ranks = _compute_avg_ranks(df_all)
    avg_ranks["cell_label"] = avg_ranks["avg_rank"].round(1).astype(str)

    # Order models: best overall rank (lowest mean) at top
    model_order = (
        avg_ranks.groupby("model_label")["avg_rank"]
        .mean()
        .sort_values(ascending=False)   # reversed because y-axis is top→bottom
        .index.tolist()
    )
    avg_ranks["model_label"] = pd.Categorical(avg_ranks["model_label"],
                                              categories=model_order)
    avg_ranks["metric"]      = pd.Categorical(avg_ranks["metric"],
                                              categories=METRIC_LABELS)

    p = (
        ggplot(avg_ranks, aes(x="metric", y="model_label",
                              fill="avg_rank"))
        + geom_tile(color="white", size=0.6)
        + geom_text(aes(label="cell_label"), size=9.5, color="#212529")
        + scale_fill_gradient(
            low="#1B5E20",
            high="#FFCDD2",
            limits=(1, N_MODELS),
            name="Avg Rank\n(1 = best)",
        )
        + labs(
            title="Model Ranking Heatmap",
            subtitle=(
                "Average ordinal rank (1 = best) per metric  |  "
                "aggregated over all 5 DR methods  |  models ordered by overall rank"
            ),
            x=None,
            y=None,
        )
        + theme_publication(base_size=10)
        + theme(
            axis_text_x=element_text(angle=35, ha="right"),
            figure_size=(11, 6.5),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "rank_heatmap.png"),
              width=11, height=6.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Borda Count
# ─────────────────────────────────────────────────────────────────────────────

def plot_borda_count(df_all: pd.DataFrame) -> None:
    """
    Borda voting: for every (DR method, metric) pair, award each model
    (N − rank) points (best model gets N−1, worst gets 0).
    Summing across all pairs yields a scalar aggregate ranking score.
    """
    borda_rows = []
    for dr_label in df_all["dr_label"].unique():
        sub = df_all[df_all["dr_label"] == dr_label].copy()
        for mc in METRIC_COLS:
            ranks = sub[mc].rank(ascending=False, method="min")
            borda = N_MODELS - ranks
            for model_label, score in zip(sub["model_label"], borda):
                borda_rows.append(
                    dict(model_label=model_label, borda=float(score))
                )

    total = (
        pd.DataFrame(borda_rows)
        .groupby("model_label")["borda"]
        .sum()
        .reset_index(name="total_borda")
        .sort_values("total_borda", ascending=True)
    )
    total["model_label"] = pd.Categorical(
        total["model_label"], categories=total["model_label"].tolist()
    )
    total["borda_label"] = total["total_borda"].astype(int).astype(str)

    p = (
        ggplot(total, aes(x="total_borda", y="model_label",
                          fill="model_label"))
        + geom_col(width=0.65, alpha=0.87, show_legend=False)
        + geom_text(
            aes(label="borda_label"),
            ha="left",
            position=position_nudge(x=1.5),
            size=9,
            color="#343A40",
        )
        + scale_fill_manual(values=MODEL_COLORS)
        + scale_x_continuous(
            expand=(0.0, 0.0, 0.08, 0.0),
        )
        + labs(
            title="Borda Count — Aggregate Model Ranking",
            subtitle=(
                "Total Borda points across all metrics and DR methods  |  "
                "higher = consistently better performer"
            ),
            x="Total Borda Score",
            y=None,
        )
        + theme_publication()
        + theme(figure_size=(10, 5.5))
    )
    save_plot(p, os.path.join(OUTDIR, "borda_count.png"),
              width=10, height=5.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Generalisation Gap
# ─────────────────────────────────────────────────────────────────────────────

def plot_generalization_gap(all_data: dict) -> None:
    """
    Scatter: 5-fold CV mean (x) vs held-out test score (y) for four metrics.
    Each point = one model × DR method combination.
    Diagonal = perfect generalisation.  Points above → test > CV (lucky split
    or underfit CV); below → overfit to CV.
    """
    rows = []
    for dr_key, data in all_data.items():
        test_df = data["model_results"]
        cv_df   = data["cv_results"]
        merged  = test_df.merge(cv_df, on="model", suffixes=("", "_cv"))
        for mc, ml in KEY_METRICS.items():
            for _, row in merged.iterrows():
                rows.append(
                    dict(
                        model_label=row["model_label"],
                        dr_label=DR_METHODS[dr_key],
                        metric=ml,
                        cv_mean=float(row[f"{mc}_mean"]),
                        test_value=float(row[mc]),
                    )
                )

    df_gap = pd.DataFrame(rows)
    df_gap["metric"] = pd.Categorical(df_gap["metric"],
                                      categories=list(KEY_METRICS.values()))

    p = (
        ggplot(df_gap, aes(x="cv_mean", y="test_value",
                           color="model_label", shape="dr_label"))
        + geom_abline(slope=1, intercept=0,
                      linetype="dashed", color="#ADB5BD", size=0.7)
        + geom_point(size=4.0, alpha=0.85)
        + facet_wrap("~metric", ncol=2, scales="free")
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_shape_manual(values=DR_SHAPES, name="DR Method")
        + labs(
            title="Generalisation Analysis: CV vs Test Performance",
            subtitle=(
                "Points above diagonal = test ≥ CV  |  "
                "below diagonal = CV optimism / overfitting  |  faceted by metric"
            ),
            x="5-Fold CV Mean Score",
            y="Test Set Score",
        )
        + theme_publication(base_size=9)
        + theme(figure_size=(13, 9))
    )
    save_plot(p, os.path.join(OUTDIR, "generalization_gap.png"),
              width=13, height=9)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Bump Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_bump_chart(df_all: pd.DataFrame) -> None:
    """
    Bump chart: x = metric, y = ordinal rank (1 = best at top).
    Each polyline traces one model's rank trajectory across metrics (averaged
    over DR methods).  Crossing lines reveal metric-dependent rank inversions —
    critical for understanding which model is "best" depends on the clinical
    objective.
    """
    # Average performance over DR methods, then rank
    avg_perf = df_all.groupby("model_label")[METRIC_COLS].mean().reset_index()

    rows = []
    for i, (mc, ml) in enumerate(METRICS.items()):
        ranks = avg_perf[mc].rank(ascending=False, method="min")
        for model_label, rank in zip(avg_perf["model_label"], ranks):
            rows.append(
                dict(
                    model_label=model_label,
                    metric_idx=float(i),
                    metric=ml,
                    rank=float(rank),
                    rank_label=str(int(rank)),
                )
            )

    df_bump = pd.DataFrame(rows)
    df_bump["metric"] = pd.Categorical(df_bump["metric"],
                                       categories=METRIC_LABELS)

    p = (
        ggplot(df_bump, aes(x="metric_idx", y="rank",
                            color="model_label", group="model_label"))
        + geom_line(size=1.3, alpha=0.82)
        + geom_point(size=9, alpha=0.92)
        + geom_text(aes(label="rank_label"),
                    size=7.5, color="white", fontweight="bold")
        + scale_x_continuous(
            breaks=list(range(len(METRIC_LABELS))),
            labels=METRIC_LABELS,
        )
        + scale_y_reverse(
            breaks=list(range(1, N_MODELS + 1)),
            limits=(N_MODELS + 0.5, 0.5),
        )
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + labs(
            title="Bump Chart — Rank Trajectory Across Metrics",
            subtitle=(
                "Rank averaged over all DR methods  |  "
                "1 = best  |  crossing lines = metric-dependent rank inversions"
            ),
            x=None,
            y="Rank (1 = Best)",
        )
        + theme_publication()
        + theme(
            axis_text_x=element_text(angle=22, ha="right"),
            figure_size=(14, 6.5),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "bump_chart.png"),
              width=14, height=6.5)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(all_data: dict) -> None:
    df_all = get_combined_test_df(all_data)
    print("[ranking] Rank heatmap …")
    plot_rank_heatmap(df_all)
    print("[ranking] Borda count …")
    plot_borda_count(df_all)
    print("[ranking] Generalisation gap …")
    plot_generalization_gap(all_data)
    print("[ranking] Bump chart …")
    plot_bump_chart(df_all)


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    run_all(load_all_results(RESULTS_DIR))