"""
plot_basic.py
=============
Section 1 – Basic result visualisations.

Plots produced
--------------
1. test_metrics_heatmap.png
   Full metric × model heatmap, faceted by DR method.

2. cv_bars_{dr_key}.png   (one file per DR method)
   Grouped bar charts of 5-fold CV means ± 1 SD, faceted by metric.

3. radar_all_models.png
   Polar / spider chart showing the multi-metric profile of each model,
   values averaged across all DR methods.

4. cleveland_dots.png
   Cleveland dot plot: test score per model per metric, faceted
   DR-method × metric for dense but readable comparison.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    annotate,
    coord_equal,
    element_blank,
    element_text,
    facet_wrap,
    facet_grid,
    geom_col,
    geom_errorbar,
    geom_line,
    geom_point,
    geom_polygon,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_gradient2,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_reverse,
    theme,
    theme_minimal,
)

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DR_METHODS,
    DR_LABEL_ORDER,
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

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs", "basic")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Test Metrics Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_heatmap(all_data: dict) -> None:
    """
    models (rows) × metrics (cols) heatmap, one panel per DR method.
    Red–yellow–green diverging palette; value annotated in each cell.
    """
    rows = []
    for dr_key, data in all_data.items():
        df = data["model_results"]
        for _, row in df.iterrows():
            for mc, ml in METRICS.items():
                rows.append(
                    dict(
                        model_label=row["model_label"],
                        dr_label=DR_METHODS[dr_key],
                        metric=ml,
                        value=float(row[mc]),
                        cell_label=f"{row[mc]:.3f}",
                    )
                )

    df_long = pd.DataFrame(rows)

    # Sort models by mean balanced accuracy across DR methods (top → bottom)
    model_order = (
        df_long[df_long["metric"] == "Balanced Acc."]
        .groupby("model_label")["value"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    df_long["model_label"] = pd.Categorical(
        df_long["model_label"], categories=model_order
    )
    df_long["metric"] = pd.Categorical(
        df_long["metric"], categories=METRIC_LABELS
    )
    df_long["dr_label"] = pd.Categorical(
        df_long["dr_label"], categories=DR_LABEL_ORDER
    )

    p = (
        ggplot(df_long, aes(x="metric", y="model_label", fill="value"))
        + geom_tile(color="white", size=0.6)
        + geom_text(aes(label="cell_label"), size=7.5, color="#333333")
        + facet_wrap("~dr_label", ncol=3)
        + scale_fill_gradient2(
            low="#C62828",
            mid="#FFF9C4",
            high="#1B5E20",
            midpoint=0.5,
            limits=(0, 1),
            name="Score",
        )
        + labs(
            title="Test-Set Performance — All Models & DR Methods",
            subtitle=(
                "Colour encodes metric score  |  green = better  |  "
                "models ordered by mean balanced accuracy"
            ),
            x=None,
            y=None,
        )
        + theme_publication(base_size=9)
        + theme(
            axis_text_x=element_text(angle=35, ha="right", size=7.5),
            figure_size=(18, 11),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "test_metrics_heatmap.png"),
              width=18, height=11)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CV Bar Charts with Error Bars
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_bars(all_data: dict) -> None:
    """
    For each DR method: bar = CV mean, error bar = ±1 SD, faceted by metric.
    """
    model_colors_mapped = MODEL_COLORS  # keys already use label strings

    for dr_key, data in all_data.items():
        cv = data["cv_results"]
        rows = []
        for _, row in cv.iterrows():
            for mc, ml in METRICS.items():
                mean_val = float(row[f"{mc}_mean"])
                std_val  = float(row[f"{mc}_std"])
                rows.append(
                    dict(
                        model_label=row["model_label"],
                        metric=ml,
                        mean=mean_val,
                        ymin=max(0.0, mean_val - std_val),
                        ymax=min(1.0, mean_val + std_val),
                    )
                )

        df_long = pd.DataFrame(rows)
        df_long["metric"] = pd.Categorical(df_long["metric"],
                                           categories=METRIC_LABELS)

        # Sort models by mean balanced accuracy for this DR method
        ba_mean = (
            df_long[df_long["metric"] == "Balanced Acc."]
            .set_index("model_label")["mean"]
            .sort_values(ascending=False)
        )
        df_long["model_label"] = pd.Categorical(
            df_long["model_label"], categories=ba_mean.index.tolist()
        )

        p = (
            ggplot(df_long, aes(x="model_label", y="mean", fill="model_label"))
            + geom_col(width=0.7, alpha=0.87, show_legend=False)
            + geom_errorbar(
                aes(ymin="ymin", ymax="ymax"),
                width=0.28,
                color="#495057",
                size=0.55,
            )
            + facet_wrap("~metric", ncol=4, scales="fixed")
            + scale_fill_manual(values=model_colors_mapped)
            + scale_y_continuous(
                limits=(0, 1.08),
                breaks=[0, 0.25, 0.5, 0.75, 1.0],
            )
            + labs(
                title=f"5-Fold CV Performance — {DR_METHODS[dr_key]}",
                subtitle="Bar = CV mean  |  Error bar = ±1 SD across folds",
                x=None,
                y="Score",
            )
            + theme_publication(base_size=9)
            + theme(
                axis_text_x=element_text(angle=40, ha="right", size=7.5),
                figure_size=(16, 8),
            )
        )
        save_plot(
            p,
            os.path.join(OUTDIR, f"cv_bars_{dr_key}.png"),
            width=16, height=8,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Radar / Spider Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(all_data: dict) -> None:
    """
    Spider chart: each model is a polygon; metric values averaged over all DR
    methods.  Uses plotnine coord_polar(theta='x').
    """
    combined = get_combined_test_df(all_data)
    avg = combined.groupby("model_label")[METRIC_COLS].mean().reset_index()

    n_metrics = len(METRIC_COLS)
    rows = []
    for _, row in avg.iterrows():
        for i, (mc, ml) in enumerate(METRICS.items()):
            rows.append(
                dict(
                    model_label=row["model_label"],
                    metric_idx=i,
                    metric=ml,
                    value=float(row[mc]),
                )
            )
        # Close the polygon by repeating the first metric at index n_metrics
        mc0, ml0 = METRIC_COLS[0], METRIC_LABELS[0]
        rows.append(
            dict(
                model_label=row["model_label"],
                metric_idx=n_metrics,
                metric=ml0,
                value=float(row[mc0]),
            )
        )

    df_radar = pd.DataFrame(rows)

    p = (
        ggplot(
            df_radar,
            aes(
                x="metric_idx",
                y="value",
                color="model_label",
                fill="model_label",
                group="model_label",
            ),
        )
        + geom_polygon(alpha=0.07, size=0.0)
        + geom_line(size=0.85)
        + geom_point(size=2.2)
        + coord_equal()
        + scale_x_continuous(
            breaks=list(range(n_metrics)),
            labels=METRIC_LABELS,
            limits=(-0.5, n_metrics),
        )
        + scale_y_continuous(limits=(0, 1.05), breaks=[0.25, 0.5, 0.75, 1.0])
        + scale_color_manual(values=MODEL_COLORS, name="Model")
        + scale_fill_manual(values=MODEL_COLORS, name="Model")
        + labs(
            title="Multi-Metric Radar Profile",
            subtitle=(
                "Each axis = one performance metric  |  "
                "values averaged across all DR methods"
            ),
            x=None,
            y=None,
        )
        + theme_publication(base_size=10)
        + theme(
            axis_text_y=element_blank(),
            axis_ticks=element_blank(),
            figure_size=(10, 10),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "radar_all_models.png"),
              width=10, height=10)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Cleveland Dot Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_cleveland_dots(all_data: dict) -> None:
    """
    Lollipop / Cleveland dot plot.
    Each row = one model, sorted within each panel by score.
    Facet grid: DR method (rows) × metric (columns).
    """
    rows = []
    for dr_key, data in all_data.items():
        df = data["model_results"]
        for _, row in df.iterrows():
            for mc, ml in METRICS.items():
                rows.append(
                    dict(
                        model_label=row["model_label"],
                        dr_label=DR_METHODS[dr_key],
                        metric=ml,
                        value=float(row[mc]),
                    )
                )
    df_long = pd.DataFrame(rows)
    df_long["metric"] = pd.Categorical(df_long["metric"],
                                       categories=METRIC_LABELS)
    df_long["dr_label"] = pd.Categorical(df_long["dr_label"],
                                         categories=DR_LABEL_ORDER)

    # Within each DR × metric panel, rank model by value so highest is on top
    df_long = df_long.sort_values(["dr_label", "metric", "value"])

    p = (
        ggplot(df_long, aes(x="value", y="model_label", color="model_label"))
        + geom_segment(
            aes(x=0, xend="value", yend="model_label"),
            color="#CED4DA",
            size=0.55,
        )
        + geom_point(size=3.2, alpha=0.9)
        + facet_grid("dr_label ~ metric")
        + scale_color_manual(values=MODEL_COLORS, guide=False)
        + scale_x_continuous(limits=(0, 1.02), breaks=[0, 0.5, 1])
        + labs(
            title="Test Metrics — Cleveland Dot Plot",
            subtitle="Lollipop length encodes score  |  rows = DR method  |  columns = metric",
            x="Score",
            y=None,
        )
        + theme_publication(base_size=7.5)
        + theme(
            axis_text_x=element_text(size=6.5),
            axis_text_y=element_text(size=6.5),
            strip_text_x=element_text(size=6.5),
            strip_text_y=element_text(size=6.5),
            figure_size=(20, 14),
        )
    )
    save_plot(p, os.path.join(OUTDIR, "cleveland_dots.png"),
              width=20, height=14)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(all_data: dict) -> None:
    print("[basic] Test metrics heatmap …")
    plot_test_heatmap(all_data)
    print("[basic] CV bar charts …")
    plot_cv_bars(all_data)
    print("[basic] Radar chart …")
    plot_radar(all_data)
    print("[basic] Cleveland dot plot …")
    plot_cleveland_dots(all_data)


if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    run_all(load_all_results(RESULTS_DIR))