from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    facet_wrap,
    geom_abline,
    geom_bar,
    geom_col,
    geom_line,
    geom_path,
    geom_point,
    geom_segment,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.plots.config import MODEL_FAMILY, OUTPUT_DIR, PALETTE
from evaluation.plots.data_loading import (
    load_benchmark_frames,
    load_dataset_summary,
    load_feature_selection_frames,
    load_prediction_curves,
)
from evaluation.plots.plotting import publication_theme, save_plot, wrap_label


def _prep_model_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["model_label"] = frame["model"].map(lambda value: wrap_label(value, width=18))
    return frame


def generate_all_plots(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in output_dir.glob("*.png"):
        old_plot.unlink()

    benchmark_frames = load_benchmark_frames()
    dataset_summary = load_dataset_summary()
    feature_frames = load_feature_selection_frames()
    prediction_curves = load_prediction_curves()

    saved_paths: list[Path] = []

    cohort = dataset_summary["cohort"]
    plot = (
        ggplot(cohort, aes(x="class", y="n", fill="class"))
        + geom_col(show_legend=False)
        + labs(
            title="Cohort Class Balance",
            subtitle="PD dominates the cohort, making AP-sensitive evaluation essential.",
            x="Class",
            y="Participants",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "01_cohort_class_balance.png", width=7, height=5))

    sites = dataset_summary["sites"].copy()
    sites["site_label"] = sites["site"].map(lambda value: wrap_label(value, width=20))
    plot = (
        ggplot(sites, aes(x="site_label", y="n"))
        + geom_col(fill="#457b9d")
        + coord_flip()
        + labs(
            title="Top Recruiting Sites",
            subtitle="The cohort is multi-centric, but site representation is uneven.",
            x="Site",
            y="Participants",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "02_site_distribution_top10.png", width=9, height=6))

    tables = dataset_summary["tables"].copy()
    top_sparse = tables.sort_values("cell_missing_pct", ascending=False).head(12)
    top_sparse["filename_label"] = top_sparse["filename"].map(lambda value: wrap_label(value, width=18))
    plot = (
        ggplot(top_sparse, aes(x="filename_label", y="cell_missing_pct", fill="likely_implementation_level"))
        + geom_col(show_legend=True)
        + coord_flip()
        + labs(
            title="Most Sparse Tables",
            subtitle="High missingness is a first-order modeling challenge in the C-OPN snapshot.",
            x="Table",
            y="Cell missingness (%)",
            fill="Likely implementation level",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "03_table_sparsity_top12.png", width=10, height=7))

    plot = (
        ggplot(tables, aes(x="column_count", y="cell_missing_pct", color="likely_implementation_level"))
        + geom_point(size=3, alpha=0.8)
        + labs(
            title="Table Width Versus Missingness",
            subtitle="Large instruments tend to be especially sparse in this repository snapshot.",
            x="Columns in table",
            y="Cell missingness (%)",
            color="Likely implementation level",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "04_table_width_vs_missingness.png", width=9, height=6))

    combined = benchmark_frames["combined"].copy()
    combined = combined.loc[combined["benchmark"].isin(["baseline", "sota", "feature_selection"])].copy()
    combined = combined.sort_values("balanced_accuracy", ascending=False)
    combined["model_label"] = combined["model"].map(lambda value: wrap_label(value, width=18))
    plot = (
        ggplot(combined, aes(x="model_label", y="balanced_accuracy", fill="benchmark"))
        + geom_col(position="dodge")
        + coord_flip()
        + labs(
            title="Balanced Accuracy Across Main Pipelines",
            subtitle="Performance ranking prioritizes AP-sensitive discrimination over raw accuracy alone.",
            x="Model",
            y="Balanced accuracy",
            fill="Benchmark",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "05_balanced_accuracy_across_pipelines.png", width=10, height=8))

    sota = benchmark_frames["sota"].copy()
    sota = _prep_model_labels(sota)
    plot = (
        ggplot(sota, aes(x="specificity_pd", y="sensitivity_recall_ap", color="family"))
        + geom_point(size=3)
        + geom_text(aes(label="model_label"), size=7, nudge_y=0.01, show_legend=False)
        + labs(
            title="Clinical Trade-Off: AP Recall Versus PD Specificity",
            subtitle="Atypical parkinsonism is rare, so the best models must balance missed AP cases against false alarms.",
            x="PD specificity",
            y="AP sensitivity / recall",
            color="Model family",
        )
        + scale_color_manual(values=PALETTE)
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "06_recall_vs_specificity_tradeoff.png", width=11, height=7))

    plot = (
        ggplot(sota, aes(x="auc_roc", y="balanced_accuracy", color="family"))
        + geom_point(size=3)
        + geom_text(aes(label="model_label"), size=7, nudge_y=0.008, show_legend=False)
        + labs(
            title="AUC Versus Balanced Accuracy",
            subtitle="High AUC does not always translate into strong AP-sensitive balanced accuracy.",
            x="AUC-ROC",
            y="Balanced accuracy",
            color="Model family",
        )
        + scale_color_manual(values=PALETTE)
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "07_auc_vs_balanced_accuracy.png", width=11, height=7))

    heatmap = sota.melt(
        id_vars=["model"],
        value_vars=["balanced_accuracy", "sensitivity_recall_ap", "specificity_pd", "auc_roc", "precision_ap"],
        var_name="metric",
        value_name="value",
    )
    heatmap["model_label"] = heatmap["model"].map(lambda value: wrap_label(value, width=18))
    plot = (
        ggplot(heatmap, aes(x="metric", y="model_label", fill="value"))
        + geom_tile()
        + labs(
            title="Metric Heatmap For Advanced Models",
            subtitle="Different model families optimize different parts of the clinical objective.",
            x="Metric",
            y="Model",
            fill="Score",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "08_sota_metric_heatmap.png", width=10, height=7))

    complexity = sota.copy()
    plot = (
        ggplot(complexity, aes(x="implementation_complexity", y="balanced_accuracy", color="family"))
        + geom_point(size=3)
        + geom_text(aes(label="model_label"), size=7, nudge_y=0.008, show_legend=False)
        + labs(
            title="Implementation Complexity Versus Performance",
            subtitle="Literature emphasizes deployability and actionability, not just discrimination.",
            x="Estimated implementation complexity",
            y="Balanced accuracy",
            color="Model family",
        )
        + scale_color_manual(values=PALETTE)
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "09_implementation_vs_performance.png", width=11, height=7))

    composition = feature_frames["composition"].copy()
    plot = (
        ggplot(composition, aes(x="stage", y="count", fill="feature_type"))
        + geom_col(position="dodge")
        + labs(
            title="Text-Aware Feature Selection Composition",
            subtitle="The sparse selector retains both numeric and text-derived signals.",
            x="Stage",
            y="Feature count",
            fill="Feature type",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "10_feature_selection_composition.png", width=8, height=5))

    selected = feature_frames["selected"].head(20).copy()
    selected["feature_label"] = selected["feature"].map(lambda value: wrap_label(value, width=26))
    plot = (
        ggplot(selected, aes(x="importance", y="feature_label", color="feature_type"))
        + geom_segment(aes(x=0, xend="importance", y="feature_label", yend="feature_label"), size=1)
        + geom_point(size=3)
        + labs(
            title="Top Selected Features",
            subtitle="Text-derived tokens remain prominent after sparse feature selection.",
            x="Selection importance",
            y="Feature",
            color="Feature type",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "11_top_selected_features.png", width=12, height=8))

    moe = benchmark_frames["moe"].copy()
    moe = _prep_model_labels(moe)
    plot = (
        ggplot(moe, aes(x="model_label", y="balanced_accuracy", fill="model"))
        + geom_col(show_legend=False)
        + coord_flip()
        + labs(
            title="Non-Neural Mixture-Of-Experts Comparison",
            subtitle="Disjoint experts are interpretable, but the gated ensemble still trails the best single-model benchmark.",
            x="Mixture component",
            y="Balanced accuracy",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "12_moe_component_performance.png", width=10, height=6))

    plot = (
        ggplot(moe, aes(x="feature_count", y="balanced_accuracy"))
        + geom_point(size=3, color="#bc4749")
        + geom_text(aes(label="model_label"), size=7, nudge_y=0.008)
        + labs(
            title="Expert Feature Count Versus Performance",
            subtitle="More modality-specific features do not automatically improve rare-class discrimination.",
            x="Feature count",
            y="Balanced accuracy",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "13_moe_feature_count_vs_performance.png", width=10, height=6))

    roc = prediction_curves["roc"].copy()
    roc["family"] = roc["model"].map(lambda value: "expert_ensemble" if "mixture" in value else MODEL_FAMILY.get(value, "other"))
    plot = (
        ggplot(roc, aes(x="x", y="y", color="model"))
        + geom_path(size=1.1)
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="#9ca3af")
        + labs(
            title="ROC Curves For Leading Pipelines",
            subtitle="The evaluation layer compares the best advanced, text-aware, and expert-based pipelines on the same holdout split.",
            x="False positive rate",
            y="True positive rate",
            color="Model",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "14_roc_curves_leading_models.png", width=9, height=7))

    pr = prediction_curves["pr"].copy()
    plot = (
        ggplot(pr, aes(x="x", y="y", color="model"))
        + geom_path(size=1.1)
        + labs(
            title="Precision-Recall Curves For Leading Pipelines",
            subtitle="PR curves are especially relevant because AP is the rare class of interest.",
            x="Recall",
            y="Precision",
            color="Model",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "15_precision_recall_curves_leading_models.png", width=9, height=7))

    family_summary = (
        benchmark_frames["combined"]
        .groupby("family", as_index=False)
        .agg(mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_auc=("auc_roc", "mean"))
    )
    plot = (
        ggplot(family_summary, aes(x="family", y="mean_balanced_accuracy", fill="family"))
        + geom_col(show_legend=False)
        + coord_flip()
        + scale_fill_manual(values=PALETTE)
        + labs(
            title="Average Performance By Model Family",
            subtitle="Imbalance-aware ensembles and boosting dominate the current search space.",
            x="Model family",
            y="Mean balanced accuracy",
        )
        + publication_theme()
    )
    saved_paths.append(save_plot(plot, output_dir, "16_family_average_performance.png", width=9, height=6))

    return saved_paths


def main() -> None:
    paths = generate_all_plots()
    print("Generated plots:")
    for path in paths:
        print(path.name)


if __name__ == "__main__":
    main()
