from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from data.data_preprocessing import describe_csv_collection, load_target_metadata
from evaluation.plots.config import IMPLEMENTATION_COMPLEXITY, MODEL_FAMILY, RESULTS_DIR


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_frames(results_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    baseline = pd.read_csv(results_dir / "model_comparison.csv")
    baseline["benchmark"] = "baseline"

    sota = pd.read_csv(results_dir / "sota_model_comparison.csv")
    sota["benchmark"] = "sota"

    moe = pd.read_csv(results_dir / "mixture_of_experts_comparison.csv")
    moe["benchmark"] = "mixture_of_experts"

    feature_summary = _read_json(results_dir / "feature_selection_summary.json")
    feature_metrics = _read_json(results_dir / "feature_selection_metrics.json")
    feature_row = pd.DataFrame(
        [
            {
                "model": "feature_selection_logistic",
                "accuracy": feature_metrics["accuracy"],
                "balanced_accuracy": feature_metrics["balanced_accuracy"],
                "sensitivity_recall_ap": feature_metrics["classification_report"]["1"]["recall"],
                "specificity_pd": feature_metrics["classification_report"]["0"]["recall"],
                "precision_ap": feature_metrics["classification_report"]["1"]["precision"],
                "f1_ap": feature_metrics["classification_report"]["1"]["f1-score"],
                "auc_roc": feature_metrics["auc_roc"],
                "benchmark": "feature_selection",
                "selected_feature_count": feature_summary["selected_feature_count"],
            }
        ]
    )

    for frame in [baseline, sota, moe, feature_row]:
        frame["family"] = frame["model"].map(MODEL_FAMILY).fillna("other")
        frame["implementation_complexity"] = frame["model"].map(IMPLEMENTATION_COMPLEXITY).fillna(3.0)

    combined = pd.concat([baseline, sota, moe, feature_row], ignore_index=True, sort=False)
    return {
        "baseline": baseline,
        "sota": sota,
        "moe": moe,
        "feature_selection": feature_row,
        "combined": combined,
    }


def load_dataset_summary() -> dict[str, pd.DataFrame]:
    targets = load_target_metadata()
    table_summary = describe_csv_collection(headers_only=False)

    cohort = (
        targets["target_multiclass"]
        .fillna("Missing")
        .value_counts()
        .rename_axis("class")
        .reset_index(name="n")
    )
    sites = (
        targets["site"]
        .fillna("Missing")
        .value_counts()
        .head(10)
        .rename_axis("site")
        .reset_index(name="n")
    )
    return {"cohort": cohort, "sites": sites, "tables": table_summary}


def load_feature_selection_frames(results_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    selected = pd.read_csv(results_dir / "selected_features.csv")
    summary = _read_json(results_dir / "feature_selection_summary.json")
    composition = pd.DataFrame(
        [
            {"stage": "original", "feature_type": "text", "count": summary["text_column_count"]},
            {"stage": "original", "feature_type": "numeric", "count": summary["numeric_column_count"]},
            {"stage": "selected", "feature_type": "text", "count": summary["selected_text_feature_count"]},
            {"stage": "selected", "feature_type": "numeric", "count": summary["selected_numeric_feature_count"]},
        ]
    )
    return {"selected": selected, "composition": composition}


def _curve_frame_from_predictions(predictions: pd.DataFrame, curve_type: str) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for model_name, frame in predictions.groupby("model"):
        y_true = frame["y_true"].to_numpy()
        y_score = frame["y_score"].to_numpy()
        if curve_type == "roc":
            x, y, _ = roc_curve(y_true, y_score)
            curve = pd.DataFrame({"x": x, "y": y, "model": model_name})
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            curve = pd.DataFrame({"x": recall, "y": precision, "model": model_name})
        records.append(curve)
    return pd.concat(records, ignore_index=True)


def load_prediction_curves(results_dir: Path = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    sota_predictions = pd.read_csv(results_dir / "sota_predictions.csv")
    feature_predictions = pd.read_csv(results_dir / "feature_selection_predictions.csv")
    feature_predictions["model"] = "feature_selection_logistic"
    moe_predictions = pd.read_csv(results_dir / "mixture_of_experts_predictions.csv")

    focus_models = {"easy_ensemble", "catboost", "xgboost_hist", "feature_selection_logistic", "mixture_of_experts_gate"}
    combined_predictions = pd.concat([sota_predictions, feature_predictions, moe_predictions], ignore_index=True)
    combined_predictions = combined_predictions.loc[combined_predictions["model"].isin(focus_models)].copy()

    roc_frame = _curve_frame_from_predictions(combined_predictions, curve_type="roc")
    pr_frame = _curve_frame_from_predictions(combined_predictions, curve_type="pr")
    return {"roc": roc_frame, "pr": pr_frame}
