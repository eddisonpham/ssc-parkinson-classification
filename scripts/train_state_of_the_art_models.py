from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_preprocessing import DEFAULT_FEATURE_TABLES, prepare_modeling_dataset
from scripts.train_models import build_preprocessors, compute_metrics, render_markdown_table, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a broader set of state-of-the-art tabular classifiers.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "advanced_experiment.yaml")
    parser.add_argument("--target", choices=["binary"], default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--site-holdout", type=str, default=None)
    parser.add_argument("--missingness-threshold", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--no-dr",
        action="store_true",
        help="Disable dimensionality reduction even if specified in the config.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_scale_pos_weight(y: pd.Series) -> float:
    positive = max(int((y == 1).sum()), 1)
    negative = max(int((y == 0).sum()), 1)
    return negative / positive


def build_model_specs(sparse_pre, dense_pre, random_seed: int, y_train: pd.Series) -> list[tuple[str, Pipeline]]:
    scale_pos_weight = get_scale_pos_weight(y_train)

    return [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_seed)),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=500,
                            random_state=random_seed,
                            class_weight="balanced_subsample",
                            min_samples_leaf=2,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            random_state=random_seed,
                            class_weight="balanced",
                            min_samples_leaf=2,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "svc_rbf",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_seed)),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                steps=[
                    ("preprocessor", dense_pre),
                    ("model", HistGradientBoostingClassifier(random_state=random_seed)),
                ]
            ),
        ),
        (
            "balanced_random_forest",
            Pipeline(
                steps=[
                    ("preprocessor", dense_pre),
                    (
                        "model",
                        BalancedRandomForestClassifier(
                            n_estimators=400,
                            random_state=random_seed,
                            replacement=True,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "easy_ensemble",
            Pipeline(
                steps=[
                    ("preprocessor", dense_pre),
                    ("model", EasyEnsembleClassifier(n_estimators=25, random_state=random_seed, n_jobs=-1)),
                ]
            ),
        ),
        (
            "xgboost_hist",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=350,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=1.0,
                            scale_pos_weight=scale_pos_weight,
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=random_seed,
                            n_jobs=4,
                        ),
                    ),
                ]
            ),
        ),
        (
            "xgboost_deeper",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=500,
                            max_depth=6,
                            learning_rate=0.03,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            min_child_weight=2,
                            reg_lambda=1.0,
                            scale_pos_weight=scale_pos_weight,
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=random_seed,
                            n_jobs=4,
                        ),
                    ),
                ]
            ),
        ),
        (
            "lightgbm_gbdt",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    (
                        "model",
                        LGBMClassifier(
                            boosting_type="gbdt",
                            n_estimators=350,
                            learning_rate=0.05,
                            num_leaves=31,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            class_weight="balanced",
                            objective="binary",
                            random_state=random_seed,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "lightgbm_extra_trees",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    (
                        "model",
                        LGBMClassifier(
                            boosting_type="gbdt",
                            extra_trees=True,
                            n_estimators=400,
                            learning_rate=0.05,
                            num_leaves=63,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            class_weight="balanced",
                            objective="binary",
                            random_state=random_seed,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "catboost",
            Pipeline(
                steps=[
                    ("preprocessor", dense_pre),
                    (
                        "model",
                        CatBoostClassifier(
                            iterations=350,
                            depth=6,
                            learning_rate=0.05,
                            loss_function="Logloss",
                            eval_metric="AUC",
                            auto_class_weights="Balanced",
                            verbose=False,
                            random_seed=random_seed,
                        ),
                    ),
                ]
            ),
        ),
    ]


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    )

    args = parse_args()
    config = load_config(args.config)

    target = args.target or config.get("target", "binary")
    output_dir = args.output_dir
    site_holdout = args.site_holdout or config.get("site_holdout")
    missingness_threshold = (
        args.missingness_threshold
        if args.missingness_threshold is not None
        else config.get("missingness_threshold", 0.65)
    )
    test_size = args.test_size if args.test_size is not None else config.get("test_size", 0.2)
    random_seed = args.random_seed if args.random_seed is not None else config.get("random_seed", 42)
    tables = config.get("tables", DEFAULT_FEATURE_TABLES)
    # Dimensionality reduction: read from config; --no-dr flag overrides to None.
    dr_config = None if args.no_dr else config.get("dimensionality_reduction")

    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_modeling_dataset(
        target=target,
        tables=tables,
        missingness_threshold=missingness_threshold,
        dr_config=dr_config,
    )

    X_train, X_test, y_train, y_test, split_info = split_dataset(
        prepared=prepared,
        test_size=test_size,
        random_seed=random_seed,
        site_holdout=site_holdout,
    )
    sparse_pre, dense_pre = build_preprocessors(prepared.X)
    model_specs = build_model_specs(sparse_pre=sparse_pre, dense_pre=dense_pre, random_seed=random_seed, y_train=y_train)

    comparison_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []

    for model_name, pipeline in model_specs:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline[-1], "predict_proba") else None
        score_input = probabilities[:, 1] if probabilities is not None else None

        metrics = compute_metrics(y_test, predictions, score_input, target)
        comparison_rows.append({"model": model_name, **metrics})
        reports[model_name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "sample_index": X_test.index,
                    "y_true": y_test.to_numpy(),
                    "y_pred": predictions,
                    "y_score": score_input if score_input is not None else pd.NA,
                }
            )
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["balanced_accuracy", "auc_roc", "accuracy"],
        ascending=[False, False, False],
    )

    comparison_df.to_csv(output_dir / "sota_model_comparison.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "sota_predictions.csv", index=False)
    with (output_dir / "sota_classification_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    experiment_summary = {
        "target": target,
        "tables": tables,
        "missingness_threshold": missingness_threshold,
        "random_seed": random_seed,
        "feature_count": int(prepared.X.shape[1]),
        "sample_count": int(prepared.X.shape[0]),
        "class_distribution": {str(key): int(value) for key, value in prepared.y.value_counts().to_dict().items()},
        "split": split_info,
        "model_count": len(model_specs),
        "dr_applied": prepared.dr_applied,
        "dr_config": dr_config,
    }
    with (output_dir / "sota_experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    lines = [
        "# State-Of-The-Art Tabular Benchmark",
        "",
        f"- Target: `{target}`",
        f"- Samples: `{prepared.X.shape[0]}`",
        f"- Features after filtering: `{prepared.X.shape[1]}`",
        f"- Missingness threshold: `{missingness_threshold}`",
        f"- Dimensionality reduction: `{'enabled' if prepared.dr_applied else 'disabled'}`",
        f"- Split: `{split_info['split_type']}`",
        f"- Models benchmarked: `{len(model_specs)}`",
        "",
        "## Metrics",
        "",
        render_markdown_table(comparison_df),
        "",
        "## Notes",
        "",
        "- This benchmark broadens the original baseline with modern gradient-boosting and imbalance-aware ensembles.",
        "- The added advanced models include `XGBoost`, `LightGBM`, `CatBoost`, `BalancedRandomForest`, and `EasyEnsemble`.",
        "- Ranking prioritizes balanced accuracy first because the AP class is rare and clinically important.",
    ]
    (output_dir / "sota_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()