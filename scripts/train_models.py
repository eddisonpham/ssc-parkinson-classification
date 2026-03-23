from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_preprocessing import DEFAULT_FEATURE_TABLES, PreparedDataset, prepare_modeling_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare baseline Parkinson classification models.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default_experiment.yaml")
    parser.add_argument("--target", choices=["binary", "multiclass"], default=None)
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


def build_preprocessors(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer]:
    numeric_columns = [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    sparse_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    dense_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return sparse_preprocessor, dense_preprocessor


def get_models(target: str, sparse_pre: ColumnTransformer, dense_pre: ColumnTransformer) -> list[tuple[str, Pipeline]]:
    if target == "binary":
        return [
            (
                "logistic_regression",
                Pipeline(
                    steps=[
                        ("preprocessor", sparse_pre),
                        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
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
                                n_estimators=400,
                                random_state=42,
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
                                n_estimators=400,
                                random_state=42,
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
                        ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
                    ]
                ),
            ),
            (
                "hist_gradient_boosting",
                Pipeline(
                    steps=[
                        ("preprocessor", dense_pre),
                        ("model", HistGradientBoostingClassifier(random_state=42)),
                    ]
                ),
            ),
        ]

    return [
        (
            "multinomial_logistic_regression",
            Pipeline(
                steps=[
                    ("preprocessor", sparse_pre),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="multinomial")),
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
                            n_estimators=400,
                            random_state=42,
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
                            n_estimators=400,
                            random_state=42,
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
                    ("model", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                steps=[
                    ("preprocessor", dense_pre),
                    ("model", HistGradientBoostingClassifier(random_state=42)),
                ]
            ),
        ),
    ]


def split_dataset(
    prepared: PreparedDataset,
    test_size: float,
    random_seed: int,
    site_holdout: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    X = prepared.X
    y = prepared.y

    if site_holdout:
        site_series = prepared.master.loc[X.index, "site"].fillna("UNKNOWN")
        mask = site_series == site_holdout
        if not mask.any():
            raise ValueError(f"Requested site holdout '{site_holdout}' was not found in the prepared dataset.")
        if (~mask).sum() == 0:
            raise ValueError("Site holdout would leave no training data.")
        X_train, X_test = X.loc[~mask], X.loc[mask]
        y_train, y_test = y.loc[~mask], y.loc[mask]
        split_info = {
            "split_type": "site_holdout",
            "site_holdout": site_holdout,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }
        return X_train, X_test, y_train, y_test, split_info

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    split_info = {
        "split_type": "stratified_random_split",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "test_fraction": test_size,
    }
    return X_train, X_test, y_train, y_test, split_info


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: Any | None,
    target: str,
) -> dict[str, float | int | str | None]:
    metrics: dict[str, float | int | str | None] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }

    if target == "binary":
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        metrics.update(
            {
                "sensitivity_recall_ap": round(float(recall), 4),
                "specificity_pd": round(float(specificity), 4),
                "precision_ap": round(float(precision), 4),
                "f1_ap": round(float(f1), 4),
            }
        )
        if y_score is not None:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        else:
            metrics["auc_roc"] = None
        return metrics

    metrics["macro_f1"] = round(float(f1_score(y_true, y_pred, average="macro")), 4)
    if y_score is not None:
        metrics["macro_auc_ovr"] = round(float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")), 4)
    else:
        metrics["macro_auc_ovr"] = None
    return metrics


def render_markdown_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, separator, *rows])


def main() -> None:
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
    model_specs = get_models(target, sparse_pre, dense_pre)

    comparison_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}

    for model_name, pipeline in model_specs:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline[-1], "predict_proba") else None

        if target == "binary" and probabilities is not None:
            score_input = probabilities[:, 1]
        elif target == "multiclass" and probabilities is not None:
            score_input = probabilities
        else:
            score_input = None

        metrics = compute_metrics(y_test, predictions, score_input, target)
        comparison_rows.append({"model": model_name, **metrics})
        reports[model_name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["balanced_accuracy", "accuracy"],
        ascending=[False, False],
    )

    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    with (output_dir / "classification_reports.json").open("w", encoding="utf-8") as handle:
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
        "dr_applied": prepared.dr_applied,
        "dr_config": dr_config,
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(experiment_summary, handle, indent=2)

    lines = [
        "# Model Comparison",
        "",
        f"- Target: `{target}`",
        f"- Samples: `{prepared.X.shape[0]}`",
        f"- Features after filtering: `{prepared.X.shape[1]}`",
        f"- Missingness threshold: `{missingness_threshold}`",
        f"- Dimensionality reduction: `{'enabled' if prepared.dr_applied else 'disabled'}`",
        f"- Split: `{split_info['split_type']}`",
    ]
    if site_holdout:
        lines.append(f"- Site holdout: `{site_holdout}`")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            render_markdown_table(comparison_df),
            "",
            "## Notes",
            "",
            "- Binary experiments treat `AP` as the positive class because sensitivity to rare atypical parkinsonism is clinically important.",
            "- Direct diagnosis-leakage columns from `clinical.csv` are excluded before modeling.",
            "- Missingness filtering is intentionally conservative because the dataset is highly sparse.",
        ]
    )
    (output_dir / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()