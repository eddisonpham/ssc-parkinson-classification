from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_preprocessing import PreparedDataset, prepare_modeling_dataset


@dataclass
class FeatureSelectionResult:
    summary: dict[str, Any]
    selected_features: pd.DataFrame
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-aware feature selection for C-OPN.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default_experiment.yaml",
        help="Path to YAML experiment config (for dimensionality_reduction key).",
    )
    parser.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--missingness-threshold", type=float, default=0.65)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-text-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--selector-c", type=float, default=0.5)
    parser.add_argument("--top-k-fallback", type=int, default=200)
    parser.add_argument(
        "--no-dr",
        action="store_true",
        help="Disable dimensionality reduction even if specified in the config.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    """Load YAML experiment config, returning empty dict if the file is absent."""
    if not path.exists():
        return {}
    import yaml  # lazy import; yaml is already a dependency of other scripts
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _tokenize_value(value: object) -> str:
    cleaned = re.sub(r"\s+", "_", str(value).strip().lower())
    cleaned = re.sub(r"[^a-z0-9_./-]+", "", cleaned)
    return cleaned


def build_text_documents(frame: pd.DataFrame, text_columns: list[str]) -> pd.Series:
    if not text_columns:
        return pd.Series([""] * len(frame), index=frame.index)

    documents: list[str] = []
    subset = frame[text_columns].copy()
    for _, row in subset.iterrows():
        tokens: list[str] = []
        for column, value in row.items():
            if pd.isna(value):
                continue
            value_token = _tokenize_value(value)
            if not value_token:
                continue
            tokens.append(f"{column}={value_token}")
        documents.append(" ".join(tokens))
    return pd.Series(documents, index=frame.index)


def transform_numeric_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if not numeric_columns:
        empty_train = sparse.csr_matrix((len(X_train), 0))
        empty_test = sparse.csr_matrix((len(X_test), 0))
        return empty_train, empty_test

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_num = scaler.fit_transform(imputer.fit_transform(X_train[numeric_columns]))
    test_num = scaler.transform(imputer.transform(X_test[numeric_columns]))
    return sparse.csr_matrix(train_num), sparse.csr_matrix(test_num)


def build_feature_space(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_text_features: int,
    min_df: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, list[str], list[str]]:
    numeric_columns = [column for column in X_train.columns if pd.api.types.is_numeric_dtype(X_train[column])]
    text_columns = [column for column in X_train.columns if column not in numeric_columns]

    train_documents = build_text_documents(X_train, text_columns)
    test_documents = build_text_documents(X_test, text_columns)

    if text_columns and train_documents.str.strip().ne("").any():
        vectorizer = TfidfVectorizer(max_features=max_text_features, min_df=min_df, ngram_range=(1, 2))
        text_train = vectorizer.fit_transform(train_documents)
        text_test = vectorizer.transform(test_documents)
        text_feature_names = np.array([f"text__{name}" for name in vectorizer.get_feature_names_out()])
    else:
        text_train = sparse.csr_matrix((len(X_train), 0))
        text_test = sparse.csr_matrix((len(X_test), 0))
        text_feature_names = np.array([], dtype=str)

    numeric_train, numeric_test = transform_numeric_features(X_train, X_test, numeric_columns)
    numeric_feature_names = np.array([f"numeric__{name}" for name in numeric_columns])

    train_matrix = sparse.hstack([text_train, numeric_train], format="csr")
    test_matrix = sparse.hstack([text_test, numeric_test], format="csr")
    feature_names = np.concatenate([text_feature_names, numeric_feature_names])
    return train_matrix, test_matrix, feature_names, text_columns, numeric_columns


def fit_selector(
    X_train_matrix: sparse.csr_matrix,
    y_train: pd.Series,
    selector_c: float,
    top_k_fallback: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    selector = LinearSVC(
        penalty="l1",
        dual=False,
        class_weight="balanced",
        C=selector_c,
        max_iter=8000,
        random_state=random_seed,
    )
    selector.fit(X_train_matrix, y_train)

    coef_matrix = np.abs(selector.coef_)
    feature_strength = coef_matrix.max(axis=0)
    selected_mask = feature_strength > 1e-8

    if not selected_mask.any():
        top_k = min(top_k_fallback, len(feature_strength))
        top_indices = np.argsort(feature_strength)[::-1][:top_k]
        selected_mask = np.zeros(len(feature_strength), dtype=bool)
        selected_mask[top_indices] = True

    return selected_mask, feature_strength


def evaluate_selected_model(
    X_train_selected: sparse.csr_matrix,
    X_test_selected: sparse.csr_matrix,
    y_train: pd.Series,
    y_test: pd.Series,
    target: str,
    random_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=4000,
        random_state=random_seed,
    )
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    probabilities = model.predict_proba(X_test_selected) if hasattr(model, "predict_proba") else None

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, predictions)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    if target == "binary" and probabilities is not None:
        metrics["auc_roc"] = round(float(roc_auc_score(y_test, probabilities[:, 1])), 4)
    elif target == "multiclass" and probabilities is not None:
        metrics["macro_auc_ovr"] = round(float(roc_auc_score(y_test, probabilities, multi_class="ovr", average="macro")), 4)

    prediction_frame = pd.DataFrame(
        {
            "sample_index": y_test.index,
            "y_true": y_test.to_numpy(),
            "y_pred": predictions,
            "y_score": probabilities[:, 1] if probabilities is not None and target == "binary" else np.nan,
        }
    )
    return metrics, prediction_frame


def run_feature_selection_experiment(
    prepared: PreparedDataset,
    target: str = "binary",
    test_size: float = 0.2,
    random_seed: int = 42,
    max_text_features: int = 5000,
    min_df: int = 5,
    selector_c: float = 0.5,
    top_k_fallback: int = 200,
) -> FeatureSelectionResult:
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=test_size,
        random_state=random_seed,
        stratify=prepared.y,
    )

    train_matrix, test_matrix, feature_names, text_columns, numeric_columns = build_feature_space(
        X_train=X_train,
        X_test=X_test,
        max_text_features=max_text_features,
        min_df=min_df,
    )

    selected_mask, feature_strength = fit_selector(
        X_train_matrix=train_matrix,
        y_train=y_train,
        selector_c=selector_c,
        top_k_fallback=top_k_fallback,
        random_seed=random_seed,
    )

    selected_feature_names = feature_names[selected_mask]
    selected_strength = feature_strength[selected_mask]
    selected_features = pd.DataFrame(
        {
            "feature": selected_feature_names,
            "importance": selected_strength,
            "feature_type": np.where(pd.Series(selected_feature_names).str.startswith("text__"), "text", "numeric"),
        }
    ).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)

    metrics, predictions = evaluate_selected_model(
        X_train_selected=train_matrix[:, selected_mask],
        X_test_selected=test_matrix[:, selected_mask],
        y_train=y_train,
        y_test=y_test,
        target=target,
        random_seed=random_seed,
    )

    summary = {
        "target": target,
        "samples": int(prepared.X.shape[0]),
        "original_feature_count": int(prepared.X.shape[1]),
        "text_column_count": len(text_columns),
        "numeric_column_count": len(numeric_columns),
        "tfidf_feature_count": int(len([name for name in feature_names if name.startswith('text__')])),
        "combined_feature_count": int(len(feature_names)),
        "selected_feature_count": int(selected_mask.sum()),
        "selected_text_feature_count": int((selected_features["feature_type"] == "text").sum()),
        "selected_numeric_feature_count": int((selected_features["feature_type"] == "numeric").sum()),
        "max_text_features": max_text_features,
        "min_df": min_df,
        "selector_c": selector_c,
        "random_seed": random_seed,
    }

    return FeatureSelectionResult(
        summary=summary,
        selected_features=selected_features,
        metrics=metrics,
        predictions=predictions,
    )


def render_markdown_summary(result: FeatureSelectionResult) -> str:
    top_features = result.selected_features.head(25).copy()
    top_features["importance"] = top_features["importance"].round(6)
    header = "| feature | importance | feature_type |"
    separator = "| --- | --- | --- |"
    rows = [
        f"| {row.feature} | {row.importance} | {row.feature_type} |"
        for row in top_features.itertuples(index=False)
    ]

    lines = [
        "# Text-Aware Feature Selection",
        "",
        f"- Target: `{result.summary['target']}`",
        f"- Samples: `{result.summary['samples']}`",
        f"- Original tabular features: `{result.summary['original_feature_count']}`",
        f"- Text columns combined into row documents: `{result.summary['text_column_count']}`",
        f"- Numeric columns kept separately: `{result.summary['numeric_column_count']}`",
        f"- TF-IDF feature count: `{result.summary['tfidf_feature_count']}`",
        f"- Combined feature count before selection: `{result.summary['combined_feature_count']}`",
        f"- Selected feature count: `{result.summary['selected_feature_count']}`",
        f"- Selected text features: `{result.summary['selected_text_feature_count']}`",
        f"- Selected numeric features: `{result.summary['selected_numeric_feature_count']}`",
        "",
        "## Selected-model performance",
        "",
        f"- Accuracy: `{result.metrics['accuracy']}`",
        f"- Balanced accuracy: `{result.metrics['balanced_accuracy']}`",
    ]

    if "auc_roc" in result.metrics:
        lines.append(f"- AUC-ROC: `{result.metrics['auc_roc']}`")
    if "macro_auc_ovr" in result.metrics:
        lines.append(f"- Macro AUC-OVR: `{result.metrics['macro_auc_ovr']}`")

    lines.extend(
        [
            "",
            "## Top selected features",
            "",
            "\n".join([header, separator, *rows]),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config for dimensionality_reduction key.
    config = _load_config(args.config)
    dr_config = None if args.no_dr else config.get("dimensionality_reduction")

    prepared = prepare_modeling_dataset(
        target=args.target,
        missingness_threshold=args.missingness_threshold,
        dr_config=dr_config,
    )
    result = run_feature_selection_experiment(
        prepared=prepared,
        target=args.target,
        test_size=args.test_size,
        random_seed=args.random_seed,
        max_text_features=args.max_text_features,
        min_df=args.min_df,
        selector_c=args.selector_c,
        top_k_fallback=args.top_k_fallback,
    )

    result.selected_features.to_csv(args.output_dir / "selected_features.csv", index=False)
    result.predictions.to_csv(args.output_dir / "feature_selection_predictions.csv", index=False)
    with (args.output_dir / "feature_selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, default=float)
    with (args.output_dir / "feature_selection_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, default=float)
    (args.output_dir / "feature_selection.md").write_text(render_markdown_summary(result), encoding="utf-8")

    print(json.dumps(result.summary, indent=2))
    print(result.selected_features.head(20).to_string(index=False))


if __name__ == "__main__":
    main()