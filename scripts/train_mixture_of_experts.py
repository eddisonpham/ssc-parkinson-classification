from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data_preprocessing import DEFAULT_FEATURE_TABLES, prepare_modeling_dataset
from scripts.train_models import build_preprocessors, compute_metrics, render_markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a non-neural mixture-of-experts pipeline.")
    parser.add_argument("--target", choices=["binary"], default="binary")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--missingness-threshold", type=float, default=0.95)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--gate-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def select_prefix_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    return [column for column in frame.columns if any(column.startswith(prefix) for prefix in prefixes)]


def build_expert_specs(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int) -> list[dict[str, Any]]:
    scale_pos_weight = max(int((y_train == 0).sum()), 1) / max(int((y_train == 1).sum()), 1)
    return [
        {
            "name": "low_burden_expert",
            "prefixes": ["demographic__", "epidemiological__", "ehi__"],
            "model_type": "lightgbm",
            "model": LGBMClassifier(
                boosting_type="gbdt",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                objective="binary",
                random_state=random_seed,
                verbose=-1,
            ),
        },
        {
            "name": "clinical_treatment_expert",
            "prefixes": ["clinical__", "medication__"],
            "model_type": "xgboost",
            "model": XGBClassifier(
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
        },
        {
            "name": "motor_cognition_expert",
            "prefixes": ["moca__", "mds_updrs__", "timed_up_go__", "schwab_and_england__"],
            "model_type": "catboost",
            "model": CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                verbose=False,
                random_seed=random_seed,
            ),
        },
        {
            "name": "questionnaire_expert",
            "prefixes": [
                "apathy_scale__",
                "bai__",
                "bdii__",
                "fatigue_severity_scale__",
                "pdq_8__",
                "pdq_39__",
                "scopa__",
                "parkinson_severity_scale__",
            ],
            "model_type": "balanced_random_forest",
            "model": BalancedRandomForestClassifier(
                n_estimators=400,
                random_state=random_seed,
                replacement=True,
                n_jobs=-1,
            ),
        },
    ]


def fit_expert_pipeline(
    model_type: str,
    model,
    X_train: pd.DataFrame,
) -> Pipeline:
    sparse_pre, dense_pre = build_preprocessors(X_train)
    preprocessor = dense_pre if model_type in {"catboost", "balanced_random_forest"} else sparse_pre
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_gate_frames(
    expert_specs: list[dict[str, Any]],
    X_expert_train: pd.DataFrame,
    X_gate_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_expert_train: pd.Series,
    y_gate_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    gate_train = pd.DataFrame(index=X_gate_train.index)
    gate_test = pd.DataFrame(index=X_test.index)
    expert_rows: list[dict[str, Any]] = []

    for expert in expert_specs:
        expert_columns = select_prefix_columns(X_expert_train, expert["prefixes"])
        if not expert_columns:
            continue

        pipeline = fit_expert_pipeline(
            model_type=expert["model_type"],
            model=expert["model"],
            X_train=X_expert_train[expert_columns],
        )
        pipeline.fit(X_expert_train[expert_columns], y_expert_train)

        gate_prob = pipeline.predict_proba(X_gate_train[expert_columns])[:, 1]
        test_prob = pipeline.predict_proba(X_test[expert_columns])[:, 1]
        gate_pred = (test_prob >= 0.5).astype(int)

        gate_train[expert["name"]] = gate_prob
        gate_test[expert["name"]] = test_prob

        expert_rows.append(
            {
                "model": expert["name"],
                "feature_count": len(expert_columns),
                **compute_metrics(y_test, gate_pred, test_prob, "binary"),
            }
        )

    return gate_train, gate_test, expert_rows


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_modeling_dataset(
        target=args.target,
        tables=DEFAULT_FEATURE_TABLES,
        missingness_threshold=args.missingness_threshold,
        # DR is intentionally disabled for the MoE pipeline: each expert is
        # trained on a disjoint feature-family prefix, and applying a global
        # DR step before splitting by prefix would break that partitioning.
        dr_config=None,
    )

    X_outer_train, X_test, y_outer_train, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=prepared.y,
    )
    X_expert_train, X_gate_train, y_expert_train, y_gate_train = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=args.gate_size,
        random_state=args.random_seed,
        stratify=y_outer_train,
    )

    expert_specs = build_expert_specs(X_expert_train, y_expert_train, args.random_seed)
    gate_train, gate_test, expert_rows = build_gate_frames(
        expert_specs=expert_specs,
        X_expert_train=X_expert_train,
        X_gate_train=X_gate_train,
        X_test=X_test,
        y_expert_train=y_expert_train,
        y_gate_train=y_gate_train,
        y_test=y_test,
    )

    gate_model = LogisticRegression(class_weight="balanced", random_state=args.random_seed)
    gate_model.fit(gate_train, y_gate_train)
    gate_prob = gate_model.predict_proba(gate_test)[:, 1]
    gate_pred = gate_model.predict(gate_test)
    average_prob = gate_test.mean(axis=1).to_numpy()
    average_pred = (average_prob >= 0.5).astype(int)

    comparison_rows = expert_rows + [
        {"model": "expert_average_ensemble", "feature_count": gate_test.shape[1], **compute_metrics(y_test, average_pred, average_prob, "binary")},
        {"model": "mixture_of_experts_gate", "feature_count": gate_test.shape[1], **compute_metrics(y_test, gate_pred, gate_prob, "binary")},
    ]
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["balanced_accuracy", "auc_roc", "accuracy"],
        ascending=[False, False, False],
    )
    prediction_frames = [
        pd.DataFrame(
            {
                "model": "mixture_of_experts_gate",
                "sample_index": X_test.index,
                "y_true": y_test.to_numpy(),
                "y_pred": gate_pred,
                "y_score": gate_prob,
            }
        ),
        pd.DataFrame(
            {
                "model": "expert_average_ensemble",
                "sample_index": X_test.index,
                "y_true": y_test.to_numpy(),
                "y_pred": average_pred,
                "y_score": average_prob,
            }
        ),
    ]

    reports = {
        "mixture_of_experts_gate": classification_report(y_test, gate_pred, output_dict=True, zero_division=0),
        "expert_average_ensemble": classification_report(y_test, average_pred, output_dict=True, zero_division=0),
    }

    comparison_df.to_csv(args.output_dir / "mixture_of_experts_comparison.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "mixture_of_experts_predictions.csv",
        index=False,
    )
    with (args.output_dir / "mixture_of_experts_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    summary = {
        "target": args.target,
        "missingness_threshold": args.missingness_threshold,
        "random_seed": args.random_seed,
        "samples": int(prepared.X.shape[0]),
        "feature_count": int(prepared.X.shape[1]),
        "expert_train_size": int(len(X_expert_train)),
        "gate_train_size": int(len(X_gate_train)),
        "test_size": int(len(X_test)),
        "expert_count": int(gate_test.shape[1]),
        "experts": [
            {
                "name": expert["name"],
                "prefixes": expert["prefixes"],
                "model_type": expert["model_type"],
                "feature_count": int(comparison_df.loc[comparison_df["model"] == expert["name"], "feature_count"].iloc[0]),
            }
            for expert in expert_specs
            if expert["name"] in comparison_df["model"].values
        ],
    }
    with (args.output_dir / "mixture_of_experts_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Non-Neural Mixture Of Experts",
        "",
        "- This pipeline is disjoint from the broad model benchmark.",
        "- Each expert is trained on a separate feature family, then a logistic gate combines expert probabilities.",
        "",
        "## Expert Layout",
        "",
    ]
    for expert in summary["experts"]:
        lines.append(
            f"- `{expert['name']}` uses `{expert['model_type']}` on `{expert['feature_count']}` features from prefixes: `{', '.join(expert['prefixes'])}`"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            render_markdown_table(comparison_df),
            "",
            "## Notes",
            "",
            "- The gate is a classical logistic regression, not a neural network.",
            "- Experts are intentionally disjoint by feature family to improve interpretability and reduce modality leakage.",
        "- This pipeline uses a looser default missingness threshold than the broad benchmark so each expert can retain more modality-specific signal.",
            "- The average ensemble is included as a simpler non-gated baseline for the same experts.",
        ]
    )
    (args.output_dir / "mixture_of_experts_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()