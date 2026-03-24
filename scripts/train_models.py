"""
train_models.py
================
Train and evaluate classifiers for C-OPN PD vs AP classification.

Pipeline
--------
1. Load scale-level features          (data_preprocessing.py)
2. Stratified 80/20 train/test split
3. Fit preprocessor (impute + scale + DR) on TRAIN only
4. Save preprocessed CSVs (pre-DR and post-DR) to results/{method}/
5. Optuna HPO (TPE sampler) per model on TRAIN, 5-fold stratified CV
6. 5-fold CV with tuned hyperparams → honest performance estimate
7. Final fit on full TRAIN, evaluate on held-out TEST
8. Save all results to results/{method}/

Models
------
8 classifiers covering linear / kernel / tree / ensemble / neural axes,
all configured for extreme class imbalance (~6% AP minority):

  1. LogisticRegression (ElasticNet)  — linear baseline; best for small n
  2. SVM (RBF kernel)                 — strong with scaled PCA/FAMD output
  3. RandomForest (balanced)          — non-linear; handles interactions
  4. BalancedRandomForest             — imbalanced-learn variant; better recall
  5. LightGBM (scale_pos_weight)      — best single gradient boosting model
  6. XGBoost (scale_pos_weight)       — complementary boosting perspective
  7. MLP (sklearn)                    — neural baseline, no extra deps
  8. EasyEnsemble                     — specialist for extreme imbalance

Hyperparameter tuning (Optuna)
-------------------------------
TPE (Tree-structured Parzen Estimator) sampler with MedianPruner.
Objective: maximise F1 score (AP class) on 5-fold stratified CV.
Tuning happens AFTER DR is fit, on the DR-reduced X_train.
This is correct: DR is data preprocessing (like scaling), not part of the
model; HP search optimises the model given fixed preprocessing.

Usage
-----
  python train_models.py --dr-method famd_hellinger --n-trials 50
  python train_models.py --dr-method pca --skip-tuning   # quick run
  python train_models.py --help
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*feature names.*")

RESULTS_DIR = Path("results")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
N_CV_FOLDS   = 5


# ---------------------------------------------------------------------------
# Model registry: name → (build_fn, suggest_fn)
# ---------------------------------------------------------------------------
# build_fn(params)  → fitted-able sklearn estimator
# suggest_fn(trial) → estimator with Optuna-suggested hyperparams
# ---------------------------------------------------------------------------

def _build_logistic(params: dict):
    from sklearn.linear_model import LogisticRegression
    # To future-proof, do NOT set penalty, just set l1_ratio (will default to 'elasticnet' if l1_ratio is not None)
    l1_ratio = params.get("l1_ratio", 0.5)
    C = params.get("C", 0.1)
    return LogisticRegression(
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )

def _suggest_logistic(trial):
    return _build_logistic({
        "C":        trial.suggest_float("C", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    })


def _build_svm(params: dict):
    from sklearn.svm import SVC
    return SVC(
        kernel="rbf",
        C=params.get("C", 1.0),
        gamma=params.get("gamma", "scale"),
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )

def _suggest_svm(trial):
    return _build_svm({
        "C":     trial.suggest_float("C", 1e-3, 100.0, log=True),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto",
                                                      1e-4, 1e-3, 1e-2, 1e-1]),
    })


def _build_random_forest(params: dict):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params.get("min_samples_leaf", 3),
        max_features=params.get("max_features", "sqrt"),
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def _suggest_random_forest(trial):
    return _build_random_forest({
        "n_estimators":    trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth":       trial.suggest_int("max_depth", 3, 25),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":    trial.suggest_categorical("max_features",
                                                     ["sqrt", "log2", 0.3, 0.5]),
    })


def _build_balanced_rf(params: dict):
    from imblearn.ensemble import BalancedRandomForestClassifier
    return BalancedRandomForestClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        max_features=params.get("max_features", "sqrt"),
        sampling_strategy="auto",
        replacement=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def _suggest_balanced_rf(trial):
    return _build_balanced_rf({
        "n_estimators":    trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth":       trial.suggest_int("max_depth", 3, 25),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features":    trial.suggest_categorical("max_features",
                                                     ["sqrt", "log2", 0.3, 0.5]),
    })


def _build_lightgbm(params: dict, scale_pos_weight: float = 1.0):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_child_samples", 10),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

def _suggest_lightgbm(trial, scale_pos_weight: float = 1.0):
    return _build_lightgbm({
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 60),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }, scale_pos_weight=scale_pos_weight)


def _build_xgboost(params: dict, scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 400),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 5),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 3),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

def _suggest_xgboost(trial, scale_pos_weight: float = 1.0):
    return _build_xgboost({
        "n_estimators":   trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":      trial.suggest_int("max_depth", 3, 10),
        "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }, scale_pos_weight=scale_pos_weight)


def _build_mlp(params: dict):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (128, 64)),
        activation=params.get("activation", "relu"),
        alpha=params.get("alpha", 1e-4),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        batch_size=params.get("batch_size", 64),
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )

def _suggest_mlp(trial):
    return _build_mlp({
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes",
            [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
        ),
        "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha":              trial.suggest_float("alpha", 1e-6, 0.1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True),
        "batch_size":         trial.suggest_categorical("batch_size", [32, 64, 128]),
    })


def _build_easy_ensemble(params: dict):
    from imblearn.ensemble import EasyEnsembleClassifier
    return EasyEnsembleClassifier(
        n_estimators=params.get("n_estimators", 30),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def _suggest_easy_ensemble(trial):
    return _build_easy_ensemble({
        "n_estimators": trial.suggest_int("n_estimators", 10, 60, step=5),
    })


# Default scale_pos_weight placeholder; filled at runtime from y_train
_DEFAULT_SPW = 1.0

MODEL_REGISTRY: dict[str, dict] = {
    "logistic_regression": {
        "build":   lambda p: _build_logistic(p),
        "suggest": lambda t: _suggest_logistic(t),
        "n_trials_default": 60,
    },
    "svm": {
        "build":   lambda p: _build_svm(p),
        "suggest": lambda t: _suggest_svm(t),
        "n_trials_default": 60,
    },
    "random_forest": {
        "build":   lambda p: _build_random_forest(p),
        "suggest": lambda t: _suggest_random_forest(t),
        "n_trials_default": 50,
    },
    "balanced_random_forest": {
        "build":   lambda p: _build_balanced_rf(p),
        "suggest": lambda t: _suggest_balanced_rf(t),
        "n_trials_default": 50,
    },
    "lightgbm": {
        "build":   lambda p: _build_lightgbm(p),
        "suggest": lambda t: _suggest_lightgbm(t),
        "n_trials_default": 80,
    },
    "xgboost": {
        "build":   lambda p: _build_xgboost(p),
        "suggest": lambda t: _suggest_xgboost(t),
        "n_trials_default": 80,
    },
    "mlp": {
        "build":   lambda p: _build_mlp(p),
        "suggest": lambda t: _suggest_mlp(t),
        "n_trials_default": 50,
    },
    "easy_ensemble": {
        "build":   lambda p: _build_easy_ensemble(p),
        "suggest": lambda t: _suggest_easy_ensemble(t),
        "n_trials_default": 30,
    },
}


def build_default_models(y_train: pd.Series) -> dict[str, object]:
    """Instantiate all models with default (pre-tuning) parameters."""
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    return {
        "logistic_regression":    _build_logistic({}),
        "svm":                    _build_svm({}),
        "random_forest":          _build_random_forest({}),
        "balanced_random_forest": _build_balanced_rf({}),
        "lightgbm":               _build_lightgbm({}, scale_pos_weight=spw),
        "xgboost":                _build_xgboost({}, scale_pos_weight=spw),
        "mlp":                    _build_mlp({}),
        "easy_ensemble":          _build_easy_ensemble({}),
    }


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int | None = None,
    timeout: int | None = None,
) -> tuple[dict, float]:
    """
    Run Optuna TPE optimisation for one model.

    Uses 5-fold stratified CV on X_train (already DR-reduced).
    Sampler: TPE (state-of-the-art Bayesian optimisation).
    Pruner: MedianPruner (stops unpromising trials early).
    Objective: maximise F1 score for AP class (label=1).

    Returns
    -------
    (best_params, best_cv_score)
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    entry = MODEL_REGISTRY[name]
    n_trials = n_trials or entry["n_trials_default"]
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    def objective(trial) -> float:
        # Inject scale_pos_weight for boosting models
        if name in ("lightgbm", "xgboost"):
            model = (
                _suggest_lightgbm(trial, scale_pos_weight=spw)
                if name == "lightgbm"
                else _suggest_xgboost(trial, scale_pos_weight=spw)
            )
        else:
            model = entry["suggest"](trial)

        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        try:
            scores = cross_val_score(
                model, X_train, y_train,
                cv=skf,
                scoring="f1",    # maximize F1 score for AP class (label=1)
                n_jobs=1,    # outer parallelism via Optuna
                error_score=0.0,
            )
            return float(scores.mean())
        except Exception:
            return 0.0

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=10,          # random exploration before TPE kicks in
        multivariate=True,            # use multivariate TPE for correlated params
        consider_magic_clip=True,     # clip extreme proposal values
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return study.best_params, study.best_value


def build_tuned_model(name: str, best_params: dict, y_train: pd.Series) -> object:
    """Instantiate a model using Optuna's best_params dict."""
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)

    if name == "logistic_regression":
        return _build_logistic(best_params)
    elif name == "svm":
        return _build_svm(best_params)
    elif name == "random_forest":
        return _build_random_forest(best_params)
    elif name == "balanced_random_forest":
        return _build_balanced_rf(best_params)
    elif name == "lightgbm":
        return _build_lightgbm(best_params, scale_pos_weight=spw)
    elif name == "xgboost":
        return _build_xgboost(best_params, scale_pos_weight=spw)
    elif name == "mlp":
        return _build_mlp(best_params)
    elif name == "easy_ensemble":
        return _build_easy_ensemble(best_params)
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, f1_score,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ap_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ap_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_ap = (
        2 * ap_precision * ap_recall / (ap_precision + ap_recall)
        if (ap_precision + ap_recall) > 0 else 0.0
    )
    # For completeness, also compute sklearn's f1_score(y_true, y_pred)
    try:
        f1_ap_sklearn = round(f1_score(y_true, y_pred, pos_label=1), 4)
    except Exception:
        f1_ap_sklearn = f1_ap
    metrics = {
        "balanced_accuracy":    round(balanced_accuracy_score(y_true, y_pred), 4),
        "ap_recall_sensitivity":round(ap_recall, 4),
        "pd_specificity":       round(specificity, 4),
        "ap_precision":         round(ap_precision, 4),
        "f1_ap":                round(f1_ap, 4),
        "f1_ap_sklearn":        f1_ap_sklearn,
        "accuracy":             round(accuracy_score(y_true, y_pred), 4),
    }
    if y_score is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_score), 4)
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation (after tuning)
# ---------------------------------------------------------------------------

def cross_validate_final(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict] = []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_val)
        y_score = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        fold_metrics.append(compute_metrics(y_val.values, y_pred, y_score))

    all_keys = fold_metrics[0].keys()
    return (
        {f"{k}_mean": round(float(np.mean([m[k] for m in fold_metrics])), 4) for k in all_keys}
        | {f"{k}_std":  round(float(np.std( [m[k] for m in fold_metrics])), 4) for k in all_keys}
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(
    data_dir: Path = Path("."),
    dr_method: str = "pca",
    n_trials: int | None = None,
    skip_tuning: bool = False,
    models_to_run: list[str] | None = None,
) -> None:
    from sklearn.metrics import classification_report

    from data.data_preprocessing import load_clinical_dataset
    from data.dimensionality_reduction import preprocess_and_reduce

    run_results_dir = RESULTS_DIR / dr_method
    run_results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load data -------------------------------------------------------
    print("=" * 65)
    print(f"  DR method: {dr_method}")
    print("=" * 65)
    dataset = load_clinical_dataset(data_dir)

    # ---- 2. Preprocess + DR -------------------------------------------------
    split = preprocess_and_reduce(
        X=dataset.X,
        y=dataset.y,
        dr_config=dr_method,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        output_dir=RESULTS_DIR,
    )
    print(
        f"  Feature matrix: {split.X_train.shape[1]} dims "
        f"(from {len(split.feature_names_in)} raw features)"
    )

    # ---- 3. Model selection -------------------------------------------------
    model_names = models_to_run or list(MODEL_REGISTRY.keys())
    print(f"\n  Models: {model_names}")

    # ---- 4. HPO via Optuna --------------------------------------------------
    best_params_map: dict[str, dict] = {}
    best_cv_map: dict[str, float] = {}

    if skip_tuning:
        print("\n  [skip-tuning] Using default hyperparameters.\n")
        for name in model_names:
            best_params_map[name] = {}
    else:
        print(f"\n{'='*65}")
        print(f"  Optuna HPO ({N_CV_FOLDS}-fold CV, F1 objective for AP class)")
        print(f"{'='*65}")
        for name in model_names:
            t0 = time.time()
            effective_trials = n_trials or MODEL_REGISTRY[name]["n_trials_default"]
            print(f"  Tuning {name:<28} ({effective_trials} trials)...", end=" ", flush=True)
            params, score = tune_model(
                name, split.X_train, split.y_train, n_trials=effective_trials
            )
            best_params_map[name] = params
            best_cv_map[name] = score
            elapsed = time.time() - t0
            print(f"best F1={score:.4f}  ({elapsed:.0f}s)")

    # ---- 5. CV with tuned models --------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Final {N_CV_FOLDS}-fold CV with tuned hyperparams")
    print(f"{'='*65}")

    cv_results: list[dict] = []
    for name in model_names:
        model = build_tuned_model(name, best_params_map[name], split.y_train)
        cv_stats = cross_validate_final(model, split.X_train, split.y_train)
        cv_stats["model"] = name
        cv_results.append(cv_stats)
        print(
            f"  {name:<28} "
            f"F1_ap_mean={cv_stats['f1_ap_mean']:.3f}"
            f"±{cv_stats['f1_ap_std']:.3f}  "
            f"AP_recall={cv_stats['ap_recall_sensitivity_mean']:.3f}  "
            f"AUC={cv_stats.get('auc_roc_mean', float('nan')):.3f}"
        )

    cv_df = pd.DataFrame(cv_results).set_index("model")
    cv_df.to_csv(run_results_dir / "cv_results.csv")

    # ---- 6. Final evaluation on held-out test set ---------------------------
    print(f"\n{'='*65}")
    print("  Final evaluation on held-out TEST set")
    print(f"{'='*65}")

    test_results: list[dict] = []
    clf_reports: dict = {}

    for name in model_names:
        model = build_tuned_model(name, best_params_map[name], split.y_train)
        model.fit(split.X_train, split.y_train)
        y_pred  = model.predict(split.X_test)
        y_score = (
            model.predict_proba(split.X_test)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        metrics = compute_metrics(split.y_test.values, y_pred, y_score)
        metrics["model"] = name
        test_results.append(metrics)
        clf_reports[name] = classification_report(
            split.y_test, y_pred, target_names=["PD", "AP"],
            output_dict=True, zero_division=0,
        )
        print(
            f"  {name:<28} "
            f"F1_ap={metrics['f1_ap']:.3f}  "
            f"AP_recall={metrics['ap_recall_sensitivity']:.3f}  "
            f"AUC={metrics.get('auc_roc', float('nan')):.3f}"
        )

    test_df = (
        pd.DataFrame(test_results)
        .set_index("model")
        .sort_values("f1_ap", ascending=False)
    )
    test_df.to_csv(run_results_dir / "model_results.csv")

    # ---- 7. Save best_params and summary ------------------------------------
    with open(run_results_dir / "best_hyperparams.json", "w") as f:
        json.dump({k: str(v) for k, v in best_params_map.items()}, f, indent=2)

    summary = {
        "dr_method": dr_method,
        "n_train": int(len(split.y_train)),
        "n_test":  int(len(split.y_test)),
        "ap_train": int((split.y_train == 1).sum()),
        "ap_test":  int((split.y_test  == 1).sum()),
        "n_features_after_dr": split.X_train.shape[1],
        "n_features_before_dr": len(split.feature_names_in),
        "cumulative_variance_explained": round(split.cumulative_variance, 4),
        "cv_folds": N_CV_FOLDS,
        "skip_tuning": skip_tuning,
        "models": model_names,
        "optuna_best_cv": best_cv_map,
    }
    with open(run_results_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _write_report(dr_method, split, cv_df, test_df, clf_reports, best_params_map, run_results_dir)
    print(f"\n  Results saved to: {run_results_dir.resolve()}/")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_report(
    dr_method: str,
    split,
    cv_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reports: dict,
    best_params: dict,
    output_dir: Path,
) -> None:
    lines = [
        "# C-OPN: PD vs Atypical Parkinsonism — Classification Results",
        "",
        f"**Dimensionality reduction method:** `{dr_method}`",
        "",
        "## Dataset",
        f"- Training: {len(split.y_train)} patients "
        f"({(split.y_train==0).sum()} PD, {(split.y_train==1).sum()} AP)",
        f"- Test: {len(split.y_test)} patients "
        f"({(split.y_test==0).sum()} PD, {(split.y_test==1).sum()} AP)",
        f"- Features before DR: {len(split.feature_names_in)}",
        f"- Features after DR:  {split.X_train.shape[1]}",
        f"- Variance retained:  {split.cumulative_variance*100:.1f}%",
        "",
        "## Cross-Validation (tuned hyperparams)",
        f"_{N_CV_FOLDS}-fold stratified CV on training set._",
        "",
        _df_to_md(cv_df[[
            c for c in cv_df.columns
            if any(c.startswith(k) for k in
                   ["f1_ap", "ap_recall", "auc"])
        ]].reset_index()),
        "",
        "## Held-Out Test Set",
        "",
        _df_to_md(test_df.reset_index()),
        "",
        "## Best Hyperparameters (Optuna TPE)",
        "",
    ]
    for name, params in best_params.items():
        lines.append(f"**{name}:** `{params}`")
    lines.append("")
    lines += ["## AP Classification Detail (Test Set)", ""]
    for model_name, report in reports.items():
        ap = report.get("AP", {})
        lines += [
            f"### {model_name}",
            f"- AP precision: {ap.get('precision', 0):.3f}",
            f"- AP recall:    {ap.get('recall', 0):.3f}",
            f"- AP F1:        {ap.get('f1-score', 0):.3f}",
            f"- AP support:   {int(ap.get('support', 0))}",
            "",
        ]

    (output_dir / "model_results.md").write_text("\n".join(lines), encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train C-OPN PD vs AP classifiers with configurable DR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DR methods:
  pca              StandardScaler → PCA (baseline)
  famd             Factor Analysis of Mixed Data
  catpca           CATPCA with ALS optimal scaling (ordinal-aware)
  hellinger        Hellinger distance feature selection (imbalance-aware)
  famd_hellinger   Sequential FAMD → Hellinger (recommended)

Examples:
  python train_models.py --dr-method famd_hellinger --n-trials 80
  python train_models.py --dr-method pca --skip-tuning
  python train_models.py --dr-method hellinger --models logistic_regression lightgbm
        """,
    )
    parser.add_argument(
        "--dr-method",
        type=str,
        default="pca",
        choices=["pca", "famd", "catpca", "hellinger", "famd_hellinger"],
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing C-OPN CSV files (default: current directory)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optuna trials per model (default: per-model default in registry)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna HPO; use default hyperparameters (for quick runs/debugging)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=(
            f"Models to run (default: all). "
            f"Options: {', '.join(MODEL_REGISTRY.keys())}"
        ),
    )
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        dr_method=args.dr_method,
        n_trials=args.n_trials,
        skip_tuning=args.skip_tuning,
        models_to_run=args.models,
    )