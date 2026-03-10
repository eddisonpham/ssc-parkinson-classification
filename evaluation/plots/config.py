from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "evaluation" / "plots"
OUTPUT_DIR = PLOTS_DIR / "output"

MODEL_FAMILY = {
    "logistic_regression": "linear",
    "svc_rbf": "kernel",
    "random_forest": "bagging_trees",
    "extra_trees": "bagging_trees",
    "hist_gradient_boosting": "boosting",
    "balanced_random_forest": "imbalance_ensemble",
    "easy_ensemble": "imbalance_ensemble",
    "xgboost_hist": "boosting",
    "xgboost_deeper": "boosting",
    "lightgbm_gbdt": "boosting",
    "lightgbm_extra_trees": "boosting",
    "catboost": "boosting",
    "feature_selection_logistic": "text_sparse_linear",
    "mixture_of_experts_gate": "expert_ensemble",
    "expert_average_ensemble": "expert_ensemble",
    "clinical_treatment_expert": "expert_component",
    "motor_cognition_expert": "expert_component",
    "questionnaire_expert": "expert_component",
    "low_burden_expert": "expert_component",
}

IMPLEMENTATION_COMPLEXITY = {
    "logistic_regression": 1.0,
    "svc_rbf": 2.5,
    "random_forest": 2.0,
    "extra_trees": 2.0,
    "hist_gradient_boosting": 2.5,
    "balanced_random_forest": 2.5,
    "easy_ensemble": 3.0,
    "xgboost_hist": 3.5,
    "xgboost_deeper": 3.8,
    "lightgbm_gbdt": 3.3,
    "lightgbm_extra_trees": 3.4,
    "catboost": 3.5,
    "feature_selection_logistic": 2.8,
    "mixture_of_experts_gate": 4.2,
    "expert_average_ensemble": 4.0,
    "clinical_treatment_expert": 3.5,
    "motor_cognition_expert": 3.6,
    "questionnaire_expert": 2.6,
    "low_burden_expert": 2.4,
}

PLOT_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity_recall_ap",
    "specificity_pd",
    "precision_ap",
    "f1_ap",
    "auc_roc",
]

PALETTE = {
    "linear": "#355070",
    "kernel": "#6d597a",
    "bagging_trees": "#457b9d",
    "boosting": "#2a9d8f",
    "imbalance_ensemble": "#e76f51",
    "text_sparse_linear": "#f4a261",
    "expert_ensemble": "#bc4749",
    "expert_component": "#8d99ae",
}
