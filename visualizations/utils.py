"""
utils.py
========
Shared constants, data-loading helpers, color palettes, and the custom
plotnine publication theme used across all visualisation modules.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    theme,
    theme_minimal,
)

# ─── Dimension-reduction methods ──────────────────────────────────────────────

DR_METHODS: Dict[str, str] = {
    "catpca":        "Categorical PCA",
    "pca":           "Standard PCA",
    "famd":          "FAMD",
    "hellinger":     "Hellinger",
    "famd_hellinger": "FAMD + Hellinger",
}

DR_ORDER: list[str] = list(DR_METHODS.keys())
DR_LABEL_ORDER: list[str] = list(DR_METHODS.values())

# ─── Models ───────────────────────────────────────────────────────────────────

MODEL_LABELS: Dict[str, str] = {
    "logistic_regression":  "Logistic Reg.",
    "svm":                  "SVM",
    "random_forest":        "Random Forest",
    "balanced_random_forest": "Balanced RF",
    "lightgbm":             "LightGBM",
    "xgboost":              "XGBoost",
    "mlp":                  "MLP",
    "easy_ensemble":        "Easy Ensemble",
}

MODEL_ORDER: list[str] = list(MODEL_LABELS.keys())
MODEL_LABEL_ORDER: list[str] = list(MODEL_LABELS.values())

# ─── Metrics ──────────────────────────────────────────────────────────────────

METRICS: Dict[str, str] = {
    "balanced_accuracy":      "Balanced Acc.",
    "ap_recall_sensitivity":  "AP Sensitivity",
    "pd_specificity":         "PD Specificity",
    "ap_precision":           "AP Precision",
    "f1_ap":                  "F1 (AP)",
    "accuracy":               "Accuracy",
    "auc_roc":                "AUC-ROC",
}

METRIC_COLS: list[str] = list(METRICS.keys())
METRIC_LABELS: list[str] = list(METRICS.values())

# Primary metrics for focused analyses (clinical relevance hierarchy)
KEY_METRICS: Dict[str, str] = {
    "balanced_accuracy":     "Balanced Acc.",
    "auc_roc":               "AUC-ROC",
    "ap_recall_sensitivity": "AP Sensitivity",
    "f1_ap":                 "F1 (AP)",
}

# ─── Color palettes ───────────────────────────────────────────────────────────
# Tableau-10 extended with WCAG-compliant accessibility tweaks.
# Each color is designed to remain distinguishable at 8% opacity on white.

MODEL_COLORS: Dict[str, str] = {
    "Logistic Reg.":  "#4E79A7",
    "SVM":            "#F28E2B",
    "Random Forest":  "#59A14F",
    "Balanced RF":    "#76B7B2",
    "LightGBM":       "#E15759",
    "XGBoost":        "#B07AA1",
    "MLP":            "#FF9DA7",
    "Easy Ensemble":  "#9C755F",
}

DR_COLORS: Dict[str, str] = {
    "Categorical PCA":   "#264653",
    "Standard PCA":      "#2A9D8F",
    "FAMD":              "#E9C46A",
    "Hellinger":         "#F4A261",
    "FAMD + Hellinger":  "#E76F51",
}

# Shape markers for DR methods (matplotlib marker strings)
DR_SHAPES: Dict[str, str] = {
    "Categorical PCA":  "o",
    "Standard PCA":     "s",
    "FAMD":             "^",
    "Hellinger":        "D",
    "FAMD + Hellinger": "v",
}

# ─── Custom plotnine theme ─────────────────────────────────────────────────────

_BASE_SIZE = 10
_BASE_FAMILY = "DejaVu Sans"   # guaranteed available in any matplotlib install


def theme_publication(base_size: int = _BASE_SIZE) -> theme:
    """
    A clean, publication-ready plotnine theme inspired by Nature/NEJM figure
    style:  white background, light panel fill, subtle major gridlines,
    bold axis titles, and compact strip labels.
    """
    return (
        theme_minimal(base_size=base_size, base_family=_BASE_FAMILY)
        + theme(
            # Canvas
            plot_background=element_rect(fill="white", color="white"),
            panel_background=element_rect(fill="#F8F9FA", color=None),
            # Grid
            panel_grid_major=element_line(color="#E2E6EA", size=0.4),
            panel_grid_minor=element_blank(),
            # Axes
            axis_line=element_line(color="#6C757D", size=0.45),
            axis_ticks=element_line(color="#6C757D", size=0.4),
            axis_text=element_text(color="#343A40", size=base_size - 1,
                                   family=_BASE_FAMILY),
            axis_title=element_text(color="#212529", size=base_size,
                                    face="bold", family=_BASE_FAMILY),
            # Titles
            plot_title=element_text(color="#212529", size=base_size + 3,
                                    face="bold", family=_BASE_FAMILY,
                                    margin={"b": 6}),
            plot_subtitle=element_text(color="#6C757D", size=base_size - 0.5,
                                       family=_BASE_FAMILY, margin={"b": 10}),
            # Legend
            legend_background=element_rect(fill="white", color="#DEE2E6",
                                           size=0.3),
            legend_key=element_rect(fill="white"),
            legend_title=element_text(face="bold", size=base_size - 1,
                                      family=_BASE_FAMILY),
            legend_text=element_text(size=base_size - 2, family=_BASE_FAMILY),
            # Facet strips
            strip_background=element_rect(fill="#E9ECEF", color="#CED4DA",
                                          size=0.4),
            strip_text=element_text(face="bold", size=base_size - 1,
                                    color="#343A40", family=_BASE_FAMILY),
        )
    )


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_all_results(results_dir: str) -> Dict[str, dict]:
    """
    Walk *results_dir* and load the three artefact files for every DR method.

    Returns
    -------
    dict
        Keys are DR method keys (e.g. "catpca").
        Values are dicts with keys "model_results", "cv_results", "summary".
    """
    all_data: Dict[str, dict] = {}

    for dr_key, dr_label in DR_METHODS.items():
        folder = os.path.join(results_dir, dr_key)
        paths = {
            "model_results": os.path.join(folder, "model_results.csv"),
            "cv_results":    os.path.join(folder, "cv_results.csv"),
            "summary":       os.path.join(folder, "experiment_summary.json"),
        }

        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"[WARN] {dr_key}: missing {missing}, skipping.")
            continue

        model_results = pd.read_csv(paths["model_results"])
        cv_results    = pd.read_csv(paths["cv_results"])
        with open(paths["summary"]) as fh:
            summary = json.load(fh)

        # Attach human-readable labels for plotting
        model_results["model_label"] = model_results["model"].map(MODEL_LABELS)
        cv_results["model_label"]    = cv_results["model"].map(MODEL_LABELS)
        model_results["dr_key"]      = dr_key
        model_results["dr_label"]    = dr_label
        cv_results["dr_key"]         = dr_key
        cv_results["dr_label"]       = dr_label

        all_data[dr_key] = {
            "model_results": model_results,
            "cv_results":    cv_results,
            "summary":       summary,
        }

    return all_data


def get_combined_test_df(all_data: Dict[str, dict]) -> pd.DataFrame:
    """Concatenate test-set result frames from every DR method."""
    return pd.concat(
        [v["model_results"] for v in all_data.values()],
        ignore_index=True,
    )


def get_combined_cv_df(all_data: Dict[str, dict]) -> pd.DataFrame:
    """Concatenate cross-validation result frames from every DR method."""
    return pd.concat(
        [v["cv_results"] for v in all_data.values()],
        ignore_index=True,
    )


# ─── Save helper ──────────────────────────────────────────────────────────────

def save_plot(plot, path: str, width: float = 10, height: float = 7,
              dpi: int = 180) -> None:
    """Save a plotnine figure, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plot.save(path, width=width, height=height, dpi=dpi, verbose=False)
    print(f"  ✓  {path}")