"""
dimensionality_reduction.py
============================
Dimensionality reduction registry for the C-OPN clinical feature matrix.

Architecture
------------
Each reducer is a subclass of BaseReducer and registered via @register_reducer.
A DRPipeline chains reducers in sequential or parallel mode.
ClinicalPreprocessor handles imputation + scaling, then delegates to the pipeline.

Registered methods
------------------
  "pca"            — StandardScaler → PCA (baseline)
  "famd"           — Factor Analysis of Mixed Data (numeric + binary/ordinal)
  "catpca"         — Simplified CATPCA with ALS optimal scaling (ordinal-aware)
  "hellinger"      — Hellinger distance feature selection (imbalance-aware)
  "famd_hellinger" — Convenience alias: sequential FAMD → Hellinger (Pipeline A)

Method selection rationale for C-OPN
--------------------------------------
After data_preprocessing.py, features are already aggregated scale scores:
  - Continuous numeric  (UPDRS parts, MoCA total, z-scores, durations)
  - Binary flags        (administered flags, yes/no clinical features)
  - Small-range ordinal (Hoehn & Yahr 1-5, Likert-derived totals)

Given this, the recommended method order is:

  1. famd_hellinger (Pipeline A) — FAMD handles the mixed
     numeric/binary structure; Hellinger then surfaces the AP-discriminative
     components from a 6% minority class without SMOTE.

  2. famd — Good standalone option; formally correct for mixed types.
     For this already-encoded feature set, produces results close to PCA
     but with proper treatment of binary column variance (p*(1-p)).

  3. catpca — Best when ordinal structure matters (Hoehn & Yahr, etc.).
     Slower due to ALS iterations.

  4. hellinger — Standalone feature *selection* (not transformation).
     Use when you want a sparse, interpretable subset of original features.

  5. pca — Baseline; included for comparison. Does NOT distinguish
     numeric from binary columns during scaling.

Methods NOT included (from the design document)
  - MCA: requires raw multi-category text columns; our features are already
    encoded to numeric/binary scale scores.
  - TF-IDF / LLM / MRL: require raw questionnaire text, not applicable at
    this (post-extraction) stage.

Outputs per run
---------------
  results/{method}/
    train_preprocessed.csv   ← imputed + scaled, BEFORE DR (labels included)
    test_preprocessed.csv
    train_{method}.csv       ← after DR transformation (labels included)
    test_{method}.csv

Critical design rule
---------------------
ALL reducers are fit on the training set only, then applied to both splits.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REDUCER_REGISTRY: dict[str, type["BaseReducer"]] = {}


def register_reducer(name: str):
    """Class decorator that adds the reducer to the global registry."""
    def decorator(cls: type) -> type:
        _REDUCER_REGISTRY[name] = cls
        return cls
    return decorator


def list_reducers() -> list[str]:
    return sorted(_REDUCER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseReducer(ABC):
    """
    All reducers implement fit / transform on pandas DataFrames.
    y is optional (required for supervised methods like Hellinger).
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseReducer":
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Column type detection utility
# ---------------------------------------------------------------------------

def detect_column_types(
    df: pd.DataFrame,
    categorical_threshold: int = 10,
) -> tuple[list[str], list[str]]:
    """
    Split columns into numeric and categorical.

    A column is considered categorical if:
      - dtype is object or category, OR
      - number of unique non-null values <= categorical_threshold

    Returns (numeric_cols, categorical_cols).
    """
    numeric, categorical = [], []
    for col in df.columns:
        if df[col].dtype.kind in ("O", "U") or str(df[col].dtype) == "category":
            categorical.append(col)
        elif df[col].nunique(dropna=True) <= categorical_threshold:
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


# ---------------------------------------------------------------------------
# 1. PCA Reducer (baseline)
# ---------------------------------------------------------------------------

@register_reducer("pca")
class PCAReducer(BaseReducer):
    """
    StandardScaler → PCA with automatic n_components selection.

    Parameters
    ----------
    variance_target : float
        Fraction of variance to retain (default 0.85).
    n_components : int | None
        Fixed component count; overrides variance_target if set.
    max_components : int
        Hard ceiling on components.
    """

    def __init__(
        self,
        variance_target: float = 0.85,
        n_components: int | None = None,
        max_components: int = 50,
    ):
        self.variance_target = variance_target
        self.n_components = n_components
        self.max_components = max_components
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._n_out: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PCAReducer":
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.n_components is not None:
            n = min(self.n_components, self.max_components, X_scaled.shape[1])
        else:
            pca_full = PCA(random_state=42).fit(X_scaled)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n = int(np.searchsorted(cumvar, self.variance_target) + 1)
            n = min(n, self.max_components, X_scaled.shape[1])

        self._pca = PCA(n_components=n, random_state=42).fit(X_scaled)
        self._n_out = n
        evr = self._pca.explained_variance_ratio_.sum()
        print(f"  [PCA] {n} components → {evr*100:.1f}% variance")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xs = self._scaler.transform(X)
        out = self._pca.transform(Xs)
        cols = [f"pc_{i+1:02d}" for i in range(self._n_out)]
        return pd.DataFrame(out, index=X.index, columns=cols)

    @property
    def output_dim(self) -> int:
        return self._n_out

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_ if self._pca else np.array([])


# ---------------------------------------------------------------------------
# 2. FAMD Reducer
# ---------------------------------------------------------------------------

@register_reducer("famd")
class FAMDReducer(BaseReducer):
    """
    Factor Analysis of Mixed Data (Pagès 2004).

    Handles C-OPN's mix of continuous scores (UPDRS, z-scores, durations)
    and binary flags (administered flags, yes/no clinical features).

    Algorithm
    ---------
    1. Detect numeric vs categorical columns.
    2. Numeric block  : standardise (z-score).
    3. Categorical block : one-hot encode; scale each dummy column j by
       1 / sqrt(p_j) where p_j = proportion of 1s in training data.
       Then weight the whole block by 1 / sqrt(n_cat) so numeric and
       categorical blocks contribute equally to the SVD.
    4. Concatenate blocks → TruncatedSVD.
    5. Output: row projections (coordinates in FAMD space).

    This is equivalent to PCA on binary columns (their std = sqrt(p*(1-p))),
    but differs for numeric columns: categorical structure is preserved
    rather than flattened into a single scale.

    Parameters
    ----------
    n_components : int
        Number of FAMD dimensions to retain.
    variance_target : float
        Used when n_components is None to auto-select n_components.
    max_components : int
        Hard ceiling.
    categorical_threshold : int
        Columns with nunique <= threshold are treated as categorical.
    """

    def __init__(
        self,
        n_components: int | None = None,
        variance_target: float = 0.85,
        max_components: int = 50,
        categorical_threshold: int = 10,
    ):
        self.n_components = n_components
        self.variance_target = variance_target
        self.max_components = max_components
        self.categorical_threshold = categorical_threshold

        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._num_scaler: StandardScaler | None = None
        self._ohe: OneHotEncoder | None = None
        self._cat_scale_: np.ndarray | None = None  # 1/sqrt(p_j) per dummy
        self._cat_weight: float = 1.0               # 1/sqrt(n_cat)
        self._svd: TruncatedSVD | None = None
        self._n_out: int = 0

    def _build_matrix(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Construct the scaled FAMD matrix from X."""
        blocks = []

        # Numeric block
        if self._num_cols:
            num_data = X[self._num_cols].to_numpy(dtype=float)
            if fit:
                self._num_scaler = StandardScaler()
                num_scaled = self._num_scaler.fit_transform(num_data)
            else:
                num_scaled = self._num_scaler.transform(num_data)
            blocks.append(num_scaled)

        # Categorical block
        if self._cat_cols:
            cat_data = X[self._cat_cols]
            if fit:
                self._ohe = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    drop=None,
                )
                dummies = self._ohe.fit_transform(cat_data)
                # Compute column-wise proportions on training data
                # Add small epsilon to avoid division by zero
                p = dummies.mean(axis=0).clip(1e-6, 1 - 1e-6)
                self._cat_scale_ = 1.0 / np.sqrt(p)
                self._cat_weight = 1.0 / np.sqrt(max(len(self._cat_cols), 1))
            else:
                dummies = self._ohe.transform(cat_data)

            cat_scaled = dummies * self._cat_scale_ * self._cat_weight
            blocks.append(cat_scaled)

        if not blocks:
            raise ValueError("No columns to process in FAMDReducer.")

        return np.hstack(blocks)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FAMDReducer":
        self._num_cols, self._cat_cols = detect_column_types(
            X, self.categorical_threshold
        )
        M = self._build_matrix(X, fit=True)

        # Auto-select n_components using the full SVD
        if self.n_components is None:
            max_k = min(M.shape[0] - 1, M.shape[1], self.max_components)
            svd_full = TruncatedSVD(n_components=max_k, random_state=42).fit(M)
            total_var = svd_full.explained_variance_ratio_.sum()
            target = min(self.variance_target, total_var - 1e-6)
            cumvar = np.cumsum(svd_full.explained_variance_ratio_)
            n = int(np.searchsorted(cumvar, target) + 1)
            n = min(n, self.max_components, max_k)
        else:
            n = min(self.n_components, self.max_components, M.shape[1])

        self._svd = TruncatedSVD(n_components=n, random_state=42).fit(M)
        self._n_out = n
        evr = self._svd.explained_variance_ratio_.sum()
        print(
            f"  [FAMD] {len(self._num_cols)} numeric + "
            f"{len(self._cat_cols)} categorical cols → "
            f"{n} components ({evr*100:.1f}% variance)"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._build_matrix(X, fit=False)
        out = self._svd.transform(M)
        cols = [f"famd_{i+1:02d}" for i in range(self._n_out)]
        return pd.DataFrame(out, index=X.index, columns=cols)

    @property
    def output_dim(self) -> int:
        return self._n_out


# ---------------------------------------------------------------------------
# 3. CATPCA Reducer (simplified ALS optimal scaling)
# ---------------------------------------------------------------------------

@register_reducer("catpca")
class CATSCAReducer(BaseReducer):
    """
    Simplified CATPCA with Alternating Least Squares (ALS) optimal scaling.

    Exploits ordinal structure in Likert-scale and stage columns (e.g.
    Hoehn & Yahr, BAI/BDI item scores) by iteratively finding numerical
    quantifications that maximise PCA variance while respecting category ordering.

    Algorithm
    ---------
    1. Detect ordinal (categorical) vs numeric columns.
    2. Ordinal-encode categorical columns (integer codes preserve ordering).
    3. ALS loop (max_iter iterations):
       a. Standardise current matrix → fit PCA → extract scores Z.
       b. For each categorical column: update quantification[k] =
          mean row-score for rows where column == category k.
       c. Re-substitute quantifications into matrix column.
       d. Check convergence: max change in quantifications < tol.
    4. Final PCA on converged quantified matrix.

    Parameters
    ----------
    n_components, variance_target, max_components : same as PCAReducer.
    max_iter : int
        Maximum ALS iterations (default 50).
    tol : float
        Convergence threshold on quantification changes (default 1e-4).
    categorical_threshold : int
        Columns with nunique <= threshold are treated as categorical/ordinal.
    """

    def __init__(
        self,
        n_components: int | None = None,
        variance_target: float = 0.85,
        max_components: int = 50,
        max_iter: int = 50,
        tol: float = 1e-4,
        categorical_threshold: int = 10,
    ):
        self.n_components = n_components
        self.variance_target = variance_target
        self.max_components = max_components
        self.max_iter = max_iter
        self.tol = tol
        self.categorical_threshold = categorical_threshold

        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._cat_quantifications: dict[str, dict[int, float]] = {}
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._n_out: int = 0

    def _quantify(self, X: pd.DataFrame) -> np.ndarray:
        """Replace categorical columns with their learned quantifications."""
        M = X.copy()
        for col in self._cat_cols:
            codes = M[col].fillna(-1).astype(int)
            quant = codes.map(
                lambda c: self._cat_quantifications[col].get(c, 0.0)  # noqa: B023
            )
            M[col] = quant
        return M.to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CATSCAReducer":
        self._num_cols, self._cat_cols = detect_column_types(
            X, self.categorical_threshold
        )

        # Initialise quantifications = ordinal integer codes
        for col in self._cat_cols:
            unique_vals = sorted(X[col].dropna().unique())
            self._cat_quantifications[col] = {
                int(v): float(i) for i, v in enumerate(unique_vals)
            }

        M = self._quantify(X)

        # ALS iterations
        for iteration in range(self.max_iter):
            scaler_tmp = StandardScaler()
            M_scaled = scaler_tmp.fit_transform(np.nan_to_num(M, nan=0.0))

            max_k = min(M_scaled.shape[0] - 1, M_scaled.shape[1])
            n_comp_iter = min(max(
                self.n_components or 20, 10
            ), max_k)
            pca_tmp = PCA(n_components=n_comp_iter, random_state=42).fit(M_scaled)
            Z = pca_tmp.transform(M_scaled)  # (n, n_comp_iter)

            max_change = 0.0
            col_indices = {col: list(X.columns).index(col) for col in self._cat_cols}

            for col in self._cat_cols:
                col_idx = col_indices[col]
                new_quant: dict[int, float] = {}
                for k, old_val in self._cat_quantifications[col].items():
                    mask = X[col].fillna(-1).astype(int) == k
                    if mask.sum() == 0:
                        new_quant[k] = old_val
                    else:
                        # Use first PC score as the quantification target
                        new_val = float(Z[mask, 0].mean())
                        new_quant[k] = new_val
                        max_change = max(max_change, abs(new_val - old_val))
                self._cat_quantifications[col] = new_quant

            # Re-build M with updated quantifications
            M = self._quantify(X)

            if max_change < self.tol and iteration > 2:
                print(f"  [CATPCA] ALS converged at iteration {iteration + 1}")
                break
        else:
            print(f"  [CATPCA] ALS reached max_iter={self.max_iter}")

        # Final PCA on converged quantified matrix
        self._scaler = StandardScaler()
        M_final = self._scaler.fit_transform(np.nan_to_num(M, nan=0.0))

        if self.n_components is None:
            max_k = min(M_final.shape[0] - 1, M_final.shape[1], self.max_components)
            pca_full = PCA(random_state=42).fit(M_final)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n = int(np.searchsorted(cumvar, self.variance_target) + 1)
            n = min(n, self.max_components, max_k)
        else:
            n = min(self.n_components, self.max_components, M_final.shape[1])

        self._pca = PCA(n_components=n, random_state=42).fit(M_final)
        self._n_out = n
        evr = self._pca.explained_variance_ratio_.sum()
        print(
            f"  [CATPCA] {n} components → {evr*100:.1f}% variance "
            f"({len(self._cat_cols)} ordinal cols quantified)"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._quantify(X)
        M_scaled = self._scaler.transform(np.nan_to_num(M, nan=0.0))
        out = self._pca.transform(M_scaled)
        cols = [f"catpca_{i+1:02d}" for i in range(self._n_out)]
        return pd.DataFrame(out, index=X.index, columns=cols)

    @property
    def output_dim(self) -> int:
        return self._n_out


# ---------------------------------------------------------------------------
# 4. Hellinger Feature Selector
# ---------------------------------------------------------------------------

@register_reducer("hellinger")
class HellingerSelector(BaseReducer):
    """
    Hellinger distance feature selection (sssHD, Fu et al. 2020).

    Addresses the 6% AP minority class. Hellinger distance between class-
    conditional distributions is insensitive to class imbalance ratio —
    its value does not change as the ratio shifts from 1:1 to 99:1.

    Unlike chi-squared or Fisher's criterion, sssHD achieves the best
    performance with the *fewest* features without requiring SMOTE.

    Algorithm
    ---------
    1. For each feature:
       - Continuous: bin into n_bins buckets (training quantiles).
       - Binary/categorical: use category frequencies directly.
       - Compute class-conditional histograms p_PD and p_AP.
       - Score = H(p_PD, p_AP) = (1/√2) * ||√p - √q||₂
    2. Rank all features by H score descending.
    3. Optional L1-SVM refinement: pass top svm_top_k features through
       a LinearSVC(penalty='l1') for joint sparse selection.
    4. Return top n_features original columns (NOT a linear transformation).

    Note: output is a *column subset* of the input, not new components.
    Feature names are preserved for interpretability.

    Parameters
    ----------
    n_features : int
        Number of features to retain (default 60).
    n_bins : int
        Histogram bins for continuous features (default 10).
    use_svm_refinement : bool
        Apply L1-SVM as a second selection stage (default True).
    svm_top_k : int
        Features passed to L1-SVM (default 150).
    svm_c : float
        Regularisation for L1-SVM (default 0.1).
    categorical_threshold : int
        nunique threshold for categorical treatment.
    """

    def __init__(
        self,
        n_features: int = 60,
        n_bins: int = 10,
        use_svm_refinement: bool = True,
        svm_top_k: int = 150,
        svm_c: float = 0.1,
        categorical_threshold: int = 10,
    ):
        self.n_features = n_features
        self.n_bins = n_bins
        self.use_svm_refinement = use_svm_refinement
        self.svm_top_k = svm_top_k
        self.svm_c = svm_c
        self.categorical_threshold = categorical_threshold

        self._selected_cols: list[str] = []
        self._h_scores: pd.Series | None = None

    @staticmethod
    def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
        """Hellinger distance between two discrete distributions."""
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)
        return float((1.0 / np.sqrt(2.0)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    def _feature_score(
        self,
        col: pd.Series,
        y: pd.Series,
        is_categorical: bool,
        bins: np.ndarray | None,
    ) -> float:
        mask_pd = y == 0
        mask_ap = y == 1
        col_pd = col[mask_pd].dropna()
        col_ap = col[mask_ap].dropna()

        if len(col_pd) == 0 or len(col_ap) == 0:
            return 0.0

        if is_categorical:
            cats = np.union1d(col_pd.unique(), col_ap.unique())
            p_pd = np.array([(col_pd == c).sum() for c in cats], dtype=float)
            p_ap = np.array([(col_ap == c).sum() for c in cats], dtype=float)
        else:
            p_pd = np.histogram(col_pd, bins=bins)[0].astype(float)
            p_ap = np.histogram(col_ap, bins=bins)[0].astype(float)

        return self._hellinger(p_pd, p_ap)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HellingerSelector":
        num_cols, cat_cols = detect_column_types(X, self.categorical_threshold)
        cat_set = set(cat_cols)

        # Pre-compute histogram bins per continuous column from training data
        bins_map: dict[str, np.ndarray] = {}
        for col in num_cols:
            vals = X[col].dropna()
            if len(vals) >= self.n_bins:
                _, edges = np.histogram(vals, bins=self.n_bins)
                bins_map[col] = edges
            else:
                bins_map[col] = np.linspace(vals.min(), vals.max() + 1e-9, 3)

        # Score every feature
        scores: dict[str, float] = {}
        for col in X.columns:
            scores[col] = self._feature_score(
                X[col], y,
                is_categorical=(col in cat_set),
                bins=bins_map.get(col),
            )

        self._h_scores = pd.Series(scores).sort_values(ascending=False)
        top_k = min(self.svm_top_k, len(self._h_scores))
        candidates = list(self._h_scores.head(top_k).index)

        if self.use_svm_refinement and len(candidates) > self.n_features:
            try:
                X_cand = X[candidates].fillna(X[candidates].median())
                svc = LinearSVC(
                    penalty="l1",
                    C=self.svm_c,
                    dual=False,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                )
                svc.fit(X_cand, y)
                support = np.abs(svc.coef_[0]) > 0
                svm_selected = [c for c, s in zip(candidates, support) if s]
                # Fall back to top-k by H score if SVM is too aggressive
                if len(svm_selected) >= self.n_features:
                    candidates = svm_selected
                else:
                    print(
                        f"  [Hellinger] L1-SVM too aggressive ({len(svm_selected)} "
                        f"features); falling back to top-H ranking."
                    )
            except Exception as e:
                print(f"  [Hellinger] L1-SVM refinement failed ({e}); using H-rank only.")

        self._selected_cols = candidates[: self.n_features]
        print(
            f"  [Hellinger] Selected {len(self._selected_cols)} features "
            f"(top H score: {self._h_scores.iloc[0]:.4f})"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self._selected_cols if c in X.columns]
        return X[present].copy()

    @property
    def output_dim(self) -> int:
        return len(self._selected_cols)

    @property
    def h_scores(self) -> pd.Series:
        return self._h_scores if self._h_scores is not None else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# DRPipeline — chains reducers sequentially or in parallel
# ---------------------------------------------------------------------------

class DRPipeline:
    """
    Chain of dimensionality reduction steps.

    Parameters
    ----------
    steps : list of (name, reducer) tuples.
    mode : "sequential" | "parallel"
        sequential → each step's output feeds the next.
        parallel   → all steps receive the original X; outputs are concatenated.
    keep_original_numeric : bool
        (parallel only) Also append original numeric columns to the concat.
    """

    def __init__(
        self,
        steps: list[tuple[str, BaseReducer]],
        mode: str = "sequential",
        keep_original_numeric: bool = False,
    ):
        if mode not in ("sequential", "parallel"):
            raise ValueError(f"mode must be 'sequential' or 'parallel', got {mode!r}")
        self.steps = steps
        self.mode = mode
        self.keep_original_numeric = keep_original_numeric
        self._orig_num_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DRPipeline":
        if self.mode == "sequential":
            current = X
            for name, reducer in self.steps:
                print(f"  Fitting {name}...")
                reducer.fit(current, y)
                current = reducer.transform(current)
        else:  # parallel
            if self.keep_original_numeric:
                self._orig_num_cols, _ = detect_column_types(X)
            for name, reducer in self.steps:
                print(f"  Fitting {name}...")
                reducer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "sequential":
            current = X
            for _, reducer in self.steps:
                current = reducer.transform(current)
            return current
        else:  # parallel
            parts = [reducer.transform(X) for _, reducer in self.steps]
            if self.keep_original_numeric and self._orig_num_cols:
                present = [c for c in self._orig_num_cols if c in X.columns]
                parts.append(X[present])
            result = pd.concat(parts, axis=1)
            # De-duplicate column names that might appear in multiple blocks
            result = result.loc[:, ~result.columns.duplicated()]
            return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def output_dim(self) -> int:
        if self.mode == "sequential":
            return self.steps[-1][1].output_dim if self.steps else 0
        return sum(r.output_dim for _, r in self.steps)


# ---------------------------------------------------------------------------
# Factory: build pipeline from config dict or string shorthand
# ---------------------------------------------------------------------------

_PIPELINE_SHORTHANDS: dict[str, list[dict]] = {
    "pca": [{"type": "pca"}],
    "famd": [{"type": "famd"}],
    "catpca": [{"type": "catpca"}],
    "hellinger": [{"type": "hellinger"}],
    # Pipeline A from the design doc: FAMD → Hellinger
    "famd_hellinger": [
        {"type": "famd", "n_components": 80},
        {"type": "hellinger", "n_features": 60, "use_svm_refinement": True},
    ],
}


def build_dr_pipeline(config: dict | str | None) -> DRPipeline | None:
    """
    Build a DRPipeline from a config dict, string shorthand, or None.

    String shorthands
    -----------------
    "pca", "famd", "catpca", "hellinger", "famd_hellinger"

    Dict format
    -----------
    {
      "mode": "sequential",          # optional, default "sequential"
      "keep_original_numeric": False, # optional
      "methods": [
        {"type": "famd", "n_components": 80},
        {"type": "hellinger", "n_features": 60, "use_svm_refinement": True}
      ]
    }
    """
    if config is None:
        return None

    if isinstance(config, str):
        if config not in _PIPELINE_SHORTHANDS:
            raise ValueError(
                f"Unknown DR method {config!r}. "
                f"Available: {list(_PIPELINE_SHORTHANDS.keys())}"
            )
        config = {
            "mode": "sequential",
            "methods": _PIPELINE_SHORTHANDS[config],
        }

    mode = config.get("mode", "sequential")
    keep_num = config.get("keep_original_numeric", False)
    methods = config["methods"]

    steps = []
    for i, method_cfg in enumerate(methods):
        method_cfg = dict(method_cfg)          # don't mutate caller's dict
        type_name = method_cfg.pop("type")
        if type_name not in _REDUCER_REGISTRY:
            raise ValueError(
                f"Unknown reducer type {type_name!r}. "
                f"Registered: {list_reducers()}"
            )
        reducer_cls = _REDUCER_REGISTRY[type_name]
        reducer = reducer_cls(**method_cfg)
        steps.append((f"{type_name}_{i}", reducer))

    return DRPipeline(steps=steps, mode=mode, keep_original_numeric=keep_num)


# ---------------------------------------------------------------------------
# PreprocessedSplit dataclass (updated)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedSplit:
    """
    Holds all relevant data after preprocessing + dimensionality reduction.

    Preprocessed (imputed + scaled, before DR)
    -------------------------------------------
    X_train_preprocessed : shape (n_train, n_features_original)
    X_test_preprocessed  : shape (n_test,  n_features_original)

    DR-reduced
    ----------
    X_train : shape (n_train, n_reduced)
    X_test  : shape (n_test,  n_reduced)
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_preprocessed: pd.DataFrame
    X_test_preprocessed: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    dr_method: str
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: float
    feature_names_in: list[str]
    train_medians: pd.Series


# ---------------------------------------------------------------------------
# ClinicalPreprocessor (updated to use DRPipeline)
# ---------------------------------------------------------------------------

class ClinicalPreprocessor:
    """
    Fit on training data, transform train and test.

    Steps
    -----
    1. Median imputation (train medians only).
    2. StandardScaler.
    3. DR pipeline (optional).

    Parameters
    ----------
    dr_config : str | dict | None
        Passed to build_dr_pipeline. If None, only imputation + scaling is done.
    """

    def __init__(self, dr_config: str | dict | None = "pca"):
        self.dr_config = dr_config
        self._medians: pd.Series | None = None
        self._scaler: StandardScaler | None = None
        self._pipeline: DRPipeline | None = None
        self._feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "ClinicalPreprocessor":
        self._feature_names = list(X_train.columns)

        # 1. Impute
        self._medians = X_train.median(numeric_only=True)
        X_imp = X_train.fillna(self._medians)

        # 2. Scale
        self._scaler = StandardScaler()
        X_scaled_arr = self._scaler.fit_transform(X_imp)
        X_scaled = pd.DataFrame(X_scaled_arr, index=X_train.index, columns=X_train.columns)

        # 3. DR pipeline
        if self.dr_config is not None:
            self._pipeline = build_dr_pipeline(self.dr_config)
            # For supervised reducers (Hellinger), pass y_train
            self._pipeline.fit(X_scaled, y_train)

        return self

    def transform_preprocessed(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return imputed + scaled data (before DR)."""
        X_imp = X.fillna(self._medians)
        X_scaled_arr = self._scaler.transform(X_imp)
        return pd.DataFrame(X_scaled_arr, index=X.index, columns=X.columns)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DR-reduced data."""
        X_pre = self.transform_preprocessed(X)
        if self._pipeline is None:
            return X_pre
        return self._pipeline.transform(X_pre)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X_train, y_train).transform(X_train)

    @property
    def n_components(self) -> int:
        if self._pipeline is None:
            return len(self._feature_names)
        return self._pipeline.output_dim

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        if self._pipeline is None:
            return np.array([])
        # Extract from PCA/FAMD reducer if available
        for _, reducer in self._pipeline.steps:
            if hasattr(reducer, "explained_variance_ratio"):
                return reducer.explained_variance_ratio
        return np.array([])


# ---------------------------------------------------------------------------
# Main function: split → preprocess → DR → save all CSVs
# ---------------------------------------------------------------------------

def preprocess_and_reduce(
    X: pd.DataFrame,
    y: pd.Series,
    dr_config: str | dict | None = "pca",
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: Path = Path("results"),
) -> PreprocessedSplit:
    """
    Stratified split → fit preprocessor on train → transform both splits.
    Saves four CSVs under output_dir / {method_name}/:
      train_preprocessed.csv, test_preprocessed.csv
      train_{method}.csv, test_{method}.csv

    Parameters
    ----------
    X, y         : feature matrix and labels from load_clinical_dataset()
    dr_config    : string shorthand, config dict, or None (no DR)
    test_size    : fraction held out for evaluation
    random_state : reproducibility seed
    output_dir   : base results directory
    """
    from sklearn.model_selection import train_test_split

    # Determine method name for output directory
    if dr_config is None:
        method_name = "no_dr"
    elif isinstance(dr_config, str):
        method_name = dr_config
    else:
        method_name = "_".join(m["type"] for m in dr_config.get("methods", []))

    run_dir = Path(output_dir) / method_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Split -----------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[preprocess] Split → train: {len(X_train_raw)} "
        f"(AP={int((y_train==1).sum())}), "
        f"test: {len(X_test_raw)} (AP={int((y_test==1).sum())})"
    )

    # ---- 2. Fit preprocessor on train only ----------------------------------
    preprocessor = ClinicalPreprocessor(dr_config=dr_config)
    preprocessor.fit(X_train_raw, y_train)

    # ---- 3. Transform -------------------------------------------------------
    X_train_pre = preprocessor.transform_preprocessed(X_train_raw)
    X_test_pre  = preprocessor.transform_preprocessed(X_test_raw)
    X_train_dr  = preprocessor.transform(X_train_raw)
    X_test_dr   = preprocessor.transform(X_test_raw)

    # ---- 4. Save CSVs -------------------------------------------------------
    def _save(df: pd.DataFrame, labels: pd.Series, fname: str) -> None:
        out = df.copy()
        out.insert(0, "label", labels.values)
        out.insert(0, "project_key", labels.index)
        out.to_csv(run_dir / fname, index=False)
        print(f"  Saved {run_dir / fname}  {df.shape}")

    _save(X_train_pre, y_train, "train_preprocessed.csv")
    _save(X_test_pre,  y_test,  "test_preprocessed.csv")
    _save(X_train_dr,  y_train, f"train_{method_name}.csv")
    _save(X_test_dr,   y_test,  f"test_{method_name}.csv")

    evr = preprocessor.explained_variance_ratio
    return PreprocessedSplit(
        X_train=X_train_dr,
        X_test=X_test_dr,
        X_train_preprocessed=X_train_pre,
        X_test_preprocessed=X_test_pre,
        y_train=y_train,
        y_test=y_test,
        dr_method=method_name,
        n_components=preprocessor.n_components,
        explained_variance_ratio=evr,
        cumulative_variance=float(evr.sum()) if len(evr) > 0 else 0.0,
        feature_names_in=preprocessor._feature_names,
        train_medians=preprocessor._medians,
    )