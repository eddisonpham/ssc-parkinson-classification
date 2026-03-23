"""
dimensionality_reduction.py
============================
Modular dimensionality-reduction (DR) methods for the C-OPN Parkinson
classification pipeline.

Methods implemented
-------------------
1.  MCAReducer              – Multiple Correspondence Analysis (categorical-only)
2.  FAMDReducer             – Factor Analysis of Mixed Data (mixed types)
3.  CATSCAReducer           – CATPCA approximation via Alternating Least Squares
4.  HellingerSelector       – sssHD: Hellinger-distance feature selector for
                              extreme class imbalance
5.  LLMEmbeddingReducer     – Sentence-encoder + PCA (optional, requires
                              sentence-transformers)
6.  MRLEmbeddingReducer     – MRL-truncation variant of LLM encoder (optional)
7.  TFIDFEmbeddingReducer   – Lightweight TF-IDF + PCA fallback for text columns
8.  DRPipeline              – Sequential/parallel composition of any reducers

Factory
-------
build_dr_pipeline(config: dict) -> DRPipeline

Public helpers
--------------
detect_column_types(X, categorical_threshold)
    – Returns (numeric_cols, categorical_cols, text_cols) with heuristics
      matching the C-OPN column profile.

Design notes
------------
•  All reducers follow the sklearn ``fit`` / ``transform`` / ``fit_transform``
   API and accept + return ``pd.DataFrame`` (not ndarray), preserving column
   provenance.
•  Missing values are handled internally; callers need not impute first.
•  Each reducer is independently configurable for ablation studies.
•  LLM-based reducers degrade gracefully: if sentence-transformers is absent,
   they fall back to TFIDFEmbeddingReducer transparently.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORICAL_THRESHOLD = 20   # nunique ≤ N → categorical
_DEFAULT_TEXT_MIN_LENGTH = 15          # mean len > N → text column
_DEFAULT_N_COMPONENTS = 50
_DEFAULT_N_BINS = 10


# ---------------------------------------------------------------------------
# Column-type detection
# ---------------------------------------------------------------------------

def detect_column_types(
    X: pd.DataFrame,
    categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    text_min_mean_length: float = _DEFAULT_TEXT_MIN_LENGTH,
) -> tuple[list[str], list[str], list[str]]:
    """Heuristically split columns into numeric, categorical, and text groups.

    Rules (applied in priority order)
    ----------------------------------
    1. Numeric dtype AND nunique > threshold  → numeric
    2. Object dtype AND mean string length > text_min_mean_length → text
    3. Object dtype OR nunique ≤ threshold    → categorical

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target column).
    categorical_threshold : int
        Maximum number of unique values for a column to be treated as
        categorical rather than numeric.
    text_min_mean_length : float
        Minimum mean character length of non-null values for a string column
        to be classified as free-text rather than categorical.

    Returns
    -------
    (numeric_cols, categorical_cols, text_cols) : tuple of list[str]
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    text_cols: list[str] = []

    for col in X.columns:
        series = X[col].dropna()
        if len(series) == 0:
            categorical_cols.append(col)
            continue

        is_numeric = pd.api.types.is_numeric_dtype(X[col])
        n_unique = X[col].nunique(dropna=True)

        if is_numeric and n_unique > categorical_threshold:
            numeric_cols.append(col)
        elif not is_numeric:
            mean_len = series.astype(str).str.len().mean()
            if mean_len > text_min_mean_length:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            # numeric but low cardinality → treat as ordinal/categorical
            categorical_cols.append(col)

    return numeric_cols, categorical_cols, text_cols


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseReducer(ABC):
    """Abstract base class for all dimensionality-reduction steps.

    Subclasses must implement ``fit`` and ``transform``.
    All methods accept and return ``pd.DataFrame``.
    """

    name: str = "base"

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseReducer":
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def _output_columns(self, prefix: str, n: int) -> list[str]:
        return [f"{prefix}__{i:03d}" for i in range(n)]


# ---------------------------------------------------------------------------
# 1. MCA – Multiple Correspondence Analysis
# ---------------------------------------------------------------------------

class MCAReducer(BaseReducer):
    """Multiple Correspondence Analysis for purely categorical feature sets.

    Algorithm
    ---------
    1. One-hot encode categorical columns (binary indicator matrix Z).
    2. Compute the correspondence matrix P = Z / n.
    3. Subtract expected values (independence model): Z_std = (P - r·c') / sqrt(r·c').
    4. Apply TruncatedSVD to Z_std.
    5. Scale principal coordinates by singular values.

    This reproduces the chi-squared geometry described in Paper [1].

    Parameters
    ----------
    n_components : int
        Number of MCA dimensions to retain.
    categorical_threshold : int
        Columns with ``nunique ≤ categorical_threshold`` are treated as
        categorical regardless of dtype.
    """

    name = "mca"

    def __init__(
        self,
        n_components: int = _DEFAULT_N_COMPONENTS,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.n_components = n_components
        self.categorical_threshold = categorical_threshold
        self._encoder: OneHotEncoder | None = None
        self._imputer: SimpleImputer | None = None
        self._svd: TruncatedSVD | None = None
        self._cat_cols: list[str] = []
        self._col_masses: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MCAReducer":
        _, cat_cols, _ = detect_column_types(X, self.categorical_threshold)
        self._cat_cols = cat_cols or list(X.columns)

        Xcat = X[self._cat_cols].astype(str)
        self._imputer = SimpleImputer(strategy="most_frequent")
        Xcat_imp = pd.DataFrame(
            self._imputer.fit_transform(Xcat),
            columns=self._cat_cols,
            index=X.index,
        )

        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        Z = self._encoder.fit_transform(Xcat_imp)

        Z_dense = np.asarray(Z.todense(), dtype=float)
        n = Z_dense.shape[0]

        P = Z_dense / n
        row_masses = P.sum(axis=1, keepdims=True)          # shape (n, 1)
        col_masses = P.sum(axis=0, keepdims=True)          # shape (1, p)
        self._col_masses = col_masses.ravel()

        # Standardised residuals (chi-square metric)
        expected = row_masses @ col_masses
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_std = (P - expected) / np.sqrt(np.maximum(expected, 1e-10))

        n_comp = min(self.n_components, min(Z_std.shape) - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(Z_std)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xcat = X[self._cat_cols].astype(str)
        Xcat_imp = pd.DataFrame(
            self._imputer.transform(Xcat),
            columns=self._cat_cols,
            index=X.index,
        )
        Z = self._encoder.transform(Xcat_imp)
        Z_dense = np.asarray(Z.todense(), dtype=float)
        n = Z_dense.shape[0]

        P = Z_dense / n
        row_masses = P.sum(axis=1, keepdims=True)
        col_masses = self._col_masses[np.newaxis, :]
        expected = row_masses @ col_masses
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_std = (P - expected) / np.sqrt(np.maximum(expected, 1e-10))

        coords = self._svd.transform(Z_std)
        cols = self._output_columns("mca", coords.shape[1])
        return pd.DataFrame(coords, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# 2. FAMD – Factor Analysis of Mixed Data
# ---------------------------------------------------------------------------

class FAMDReducer(BaseReducer):
    """Factor Analysis of Mixed Data (categorical + numeric).

    Implements the method described in Papers [1, 2].

    Algorithm
    ---------
    Numeric block  : standardise (mean 0, std 1) after median imputation.
    Categorical block : apply the MCA indicator encoding (frequency-scaled
                        one-hot), giving columns the same chi-square geometry
                        as in MCA.
    Joint SVD       : concatenate both blocks and run TruncatedSVD.

    Parameters
    ----------
    n_components : int
        Number of FAMD dimensions to retain.
    categorical_threshold : int
        Columns with ``nunique ≤ categorical_threshold`` are treated as
        categorical regardless of dtype.
    """

    name = "famd"

    def __init__(
        self,
        n_components: int = _DEFAULT_N_COMPONENTS,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.n_components = n_components
        self.categorical_threshold = categorical_threshold

        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._num_imputer: SimpleImputer | None = None
        self._num_scaler: StandardScaler | None = None
        self._cat_imputer: SimpleImputer | None = None
        self._cat_encoder: OneHotEncoder | None = None
        self._svd: TruncatedSVD | None = None
        self._cat_col_freq: np.ndarray | None = None  # MCA frequency scaling

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FAMDReducer":
        num_cols, cat_cols, _ = detect_column_types(X, self.categorical_threshold)
        self._num_cols = num_cols
        self._cat_cols = cat_cols

        blocks: list[np.ndarray] = []

        # ---- numeric block ----
        if self._num_cols:
            self._num_imputer = SimpleImputer(strategy="median")
            self._num_scaler = StandardScaler()
            Xnum = self._num_imputer.fit_transform(X[self._num_cols])
            Xnum_sc = self._num_scaler.fit_transform(Xnum)
            blocks.append(Xnum_sc)

        # ---- categorical block (MCA-style frequency scaling) ----
        if self._cat_cols:
            Xcat = X[self._cat_cols].astype(str)
            self._cat_imputer = SimpleImputer(strategy="most_frequent")
            Xcat_imp = self._cat_imputer.fit_transform(Xcat)
            self._cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            Z = self._cat_encoder.fit_transform(Xcat_imp)
            # Frequency scaling: divide by sqrt(col frequency)
            freq = Z.mean(axis=0)
            freq = np.where(freq < 1e-8, 1e-8, freq)
            self._cat_col_freq = freq
            Z_scaled = Z / np.sqrt(freq)
            blocks.append(Z_scaled)

        joint = np.hstack(blocks)
        n_comp = min(self.n_components, min(joint.shape) - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(joint)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        blocks: list[np.ndarray] = []

        if self._num_cols:
            Xnum = self._num_imputer.transform(X[self._num_cols])
            blocks.append(self._num_scaler.transform(Xnum))

        if self._cat_cols:
            Xcat = X[self._cat_cols].astype(str)
            Xcat_imp = self._cat_imputer.transform(Xcat)
            Z = self._cat_encoder.transform(Xcat_imp)
            Z_scaled = Z / np.sqrt(self._cat_col_freq)
            blocks.append(Z_scaled)

        joint = np.hstack(blocks)
        coords = self._svd.transform(joint)
        cols = self._output_columns("famd", coords.shape[1])
        return pd.DataFrame(coords, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# 3. CATPCA – Categorical PCA with Optimal Scaling (ALS approximation)
# ---------------------------------------------------------------------------

class CATSCAReducer(BaseReducer):
    """Categorical PCA via Alternating Least Squares (CATPCA approximation).

    Implements the optimal-scaling concept from Papers [1, 2].

    True CATPCA iterates:
        1. Quantify category levels to maximise correlation with PCA scores.
        2. Re-run PCA on quantified matrix.
    until convergence.  This implementation runs the full ALS loop for
    ordinal/Likert columns and falls back to numeric PCA for true continuous
    columns.

    Parameters
    ----------
    n_components : int
        Number of principal components.
    max_iter : int
        Maximum ALS iterations for optimal scaling.
    tol : float
        Convergence tolerance (change in loss between iterations).
    categorical_threshold : int
        Columns with ``nunique ≤ threshold`` are quantified via ALS.
    """

    name = "catpca"

    def __init__(
        self,
        n_components: int = _DEFAULT_N_COMPONENTS,
        max_iter: int = 50,
        tol: float = 1e-4,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.categorical_threshold = categorical_threshold

        self._num_cols: list[str] = []
        self._ord_cols: list[str] = []
        self._num_imputer: SimpleImputer | None = None
        self._ord_imputer: SimpleImputer | None = None
        self._ord_encoder: OrdinalEncoder | None = None
        self._quantifications: dict[str, np.ndarray] = {}  # col → level→value
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CATSCAReducer":
        num_cols, cat_cols, _ = detect_column_types(X, self.categorical_threshold)
        # Treat low-cardinality numerics as ordinal too
        self._num_cols = num_cols
        self._ord_cols = cat_cols

        # Impute
        self._num_imputer = SimpleImputer(strategy="median")
        self._ord_imputer = SimpleImputer(strategy="most_frequent")

        Xnum = (
            self._num_imputer.fit_transform(X[self._num_cols])
            if self._num_cols else np.empty((len(X), 0))
        )

        # Ordinal-encode categorical
        if self._ord_cols:
            Xcat = X[self._ord_cols].astype(str)
            Xcat_imp = pd.DataFrame(
                self._ord_imputer.fit_transform(Xcat),
                columns=self._ord_cols,
            )
            self._ord_encoder = OrdinalEncoder()
            Xord = self._ord_encoder.fit_transform(Xcat_imp)
        else:
            Xord = np.empty((len(X), 0))

        # ALS optimal scaling on ordinal block
        Xquant = self._als_quantify(Xord, self._ord_cols)

        joint = np.hstack([Xnum, Xquant])
        self._scaler = StandardScaler()
        joint_sc = self._scaler.fit_transform(joint)

        n_comp = min(self.n_components, min(joint_sc.shape))
        self._pca = PCA(n_components=n_comp, random_state=42)
        self._pca.fit(joint_sc)
        return self

    def _als_quantify(self, Xord: np.ndarray, cols: list[str]) -> np.ndarray:
        """Iteratively find numeric quantifications for each ordinal column."""
        if Xord.shape[1] == 0:
            return Xord
        Xquant = Xord.copy().astype(float)
        n, p = Xquant.shape
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            # Step 1: PCA on current quantified matrix
            scaler_tmp = StandardScaler()
            Xsc = scaler_tmp.fit_transform(Xquant)
            n_comp_tmp = min(self.n_components, min(Xsc.shape))
            pca_tmp = PCA(n_components=n_comp_tmp)
            scores = pca_tmp.fit_transform(Xsc)  # (n, k)

            # Step 2: Update quantifications per column via LS regression
            for j in range(p):
                levels = np.unique(Xord[:, j])
                for lv in levels:
                    mask = Xord[:, j] == lv
                    if mask.sum() == 0:
                        continue
                    # Optimal quantification = mean of PCA scores for that level
                    Xquant[mask, j] = scores[mask].mean()

            # Convergence check
            loss = np.mean((Xquant - Xord) ** 2)
            if abs(prev_loss - loss) < self.tol:
                logger.debug(f"CATPCA ALS converged in {iteration + 1} iterations.")
                break
            prev_loss = loss

        return Xquant

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xnum = (
            self._num_imputer.transform(X[self._num_cols])
            if self._num_cols else np.empty((len(X), 0))
        )
        if self._ord_cols:
            Xcat = X[self._ord_cols].astype(str)
            Xcat_imp = pd.DataFrame(
                self._ord_imputer.transform(Xcat),
                columns=self._ord_cols,
            )
            Xord = self._ord_encoder.transform(Xcat_imp)
        else:
            Xord = np.empty((len(X), 0))

        Xquant = self._als_quantify(Xord, self._ord_cols)
        joint = np.hstack([Xnum, Xquant])
        joint_sc = self._scaler.transform(joint)
        coords = self._pca.transform(joint_sc)
        cols = self._output_columns("catpca", coords.shape[1])
        return pd.DataFrame(coords, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# 4. HellingerSelector – sssHD (Paper [3])
# ---------------------------------------------------------------------------

class HellingerSelector(BaseReducer):
    """Hellinger-distance stable sparse feature selector (sssHD).

    Addresses extreme class imbalance (AP vs PD: ~6% minority) by using
    Hellinger distance — a class-insensitive divergence measure — to rank
    features, then optionally refines selection with a sparse L1-SVM.

    This reproduces the key theoretical property of Paper [3]:
    ``H(p, q)`` is invariant to changes in the class-imbalance ratio.

    Parameters
    ----------
    n_features : int or None
        Hard cap on features retained.  If ``None``, uses the L1-SVM
        to determine sparsity automatically.
    n_bins : int
        Number of histogram bins for continuous feature discretisation.
    use_svm_refinement : bool
        If True, re-rank with a L1 LinearSVC on the top ``svm_top_k``
        Hellinger-ranked features, mimicking the sssHD pipeline.
    svm_top_k : int
        Number of Hellinger-ranked features passed to the L1-SVM.
    svm_c : float
        Regularisation strength for the L1 LinearSVC.
    categorical_threshold : int
        Columns with ``nunique ≤ threshold`` are treated as categorical
        and discretised directly rather than binned.
    """

    name = "hellinger"

    def __init__(
        self,
        n_features: int = 100,
        n_bins: int = _DEFAULT_N_BINS,
        use_svm_refinement: bool = True,
        svm_top_k: int = 300,
        svm_c: float = 0.5,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.n_features = n_features
        self.n_bins = n_bins
        self.use_svm_refinement = use_svm_refinement
        self.svm_top_k = svm_top_k
        self.svm_c = svm_c
        self.categorical_threshold = categorical_threshold

        self._selected_cols: list[str] = []
        self._imputer: SimpleImputer | None = None
        self._encoder: OneHotEncoder | None = None
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._hellinger_scores: pd.Series | None = None

    # ---- Hellinger distance helpers ----------------------------------------

    @staticmethod
    def _safe_normalize(arr: np.ndarray) -> np.ndarray:
        total = arr.sum()
        return arr / total if total > 0 else arr

    @staticmethod
    def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
        """Hellinger distance between two probability vectors."""
        p = HellingerSelector._safe_normalize(p.astype(float))
        q = HellingerSelector._safe_normalize(q.astype(float))
        return float((1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

    def _score_column(
        self,
        values: np.ndarray,
        y: np.ndarray,
        is_categorical: bool,
    ) -> float:
        """Compute Hellinger distance between class-conditional distributions."""
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0

        if is_categorical:
            levels = np.unique(values)
            p0 = np.array([np.sum((values == lv) & (y == classes[0])) for lv in levels], dtype=float)
            p1 = np.array([np.sum((values == lv) & (y == classes[1])) for lv in levels], dtype=float)
        else:
            bins = np.histogram_bin_edges(values, bins=self.n_bins)
            p0 = np.histogram(values[y == classes[0]], bins=bins)[0].astype(float)
            p1 = np.histogram(values[y == classes[1]], bins=bins)[0].astype(float)

        return self._hellinger(p0, p1)

    # ---- Fit / transform ---------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HellingerSelector":
        if y is None:
            raise ValueError("HellingerSelector requires a target vector y.")

        num_cols, cat_cols, text_cols = detect_column_types(X, self.categorical_threshold)
        self._num_cols = num_cols
        self._cat_cols = cat_cols + text_cols  # treat text as categorical here

        # Impute
        self._imputer = SimpleImputer(strategy="median")
        Xnum_imp = (
            pd.DataFrame(
                self._imputer.fit_transform(X[self._num_cols]),
                columns=self._num_cols,
                index=X.index,
            )
            if self._num_cols else pd.DataFrame(index=X.index)
        )
        Xcat = X[self._cat_cols].fillna("__missing__").astype(str) if self._cat_cols else pd.DataFrame(index=X.index)

        y_arr = np.array(y)
        scores: dict[str, float] = {}

        for col in self._num_cols:
            scores[col] = self._score_column(Xnum_imp[col].to_numpy(), y_arr, is_categorical=False)

        for col in self._cat_cols:
            scores[col] = self._score_column(Xcat[col].to_numpy(), y_arr, is_categorical=True)

        self._hellinger_scores = pd.Series(scores).sort_values(ascending=False)

        # Optional L1-SVM refinement
        if self.use_svm_refinement:
            self._selected_cols = self._svm_refine(X, y, Xnum_imp, Xcat)
        else:
            top_k = self.n_features or len(self._hellinger_scores)
            self._selected_cols = list(self._hellinger_scores.head(top_k).index)

        logger.info(
            f"HellingerSelector: scored {len(scores)} features, "
            f"selected {len(self._selected_cols)}."
        )
        return self

    def _svm_refine(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        Xnum_imp: pd.DataFrame,
        Xcat: pd.DataFrame,
    ) -> list[str]:
        """Apply L1-SVM on Hellinger top-k to determine final feature set."""
        top_k = min(self.svm_top_k, len(self._hellinger_scores))
        top_cols = list(self._hellinger_scores.head(top_k).index)

        # Build a simple numeric representation for SVM
        num_part = [c for c in top_cols if c in self._num_cols]
        cat_part = [c for c in top_cols if c in self._cat_cols]

        blocks: list[np.ndarray] = []
        col_names: list[str] = []

        if num_part:
            blocks.append(Xnum_imp[num_part].to_numpy())
            col_names.extend(num_part)

        if cat_part:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_imp = SimpleImputer(strategy="most_frequent")
            Xc = cat_imp.fit_transform(X[cat_part].astype(str))
            Z = enc.fit_transform(Xc)
            ohe_names = [f"{c}__ohe_{i}" for c, cats in zip(cat_part, enc.categories_) for i in range(len(cats))]
            blocks.append(Z)
            col_names.extend(ohe_names)

        if not blocks:
            n_keep = self.n_features or top_k
            return list(self._hellinger_scores.head(n_keep).index)

        joint = np.hstack(blocks)
        svc = LinearSVC(
            penalty="l1",
            dual=False,
            class_weight="balanced",
            C=self.svm_c,
            max_iter=8000,
            random_state=42,
        )
        svc.fit(joint, y)
        coef_strength = np.abs(svc.coef_).max(axis=0)

        # Map OHE columns back to original categorical column names
        selected_ohe = set(np.array(col_names)[coef_strength > 1e-8])
        selected: list[str] = []
        for col in top_cols:
            if col in self._num_cols and col in selected_ohe:
                selected.append(col)
            elif col in self._cat_cols:
                if any(n.startswith(f"{col}__ohe_") for n in selected_ohe):
                    selected.append(col)

        # Enforce n_features cap
        if self.n_features and len(selected) > self.n_features:
            # Re-rank by Hellinger score
            selected = sorted(selected, key=lambda c: -self._hellinger_scores.get(c, 0))
            selected = selected[: self.n_features]

        return selected if selected else list(self._hellinger_scores.head(self.n_features or 50).index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the selected feature columns (no projection)."""
        cols_present = [c for c in self._selected_cols if c in X.columns]
        return X[cols_present].copy()

    @property
    def hellinger_scores(self) -> pd.Series:
        """All feature Hellinger scores sorted descending."""
        return self._hellinger_scores if self._hellinger_scores is not None else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 5. TF-IDF + PCA reducer (lightweight text baseline)
# ---------------------------------------------------------------------------

class TFIDFEmbeddingReducer(BaseReducer):
    """Lightweight TF-IDF + TruncatedSVD (LSA) for text columns.

    This is the fallback when sentence-transformers is unavailable and
    serves as the baseline text encoder described in ``feature_selection.py``.

    Parameters
    ----------
    n_components : int
        Output embedding dimension after SVD.
    max_features : int
        TF-IDF vocabulary size cap.
    ngram_range : tuple
        TF-IDF n-gram range.
    min_df : int
        Minimum document frequency for TF-IDF.
    """

    name = "tfidf_embedding"

    def __init__(
        self,
        n_components: int = 64,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 3,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.categorical_threshold = categorical_threshold

        self._text_cols: list[str] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def _to_documents(self, X: pd.DataFrame) -> list[str]:
        docs = []
        for _, row in X[self._text_cols].iterrows():
            tokens = []
            for col, val in row.items():
                if pd.isna(val):
                    continue
                token = str(val).strip().lower().replace(" ", "_")
                tokens.append(f"{col}={token}")
            docs.append(" ".join(tokens))
        return docs

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TFIDFEmbeddingReducer":
        _, cat_cols, text_cols = detect_column_types(X, self.categorical_threshold)
        self._text_cols = text_cols + cat_cols  # encode all non-numeric as text tokens
        if not self._text_cols:
            self._text_cols = list(X.columns)

        docs = self._to_documents(X)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        Z = self._vectorizer.fit_transform(docs)
        n_comp = min(self.n_components, min(Z.shape) - 1)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._svd.fit(Z)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        docs = self._to_documents(X)
        Z = self._vectorizer.transform(docs)
        coords = self._svd.transform(Z)
        cols = self._output_columns("tfidf_emb", coords.shape[1])
        return pd.DataFrame(coords, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# 6. LLM Embedding + PCA reducer (Paper [5])
# ---------------------------------------------------------------------------

class LLMEmbeddingReducer(BaseReducer):
    """Sentence-encoder + PCA for text columns (Paper [5]).

    Requires ``sentence-transformers`` to be installed::

        pip install sentence-transformers

    Falls back to :class:`TFIDFEmbeddingReducer` if the library is absent or
    the model cannot be loaded (e.g. no network access).

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformer model to use.
        Recommended: ``"sentence-transformers/all-mpnet-base-v2"``
    n_components : int
        Embedding dimension after PCA compression. The Oxford 2025 paper
        recommends ≤ 128 when labeled minority-class rows are scarce.
    batch_size : int
        Encoding batch size.
    """

    name = "llm_embedding"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        n_components: int = 64,
        batch_size: int = 64,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.n_components = n_components
        self.batch_size = batch_size
        self.categorical_threshold = categorical_threshold

        self._text_cols: list[str] = []
        self._pca: PCA | None = None
        self._model = None
        self._fallback: TFIDFEmbeddingReducer | None = None

    def _try_load_model(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded LLM encoder: {self.model_name}")
        except Exception as exc:
            logger.warning(
                f"sentence-transformers unavailable or model load failed ({exc}). "
                "Falling back to TFIDFEmbeddingReducer."
            )
            self._model = None

    def _column_text(self, X: pd.DataFrame) -> list[str]:
        return [
            " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if not pd.isna(val)
            )
            for _, row in X[self._text_cols].iterrows()
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LLMEmbeddingReducer":
        _, cat_cols, text_cols = detect_column_types(X, self.categorical_threshold)
        self._text_cols = text_cols if text_cols else cat_cols

        self._try_load_model()
        if self._model is None:
            self._fallback = TFIDFEmbeddingReducer(
                n_components=self.n_components,
                categorical_threshold=self.categorical_threshold,
            )
            self._fallback.fit(X, y)
            return self

        texts = self._column_text(X)
        embeddings = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        n_comp = min(self.n_components, min(embeddings.shape) - 1)
        self._pca = PCA(n_components=n_comp, random_state=42)
        self._pca.fit(embeddings)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fallback is not None:
            return self._fallback.transform(X)

        texts = self._column_text(X)
        embeddings = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        coords = self._pca.transform(embeddings)
        cols = self._output_columns("llm_emb", coords.shape[1])
        return pd.DataFrame(coords, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# 7. MRL Embedding Reducer (Paper [4])
# ---------------------------------------------------------------------------

class MRLEmbeddingReducer(BaseReducer):
    """Matryoshka Representation Learning truncation reducer (Paper [4]).

    Uses an MRL-trained sentence encoder whose early dimensions carry the most
    information, so truncation to ``target_dim`` is information-optimal.

    Recommended MRL models
    -----------------------
    • ``nomic-ai/nomic-embed-text-v1.5``
    • ``NeuML/pubmedbert-base-embeddings-matryoshka``

    Falls back to :class:`TFIDFEmbeddingReducer` if sentence-transformers is
    absent.

    Parameters
    ----------
    model_name : str
        MRL-enabled HuggingFace sentence-transformer model.
    target_dim : int
        Truncation dimension.  The MRL paper shows 64–128 is superior to
        post-hoc PCA at these sizes.
    """

    name = "mrl_embedding"

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        target_dim: int = 128,
        batch_size: int = 64,
        categorical_threshold: int = _DEFAULT_CATEGORICAL_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.categorical_threshold = categorical_threshold

        self._text_cols: list[str] = []
        self._model = None
        self._fallback: TFIDFEmbeddingReducer | None = None

    def _column_text(self, X: pd.DataFrame) -> list[str]:
        return [
            " | ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if not pd.isna(val)
            )
            for _, row in X[self._text_cols].iterrows()
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MRLEmbeddingReducer":
        _, cat_cols, text_cols = detect_column_types(X, self.categorical_threshold)
        self._text_cols = text_cols if text_cols else cat_cols

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded MRL encoder: {self.model_name}")
        except Exception as exc:
            logger.warning(
                f"MRL model load failed ({exc}). Falling back to TFIDFEmbeddingReducer."
            )
            self._model = None
            self._fallback = TFIDFEmbeddingReducer(
                n_components=self.target_dim,
                categorical_threshold=self.categorical_threshold,
            )
            self._fallback.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fallback is not None:
            return self._fallback.transform(X)

        texts = self._column_text(X)
        full_emb = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        # MRL truncation: slice first target_dim dimensions
        truncated = full_emb[:, : self.target_dim]
        cols = self._output_columns("mrl_emb", truncated.shape[1])
        return pd.DataFrame(truncated, columns=cols, index=X.index)


# ---------------------------------------------------------------------------
# 8. DRPipeline – sequential composition
# ---------------------------------------------------------------------------

@dataclass
class DRPipeline:
    """Sequential or mixed application of dimensionality-reduction steps.

    In *sequential* mode each step's output feeds the next step.
    In *parallel* mode each step operates on the original X and outputs
    are concatenated.

    Parameters
    ----------
    steps : list of (name, reducer) tuples
        Ordered list of reducer steps.
    mode : {"sequential", "parallel"}
        Composition strategy.
    keep_original_numeric : bool
        When mode is ``"parallel"``, also include the imputed original
        numeric columns alongside the DR outputs.
    """

    steps: list[tuple[str, BaseReducer]] = field(default_factory=list)
    mode: str = "sequential"       # "sequential" | "parallel"
    keep_original_numeric: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DRPipeline":
        if self.mode == "sequential":
            Xcur = X
            for name, reducer in self.steps:
                logger.info(f"DR pipeline fit step: {name}")
                reducer.fit(Xcur, y)
                Xcur = reducer.transform(Xcur)
        else:  # parallel
            for name, reducer in self.steps:
                logger.info(f"DR pipeline fit step (parallel): {name}")
                reducer.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "sequential":
            Xcur = X
            for _, reducer in self.steps:
                Xcur = reducer.transform(Xcur)
            return Xcur

        # Parallel: concatenate all outputs
        parts: list[pd.DataFrame] = []
        if self.keep_original_numeric:
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            if num_cols:
                imp = SimpleImputer(strategy="median")
                parts.append(
                    pd.DataFrame(
                        imp.fit_transform(X[num_cols]),
                        columns=[f"orig__{c}" for c in num_cols],
                        index=X.index,
                    )
                )
        for _, reducer in self.steps:
            parts.append(reducer.transform(X))
        return pd.concat(parts, axis=1)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Factory – build from config dict
# ---------------------------------------------------------------------------

_REDUCER_REGISTRY: dict[str, type[BaseReducer]] = {
    "mca": MCAReducer,
    "famd": FAMDReducer,
    "catpca": CATSCAReducer,
    "hellinger": HellingerSelector,
    "tfidf_embedding": TFIDFEmbeddingReducer,
    "llm_embedding": LLMEmbeddingReducer,
    "mrl_embedding": MRLEmbeddingReducer,
}


def build_dr_pipeline(config: dict[str, Any]) -> DRPipeline | None:
    """Construct a :class:`DRPipeline` from a YAML-derived config dictionary.

    Config format example
    ---------------------
    .. code-block:: yaml

        dimensionality_reduction:
          mode: sequential          # "sequential" or "parallel"
          keep_original_numeric: false
          methods:
            - type: famd
              n_components: 50
            - type: hellinger
              n_features: 100
              use_svm_refinement: true

    Parameters
    ----------
    config : dict
        The parsed ``dimensionality_reduction`` sub-dict from the experiment
        YAML. If the key is absent or ``None``, returns ``None`` (no DR).

    Returns
    -------
    DRPipeline or None
    """
    if not config:
        return None

    mode = config.get("mode", "sequential")
    keep_orig = config.get("keep_original_numeric", False)
    method_specs = config.get("methods", [])

    if not method_specs:
        return None

    steps: list[tuple[str, BaseReducer]] = []
    for spec in method_specs:
        reducer_type = spec.get("type", "").lower()
        if reducer_type not in _REDUCER_REGISTRY:
            raise ValueError(
                f"Unknown DR method '{reducer_type}'. "
                f"Available: {list(_REDUCER_REGISTRY)}"
            )
        klass = _REDUCER_REGISTRY[reducer_type]
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        try:
            reducer = klass(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"Invalid parameters for '{reducer_type}': {exc}"
            ) from exc
        steps.append((reducer_type, reducer))

    return DRPipeline(steps=steps, mode=mode, keep_original_numeric=keep_orig)


# ---------------------------------------------------------------------------
# Convenience: apply DR inside prepare_modeling_dataset
# ---------------------------------------------------------------------------

def apply_dr_to_prepared_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    dr_config: dict[str, Any] | None,
) -> tuple[pd.DataFrame, "DRPipeline | None"]:
    """Fit and apply a DR pipeline to a modeling matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix (after missingness filtering, leakage removal).
    y : pd.Series
        Target vector for supervised selectors (e.g. HellingerSelector).
    dr_config : dict or None
        Parsed ``dimensionality_reduction`` config block.

    Returns
    -------
    X_reduced : pd.DataFrame
    pipeline : DRPipeline or None
    """
    if not dr_config:
        return X, None

    pipeline = build_dr_pipeline(dr_config)
    if pipeline is None:
        return X, None

    logger.info(
        f"Applying DR pipeline ({pipeline.mode} mode, "
        f"{len(pipeline.steps)} step(s)) to matrix of shape {X.shape}."
    )
    X_reduced = pipeline.fit_transform(X, y)
    logger.info(f"DR complete. Output shape: {X_reduced.shape}.")
    return X_reduced, pipeline