# Dimensionality Reduction — C-OPN Parkinson Classification

> **File:** `dimensionality_reduction.py`
> **Integrates with:** `data_preprocessing.py` → `prepare_modeling_dataset(dr_config=…)`

---

## Why Dimensionality Reduction?

The C-OPN baseline snapshot presents three compounding challenges that make
standard feature use sub-optimal:

| Challenge | Detail |
|---|---|
| **Wide, mixed-type data** | ~224 features after default filtering; mix of bilingual categorical MCQ, Likert scales, numeric scales, and free-text responses |
| **Extreme class imbalance** | 2852 PD vs 171 AP (≈ 6% minority); standard variance-based DR ignores rare-class structure |
| **High missingness** | Many instruments are sparse (>50% missing cells in some tables) |

The five curated papers drive method selection:

| Paper | Method | Primary Strength |
|---|---|---|
| [1] Nguyen & Holmes 2019 | MCA / FAMD / CATPCA | Correct DR for categorical + mixed data |
| [2] Morera-Fernández et al. 2024 | CATPCA vs FAMD (empirical) | Rare-class improvement without row modification |
| [3] Fu et al. 2020 (sssHD) | Hellinger distance feature selection | Skew-insensitive; no SMOTE required |
| [4] Kusupati et al. 2022 (MRL) | MRL text embeddings | Tunable-dimension semantic embeddings |
| [5] Drinkall et al. 2025 | Encoder LLM → PCA | Low-sample, high-dim text compression |

---

## Available Methods

### 1. `MCAReducer` — Multiple Correspondence Analysis

**Paper:** [1]  
**Best for:** Purely categorical / MCQ columns.

MCA applies Correspondence Analysis to the binary indicator (one-hot) matrix.
It uses **chi-squared distance** on category frequencies rather than Euclidean
variance, making it appropriate for nominal data.

**Algorithm:**
1. One-hot encode categorical columns → indicator matrix **Z** (n × Σkⱼ).
2. Compute correspondence matrix **P** = Z / n.
3. Subtract independence model: Z_std = (P – r·cᵀ) / √(r·cᵀ).
4. TruncatedSVD on Z_std → principal coordinates.

**Config:**
```yaml
- type: mca
  n_components: 50
  categorical_threshold: 20   # nunique ≤ N → categorical
```

---

### 2. `FAMDReducer` — Factor Analysis of Mixed Data

**Papers:** [1, 2]  
**Best for:** Mixed numeric + categorical columns (the default C-OPN profile).

FAMD is the recommended first step for C-OPN data. It combines:
- **PCA** on standardised numeric columns.
- **MCA-style frequency scaling** on one-hot encoded categorical columns.
- Joint **TruncatedSVD** on the concatenated block matrix.

**Config:**
```yaml
- type: famd
  n_components: 80
  categorical_threshold: 20
```

**Trade-off vs MCA:** FAMD handles numeric columns; MCA does not. FAMD is
preferred for the C-OPN feature set where numeric motor scores coexist with
categorical questionnaire answers.

---

### 3. `CATSCAReducer` — CATPCA (Optimal Scaling ALS)

**Papers:** [1, 2]  
**Best for:** Likert-scale or ordered ordinal columns (PDQ-39, BAI, BDII,
SCOPA, etc.).

CATPCA applies **Alternating Least Squares (ALS)** optimal scaling: each
ordinal column is quantified numerically to maximise correlation with the
emerging PCA components, explicitly exploiting ordering within levels.

**Algorithm:**
1. Ordinal-encode categorical columns.
2. Iterate (ALS):
   a. PCA on current quantified matrix.
   b. Update quantification of each category level = mean PCA score of that
      level's rows.
3. Final PCA on converged quantified matrix.

**Config:**
```yaml
- type: catpca
  n_components: 60
  max_iter: 50
  tol: 1e-4
  categorical_threshold: 20
```

**Trade-off vs FAMD:** CATPCA outperforms FAMD when variables are ordinal (e.g.
"Never / Rarely / Sometimes / Always"); FAMD treats categories as unordered.
For strictly nominal columns, both are equivalent.

---

### 4. `HellingerSelector` — sssHD Feature Selection

**Paper:** [3]  
**Best for:** Extreme class imbalance (AP ≈ 6%); any feature space.

Hellinger distance is **class-insensitive** — its value does not change as the
imbalance ratio shifts from 1:1 to 99:1. This prevents the majority class from
dominating feature scores.

**Algorithm:**
1. For each feature, compute class-conditional histograms (numeric) or
   frequency tables (categorical) for PD and AP.
2. Score = Hellinger distance H(p₀, p₁).
3. Rank all features by score descending.
4. Optional L1-SVM refinement: apply LinearSVC(penalty="l1") on the
   Hellinger top-k features for joint sparse selection.

**Config:**
```yaml
- type: hellinger
  n_features: 100          # hard cap on retained features
  n_bins: 10               # histogram bins for continuous features
  use_svm_refinement: true
  svm_top_k: 300           # features passed to L1-SVM
  svm_c: 0.5
  categorical_threshold: 20
```

**Key property (from Paper [3]):** Unlike chi-squared or Fisher's criterion,
sssHD achieves the best performance with the *fewest* features, and limited
differences between performing and not performing SMOTE/oversampling.

---

### 5. `TFIDFEmbeddingReducer` — TF-IDF + LSA (text fallback)

**Best for:** Bilingual MCQ responses, free-text fields; no GPU required.

Converts non-numeric columns into row-level `column=value` token documents,
vectorises with TF-IDF, and compresses via TruncatedSVD (Latent Semantic
Analysis). This is the lightweight text baseline already used in
`feature_selection.py`.

**Config:**
```yaml
- type: tfidf_embedding
  n_components: 64
  max_features: 5000
  ngram_range: [1, 2]
  min_df: 3
```

---

### 6. `LLMEmbeddingReducer` — Sentence Encoder + PCA

**Paper:** [5] (Drinkall et al. 2025)  
**Best for:** Free-text questionnaire responses where semantic similarity matters.

Uses a pre-trained encoder-based sentence transformer to embed text columns,
then compresses via PCA. The Oxford 2025 paper demonstrates this strategy
improves generalisation when labeled minority-class rows are scarce (n << d).

**Requires:** `pip install sentence-transformers`  
**Falls back** to `TFIDFEmbeddingReducer` if library or model is unavailable.

**Config:**
```yaml
- type: llm_embedding
  model_name: "sentence-transformers/all-mpnet-base-v2"
  n_components: 64   # ≤128 recommended for ~170 AP rows
```

---

### 7. `MRLEmbeddingReducer` — MRL Truncation

**Paper:** [4] (Kusupati et al. 2022)  
**Best for:** Text embedding with tunable output dimension (no PCA needed).

Uses an MRL-trained sentence encoder that frontloads information into early
dimensions. Truncating to `target_dim` produces a self-contained sub-embedding
that was *explicitly trained* at that size — superior to post-hoc PCA at
d ≤ 128.

**Recommended models:**
- `nomic-ai/nomic-embed-text-v1.5`
- `NeuML/pubmedbert-base-embeddings-matryoshka`

**Requires:** `pip install sentence-transformers`  
**Falls back** to `TFIDFEmbeddingReducer` if unavailable.

**Config:**
```yaml
- type: mrl_embedding
  model_name: "nomic-ai/nomic-embed-text-v1.5"
  target_dim: 128
```

---

## Pipeline Modes

### Sequential (`mode: sequential`)
Each step's output becomes the next step's input.

```
Raw X → [FAMD → 80 dims] → [Hellinger → 60 dims] → Model
```

Use when you want progressive dimensionality compression.

### Parallel (`mode: parallel`)
Each step sees the original X; outputs are concatenated.

```
Raw X → FAMD → 50 dims ─┐
Raw X → TF-IDF → 32 dims ─┤→ concat → Model
Raw X (numeric) ──────────┘  (keep_original_numeric: true)
```

Use when different reducers capture complementary structure.

---

## Recommended Pipelines for C-OPN

### Pipeline A — Fast mixed-data (default for `advanced_experiment.yaml`)
```yaml
dimensionality_reduction:
  mode: sequential
  methods:
    - type: famd
      n_components: 80
    - type: hellinger
      n_features: 60
      use_svm_refinement: true
```
**Rationale:** FAMD handles the mixed numeric/categorical C-OPN structure;
Hellinger ensures the 6% AP minority drives final feature selection.

### Pipeline B — Ordinal-aware
```yaml
dimensionality_reduction:
  mode: sequential
  methods:
    - type: catpca
      n_components: 60
    - type: hellinger
      n_features: 50
```
**Rationale:** Exploits ordering in Likert-scale instruments (BAI, BDI,
PDQ-39, SCOPA). Slower due to ALS iterations.

### Pipeline C — Text-enriched parallel
```yaml
dimensionality_reduction:
  mode: parallel
  keep_original_numeric: true
  methods:
    - type: famd
      n_components: 50
    - type: tfidf_embedding
      n_components: 32
```
**Rationale:** Captures both structural (FAMD) and semantic (TF-IDF)
information from bilingual questionnaire responses.

### No DR (baseline behaviour)
```yaml
dimensionality_reduction: null
```

---

## Method Trade-off Summary

| Method | Categorical | Mixed | Imbalance-aware | Text-aware | Needs GPU |
|---|:---:|:---:|:---:|:---:|:---:|
| MCA | ★★★★★ | — | ★★★ | — | No |
| FAMD | ★★★★★ | ★★★★★ | ★★★ | — | No |
| CATPCA | ★★★★★ | ★★★★ | ★★★ | — | No |
| Hellinger | ★★★ | ★★★★ | ★★★★★ | — | No |
| TF-IDF + LSA | — | — | ★★★ | ★★★★ | No |
| LLM + PCA | — | — | ★★★ | ★★★★★ | Optional |
| MRL Truncation | — | — | ★★★ | ★★★★★ | Optional |

---

## Integration with `prepare_modeling_dataset`

```python
from data_preprocessing import prepare_modeling_dataset

dr_config = {
    "mode": "sequential",
    "methods": [
        {"type": "famd", "n_components": 80},
        {"type": "hellinger", "n_features": 60, "use_svm_refinement": True},
    ],
}

prepared = prepare_modeling_dataset(
    target="binary",
    dr_config=dr_config,
)

print(prepared.dr_applied)   # True
print(prepared.X.shape)       # (3023, 60)  ← reduced
```

`PreparedDataset.dr_pipeline` holds the fitted `DRPipeline` for
later inspection or transform of held-out data.

---

## Programmatic API

```python
from dimensionality_reduction import (
    FAMDReducer, HellingerSelector, DRPipeline, build_dr_pipeline,
    detect_column_types,
)

# Detect column types
num, cat, text = detect_column_types(X, categorical_threshold=20)

# Build reducers manually
famd = FAMDReducer(n_components=80)
hell = HellingerSelector(n_features=60, use_svm_refinement=True)

pipeline = DRPipeline(
    steps=[("famd", famd), ("hellinger", hell)],
    mode="sequential",
)
X_reduced = pipeline.fit_transform(X, y)

# Or build from config dict
pipeline = build_dr_pipeline(config["dimensionality_reduction"])
X_reduced = pipeline.fit_transform(X, y)
```

---

## Adding a New Reducer

1. Subclass `BaseReducer` and implement `fit(X, y)` and `transform(X)`.
2. Register it in `_REDUCER_REGISTRY`:
   ```python
   _REDUCER_REGISTRY["my_method"] = MyReducer
   ```
3. Add a YAML config block example to `default_experiment.yaml`.
4. Document it in this file.

---

## Limitations

- **CATPCA ALS** can be slow (O(n · p · max_iter)) on large feature sets. Limit
  `max_iter` to 30–50 for production runs.
- **LLM / MRL reducers** require network access to download models; graceful
  fallback to TF-IDF is automatic when unavailable.
- **HellingerSelector** requires binary or integer class labels; it is not
  currently adapted for `multiclass` targets.
- **FAMD** applies equal weighting to numeric and categorical blocks. If the
  feature mix is heavily imbalanced (e.g. 200 categorical vs 5 numeric), the
  categorical block will dominate the SVD.
- All methods operate on the **baseline (enrollment) snapshot only**. Temporal
  follow-up data, if available, would require longitudinal-aware variants.