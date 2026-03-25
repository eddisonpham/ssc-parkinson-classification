# Evaluation Visualisation Suite
### Parkinsonism vs Atypical Parkinsonism Classification

---

## Table of Contents
1. [Setup](#setup)
2. [Literature Grounding](#literature-grounding)
3. [Reasoning Framework](#reasoning-framework)
4. [Section 1 — Basic Results](#section-1--basic-results)
5. [Section 2 — Comparative Analysis](#section-2--comparative-analysis)
6. [Section 3 — DR Ablation Study](#section-3--dr-ablation-study)
7. [Section 4 — Model Ranking](#section-4--model-ranking)
8. [Design Decisions](#design-decisions)
9. [Output Index](#output-index)

---

## Setup

```bash
# From the project root (directory containing results/)
pip install plotnine pandas numpy

python visualizations/run_all.py
```

Individual modules can be run independently:

```bash
python visualizations/plot_basic.py
python visualizations/plot_comparative.py
python visualizations/plot_ablation.py
python visualizations/plot_ranking.py
```

**Directory layout**

```
<project_root>/
├── results/
│   ├── catpca/          # Categorical PCA
│   ├── pca/             # Standard PCA
│   ├── famd/            # Factor Analysis for Mixed Data
│   ├── hellinger/       # Hellinger Feature Selector
│   └── famd_hellinger/  # FAMD + Hellinger
└── visualizations/
    ├── utils.py
    ├── plot_basic.py
    ├── plot_comparative.py
    ├── plot_ablation.py
    ├── plot_ranking.py
    ├── run_all.py
    ├── DOCUMENTATION.md
    └── outputs/
        ├── basic/
        ├── comparative/
        ├── ablation/
        └── ranking/
```

---

## Literature Grounding

The visualisation choices are grounded in established practices from the
clinical ML benchmarking literature:

| Paper | Contribution | Vis. adopted |
|---|---|---|
| Saito & Rehmsmeier (2015). *PLoS ONE* | Precision–Recall curves superior to ROC for imbalanced datasets | Precision–Recall scatter (§2) |
| Wainberg et al. (2016). *Nat Biotechnol* | Benchmarking framework for biomedical ML; multi-metric comparison | Heatmaps, parallel coordinates |
| Demšar (2006). *JMLR* | Statistical comparison of classifiers; rank-based tests preferred | Rank heatmap, Borda count (§4) |
| Roffo et al. (2016). *ICCV* | Infinite Feature Selection; rank aggregation for feature importance | Rank aggregation methodology |
| Gijsbers et al. (2019). *NeurIPS* | OpenML benchmarking; critical difference diagrams | Bump chart rank trajectory (§4) |
| Chicco & Jurman (2020). *BMC Genomics* | MCC and balanced accuracy superior to accuracy for imbalance | Balanced accuracy as primary metric |
| Takaya et al. (2023). *npj Parkinson's Disease* | ML classification for parkinsonian syndromes | Disease-specific threshold discussion |

**Key design principles from this literature:**

1. **Never use raw accuracy alone** for imbalanced classification.  AP
   prevalence is ~5.6 % in this dataset; a trivial "always predict PD"
   baseline reaches 94 % accuracy.  All visualisations foreground
   *balanced accuracy*, *AUC-ROC*, and *AP sensitivity*.

2. **Rank-based aggregation** (Demšar 2006; Gijsbers 2019) is preferred
   over direct metric averaging when comparing heterogeneous classifiers,
   because it is robust to metric scale and outliers.

3. **Precision–Recall space** (Saito 2015) is more informative than ROC
   space when the positive class (AP) is rare, because it is insensitive
   to the large number of true negatives.

4. **Generalisation analysis** (CV vs test scatter) is standard practice
   in clinical benchmarks to detect CV optimism before deployment.

---

## Reasoning Framework

The visualisation suite is structured in four layers of increasing
analytical depth:

```
Layer 1 – What are the raw numbers?          → Basic Results
Layer 2 – How do metrics relate to each other?→ Comparative Analysis
Layer 3 – Does DR method matter?             → Ablation Study
Layer 4 – Which model wins overall?          → Ranking
```

---

## Section 1 — Basic Results

### 1a. Test Metrics Heatmap (`test_metrics_heatmap.png`)

**Why:** The heatmap is the canonical "first look" figure in clinical ML
papers (e.g., Wainberg 2016).  It encodes all N×M metric–model combinations
in a single view using colour intensity, with numeric annotations for precise
reading.  The red–yellow–green diverging palette (midpoint = 0.5) makes
above/below-chance performance immediately visible.

**Choices:**
- Models sorted top→bottom by mean balanced accuracy (most informative
  single metric for imbalanced data).
- Faceted by DR method so inter-method differences are spatially separable.
- Cell text rounded to 3 decimal places — sufficient precision without
  noise amplification.

---

### 1b. CV Bar Charts with Error Bars (`cv_bars_{dr_key}.png`)

**Why:** Cross-validation results require uncertainty quantification.  The
bar + error bar (±1 SD across folds) combination is the standard in medical
AI papers (e.g., Rajpurkar et al. 2022, *Nat Med*) because it simultaneously
shows point estimate and fold-to-fold variability.

**Choices:**
- One figure per DR method to keep panels uncluttered.
- Fixed y-axis (0–1) across facets so absolute scale is preserved.
- Bars coloured by model (consistent palette throughout suite) so the reader
  can identify a model at a glance without reading axis labels.

---

### 1c. Radar / Spider Chart (`radar_all_models.png`)

**Why:** Spider charts are widely used in clinical decision-support papers
(e.g., He et al. 2021, *eLife*) to display the *multi-dimensional profile*
of each classifier — the shape of the polygon reveals whether a model is a
generalist or a specialist.  Averaging over DR methods gives a stable
"signature" for each model.

**Choices:**
- Values averaged over all DR methods to produce a canonical per-model
  profile independent of preprocessing.
- Polygon fill at very low alpha (0.07) to prevent occlusion with 8 models.
- Axis labels positioned around the perimeter via `coord_polar`.

---

### 1d. Cleveland Dot Plot (`cleveland_dots.png`)

**Why:** Cleveland & McGill (1984) demonstrated that people decode position
on a common scale (dot) more accurately than length (bar) or area (bubble).
Cleveland dot plots have been revived in clinical benchmarks precisely because
of this perceptual advantage.  The lollipop stem guides the eye from baseline
to the dot without wasting ink.

**Choices:**
- Full facet grid (DR × metric) for a complete overview in one figure.
- Model identity encoded by colour only — no redundant shape encoding — to
  keep the plot uncluttered.

---

## Section 2 — Comparative Analysis

### 2a. Sensitivity vs Specificity (`sensitivity_vs_specificity.png`)

**Why:** The sensitivity–specificity plane is the standard diagnostic
performance chart in clinical medicine (Knottnerus & Buntinx 2009,
*BMJ*).  For Parkinsonism classification, it answers the clinical question:
"Can the model catch enough AP cases (sensitivity) while not
misclassifying too many PD patients (specificity)?"  Reference lines at
sensible clinical thresholds guide interpretation.

---

### 2b. Precision–Recall Scatter (`precision_recall_scatter.png`)

**Why:** With 94 % PD prevalence, a classifier that always predicts PD
achieves AUC-ROC ≈ 0.5 but AP precision ≈ 0 and AP recall = 0.  The
PR plane exposes this failure mode.  Saito & Rehmsmeier (2015) showed PR
is strictly superior to ROC for class-imbalanced tasks.

**Choices:**
- Each point is one model × DR combination (40 total), coloured by model
  and shaped by DR method.
- The dotted diagonal from (0,1) to (1,0) is the F1 iso-curve where
  precision = recall (F1 is maximised here).

---

### 2c. AUC-ROC vs Balanced Accuracy (`auc_vs_balanced_accuracy.png`)

**Why:** AUC-ROC is threshold-independent (evaluates the entire ranking);
balanced accuracy is threshold-dependent (evaluates a single cut-point at
0.5 by default).  Divergence between the two reveals whether a model's
decision boundary is well-calibrated.  A model far above the diagonal has
good discrimination but a poorly calibrated default threshold.

---

### 2d. Metric Correlation Heatmap (`metric_correlation_heatmap.png`)

**Why:** Redundant metrics provide no additional information.  If AUC-ROC
and balanced accuracy correlate at r = 0.95, reporting both is decorative.
Conversely, low correlation (e.g., accuracy vs AP sensitivity) reveals that
the two metrics are capturing different aspects of performance — both are
needed for a complete picture.  This chart informs metric selection for
concise reporting.

**Choices:**
- Pearson r across all 40 experiments (8 models × 5 DR methods) to get
  a robust estimate.
- Blue–white–red diverging palette: blue = negative correlation (metrics
  trade off), red = positive (metrics agree).

---

### 2e. Parallel Coordinates (`parallel_coordinates.png`)

**Why:** Parallel coordinates (Inselberg 1985) are the standard tool for
visualising high-dimensional tabular data.  Each polyline represents one
experiment; crossing lines reveal inversions between adjacent metrics.
Min-max normalisation per metric puts all axes on the same visual scale.

**Choices:**
- Lines coloured by model so model-specific patterns are visible through
  transparency stacking.
- Alpha = 0.42 to reveal density (many overlapping lines = consensus).

---

## Section 3 — DR Ablation Study

### 3a. Ablation Line Plot (`dr_ablation_lines.png`)

**Why:** A connected line plot where x = DR method is the standard
ablation figure in ML papers.  The slope of each line segment directly
encodes sensitivity to the DR choice.  Flat lines indicate DR-robust
models; steep lines indicate that the DR method is a critical hyperparameter.

---

### 3b. Grouped Bar Chart (`dr_ablation_bars.png`)

**Why:** For four key clinical metrics, the grouped bar chart allows
direct quantity comparison within each (model, DR) cell.  Unlike the line
plot, absolute values are easily read off the y-axis.  The two views are
complementary: line plot for trend; bar chart for magnitude.

---

### 3c. DR × Model Heatmap (`dr_model_heatmap.png`)

**Why:** The two-dimensional heatmap (DR method × model) is the most
information-dense ablation summary.  It simultaneously shows the best
DR–model combination (top-right "hot spot") and the worst (bottom-left
"cold spot") for each metric.  Faceting by metric preserves full detail
without losing the 2D ablation structure.

---

### 3d. Stability Chart (`dr_stability.png`)

**Why:** Standard deviation of test metric across DR methods quantifies
DR-sensitivity per model.  A model with high SD should not be deployed
without first determining the optimal DR method for the specific dataset.
A model with low SD is "plug-and-play" — less sensitive to DR choice.
Ordering by mean SD surfaces the most stable models at the bottom.

---

## Section 4 — Model Ranking

### 4a. Rank Heatmap (`rank_heatmap.png`)

**Why:** Directly averaging metric values across DR methods can be
misleading because metrics have different scales and variances.
Ordinal rank aggregation (Demšar 2006) is scale-invariant and robust to
outliers.  The heatmap shows per-metric ranks so the analyst can see
*where* each model wins and loses, not just an opaque aggregate score.

---

### 4b. Borda Count (`borda_count.png`)

**Why:** Borda count is the simplest and most interpretable rank
aggregation method.  It has been used in computational biology (e.g., Roffo
2016) and benchmarking pipelines (Gijsbers 2019) as an intuitive
alternative to statistical tests such as Wilcoxon signed-rank.  Each model
receives (N − rank) points per (DR, metric) pair; the aggregate is a
single comparable scalar.

**Interpretation:**
- Maximum possible score = (N−1) × n_metrics × n_DR_methods
  = 7 × 7 × 5 = 245.
- A score of 245 would mean rank 1 in every metric × DR combination.

---

### 4c. Generalisation Gap (`generalization_gap.png`)

**Why:** CV optimism — the tendency for cross-validation scores to
overestimate test performance — is a known failure mode in medical ML
(Roberts et al. 2021, *Nat Mach Intell*).  The diagonal scatter (CV mean
vs test) is the standard diagnostic.  Points systematically below the
diagonal indicate that CV tuning has overfit to the validation folds.

**Choices:**
- Faceted by four key metrics so the reader can see which metric is most
  affected by CV optimism.
- Shape encodes DR method, colour encodes model — consistent with all
  other scatter plots in the suite.

---

### 4d. Bump Chart (`bump_chart.png`)

**Why:** Bump charts (also called "slopegraphs" when n=2 axes) are the
standard tool for tracking rank changes across ordered categories.  In our
context, crossing lines between metrics reveal *metric-dependent rank
inversions*: a model may be rank 1 on AUC-ROC but rank 6 on AP
sensitivity — a critical clinical distinction.

**Choices:**
- Labelled dot for each rank to remove the need for a separate legend look-up.
- y-axis reversed (1 at top) so "better" is visually higher.
- Thick lines (1.3 pt) with moderate alpha to handle crossing.

---

## Design Decisions

### Colour Palette

The 8-model palette is based on the Tableau 10 categorical scheme, which
was empirically optimised for discriminability (Wang et al. 2018, *IEEE
TVCG*).  All colours remain distinguishable at 8 % greyscale opacity.
The 5-DR-method palette uses a warm sequential ramp (dark teal → coral)
inspired by ColorBrewer's "Spectral" scheme.

### Typography

`DejaVu Sans` is used throughout — it is guaranteed present in any
matplotlib installation and has excellent legibility at small sizes common
in faceted plots.

### Resolution & Dimensions

All figures saved at 180 DPI.  Width and height are set per-figure to
match the aspect ratio of the content:
- Single-panel scatters: 10–11 × 7–7.5 in.
- Multi-panel facets: 14–20 × 8–14 in.
- Square plots (radar, correlation): 9–11 × 7–10 in.

These dimensions produce ~1600–3600 px on the long axis, suitable for
journal submission (typically requires ≥ 300 DPI at final print size).

---

## Output Index

| File | Section | Type |
|---|---|---|
| `basic/test_metrics_heatmap.png` | 1a | Tile heatmap |
| `basic/cv_bars_{dr}.png` (×5) | 1b | Bar + errorbar |
| `basic/radar_all_models.png` | 1c | Polar polygon |
| `basic/cleveland_dots.png` | 1d | Lollipop |
| `comparative/sensitivity_vs_specificity.png` | 2a | Scatter |
| `comparative/precision_recall_scatter.png` | 2b | Scatter |
| `comparative/auc_vs_balanced_accuracy.png` | 2c | Scatter |
| `comparative/metric_correlation_heatmap.png` | 2d | Tile heatmap |
| `comparative/parallel_coordinates.png` | 2e | Polyline |
| `ablation/dr_ablation_lines.png` | 3a | Line + point |
| `ablation/dr_ablation_bars.png` | 3b | Grouped bar |
| `ablation/dr_model_heatmap.png` | 3c | Tile heatmap |
| `ablation/dr_stability.png` | 3d | Bar |
| `ranking/rank_heatmap.png` | 4a | Tile heatmap |
| `ranking/borda_count.png` | 4b | Horizontal bar |
| `ranking/generalization_gap.png` | 4c | Scatter |
| `ranking/bump_chart.png` | 4d | Bump / slope |

**Total: 21 figures** (5 per DR method for CV bars + 16 cross-method figures)