# Evaluation Discussion Plan

## Aim
Frame the analysis around clinically meaningful discrimination of Parkinson's disease (`PD`) versus atypical parkinsonism (`AP`), not raw accuracy alone.

## Literature-Grounded Discussion Threads
### 1. Rare-Class Detection Matters More Than Overall Accuracy
- The literature review emphasizes atypical red flags such as early falls, severe autonomic dysfunction, poor levodopa response, and rapid progression.
- Because `AP` is rare in the dataset, a model can look strong on accuracy while still missing many clinically important atypical cases.
- Discussion should prioritize `balanced_accuracy`, `AP recall`, and `AUC-ROC`, with precision interpreted in the context of screening versus confirmatory use.

### 2. Differential Diagnosis Is Clinically Heterogeneous
- `AP` is not one disease; it mixes `PSP`, `MSA`, `CBS`, `DLB`, and other rarer conditions.
- This makes binary `PD` versus `AP` classification intrinsically noisy and supports discussing subtype heterogeneity as a likely ceiling on performance.
- Plots should help explain why some models favor sensitivity while others favor specificity.

### 3. Implementation Feasibility Is Part Of Model Quality
- The case study and literature both stress deployment barriers, actionability, and workflow fit.
- Discussion should compare lower-burden models against more complex ensembles, highlighting whether extra complexity materially improves AP-sensitive performance.
- This is where implementation complexity versus balanced accuracy is important.

### 4. Missingness And Modality Coverage Shape What The Models Learn
- The C-OPN snapshot is highly sparse, especially for richer assessments and long questionnaires.
- A key analysis thread is how sparsity affects model families and whether some modalities remain informative despite incomplete coverage.
- Table-level missingness plots and expert-family plots should support this discussion.

### 5. Text-Like Clinical Responses Add Signal
- The current repository contains many non-numeric bilingual and semi-free-text response fields.
- The text-aware sparse selector should be discussed as a practical way to absorb these features without jumping directly to large neural models.
- The selected-feature plots should help show what kinds of responses survive selection.

### 6. Expert Decomposition Helps Interpretation Even If It Does Not Win
- A non-neural mixture-of-experts pipeline is useful for analyzing which clinical domains are informative.
- Discussion should compare expert-specific strengths against the gated ensemble and the strongest monolithic models.
- Even if the MoE pipeline is not the top performer, it is still valuable analytically.

### 7. Reporting Should Follow TRIPOD/STARD Logic
- The final narrative should clearly state the target, cohort composition, missing-data handling, predictor families, validation split, and clinically relevant performance metrics.
- Limitations should include imbalance, subtype heterogeneity, missingness, and the single-snapshot nature of the current extracted dataset.

## Plot-Driven Narrative Order
1. Start with cohort imbalance and site distribution to frame the dataset.
2. Show table sparsity and width to explain why preprocessing and feature filtering matter.
3. Present the broad benchmark and the AP recall versus specificity trade-off.
4. Compare model families by implementation burden and balanced accuracy.
5. Show what the text-aware selector retained and how it performed.
6. End with the mixture-of-experts decomposition to discuss modality-level signal and interpretability.
