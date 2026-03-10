# Parkinson Versus Atypical Parkinsonism Classification

Experimental case-study repository for classifying Parkinson's disease (`PD`) versus atypical parkinsonism (`AP`) using Canadian Open Parkinson Network (`C-OPN`) baseline data.

This repository now contains:
- reusable preprocessing and schema-audit utilities
- exploratory analysis for the multi-table C-OPN snapshot
- baseline and expanded tabular model benchmarks
- text-aware feature selection for non-numeric clinical responses
- a separate non-neural mixture-of-experts pipeline
- a separate evaluation and graph-generation pipeline with publication-style plots

## Repository Structure
```text
ssc-case-study-parkinson-classification/
├── INSTRUCTION.md
├── README.md
├── README2.md
├── Parkinsons_Classification_Literature_Review.md
├── configs/
│   ├── default_experiment.yaml
│   └── advanced_experiment.yaml
├── data/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── exploratory_data_analysis.ipynb
│   └── ssc_data/
├── evaluation/
│   └── plots/
│       ├── config.py
│       ├── data_loading.py
│       ├── discussion_plan.md
│       ├── generate_plots.py
│       ├── plotting.py
│       ├── refresh_results.py
│       └── output/
├── results/
├── scripts/
│   ├── feature_selection.py
│   ├── train_models.py
│   ├── train_mixture_of_experts.py
│   └── train_state_of_the_art_models.py
└── requirements.txt
```

## Dataset Summary
The current repository snapshot is a baseline-only, patient-level, multi-table export keyed by `Project key`.

Observed cohort counts from `enrollement.csv`:
- `PD`: `2852`
- `AP`: `171`
- `HC`: `410`
- missing cohort label: `108`

Primary label sources:
- cohort-level group label from `data/ssc_data/enrollement.csv`
- AP subtype detail from `data/ssc_data/clinical.csv`

Important caveat:
- the project brief references a data dictionary and an Excel workbook, but no authoritative data dictionary is present in this workspace
- implementation-level mapping in the analysis is therefore provisional

## Preprocessing Design
The preprocessing logic lives in `data/data_preprocessing.py`.

Key design choices:
- patient-level merge on `Project key`
- feature names always include the source CSV prefix
- duplicate normalized headers from the same CSV are disambiguated with suffixes such as `__dup2`
- diagnosis-leakage columns from `clinical.csv` are explicitly removed before modeling
- default modeling filter drops features with more than `65%` missingness
- numeric features use median imputation
- categorical features use most-frequent imputation plus one-hot encoding

This keeps the pipeline reproducible and avoids accidental target leakage.

## Modeling Pipelines
### 1. Baseline benchmark
`scripts/train_models.py`

Benchmarks:
- logistic regression
- random forest
- extra trees
- RBF SVM
- histogram gradient boosting

Primary outputs:
- `results/model_comparison.csv`
- `results/model_comparison.md`
- `results/classification_reports.json`
- `results/experiment_summary.json`

### 2. Text-aware feature selection
`scripts/feature_selection.py`

This pipeline handles the large number of non-numeric C-OPN columns by:
- converting row-level non-numeric fields into `column=value` text tokens
- vectorizing them with TF-IDF
- combining those with numeric features
- selecting features via a sparse linear selector
- fitting a final logistic model on the selected set

Primary outputs:
- `results/selected_features.csv`
- `results/feature_selection_summary.json`
- `results/feature_selection_metrics.json`
- `results/feature_selection_predictions.csv`
- `results/feature_selection.md`

### 3. Expanded state-of-the-art tabular benchmark
`scripts/train_state_of_the_art_models.py`

Models benchmarked:
- logistic regression
- random forest
- extra trees
- RBF SVM
- histogram gradient boosting
- balanced random forest
- easy ensemble
- XGBoost hist
- XGBoost deeper variant
- LightGBM GBDT
- LightGBM extra-trees variant
- CatBoost

Primary outputs:
- `results/sota_model_comparison.csv`
- `results/sota_model_comparison.md`
- `results/sota_predictions.csv`
- `results/sota_classification_reports.json`
- `results/sota_experiment_summary.json`

### 4. Non-neural mixture of experts
`scripts/train_mixture_of_experts.py`

This is a separate, disjoint expert-based pipeline. It trains experts on different feature families:
- low-burden demographic and exposure features
- clinical and treatment features
- motor and cognition features
- questionnaire features

A classical logistic regression gate combines expert probabilities. No neural networks are used.

Primary outputs:
- `results/mixture_of_experts_comparison.csv`
- `results/mixture_of_experts_comparison.md`
- `results/mixture_of_experts_predictions.csv`
- `results/mixture_of_experts_reports.json`
- `results/mixture_of_experts_summary.json`

## Current Results
All current benchmark results were refreshed together, as recorded in `results/result_manifest.json`.

### Data used in the main binary task
- labeled `PD/AP` samples: `3023`
- retained tabular features after default filtering: `224`
- class distribution: `2852 PD`, `171 AP`

### Best baseline result
From `results/model_comparison.md`:
- `svc_rbf`: balanced accuracy `0.6479`, AP recall `0.5882`, AUC `0.7110`

### Best expanded benchmark result
From `results/sota_model_comparison.md`:
- `easy_ensemble`: balanced accuracy `0.7606`, AP recall `0.9118`, AUC `0.8509`

Other strong expanded models:
- `catboost`: balanced accuracy `0.6959`, AUC `0.8224`
- `xgboost_hist`: balanced accuracy `0.6945`, AUC `0.7934`
- `balanced_random_forest`: balanced accuracy `0.6915`, AUC `0.8220`

### Text-aware feature-selection result
From `results/feature_selection.md`:
- balanced accuracy `0.7535`
- AUC `0.8067`
- selected features: `165`
- selected text-derived features: `94`
- selected numeric features: `71`

### Mixture-of-experts result
From `results/mixture_of_experts_comparison.md`:
- `mixture_of_experts_gate`: balanced accuracy `0.6537`, AUC `0.7809`

Interpretation:
- the separate expert decomposition is useful analytically and interpretively
- the strongest single-model benchmark still outperforms the current gated MoE pipeline

## Evaluation And Plotting
The evaluation pipeline is intentionally separate from the modeling scripts.

Files:
- `evaluation/plots/refresh_results.py`
- `evaluation/plots/generate_plots.py`
- `evaluation/plots/discussion_plan.md`

What it does:
- refreshes the latest model results before plotting
- loads refreshed result artifacts only
- generates neat, reusable `plotnine` figures into `evaluation/plots/output`
- supports discussion around imbalance, sparsity, trade-offs, implementation burden, text-derived signals, and expert decomposition

Generated plot set currently includes:
- cohort balance
- site distribution
- table sparsity
- table width versus missingness
- balanced accuracy across pipelines
- AP recall versus PD specificity trade-off
- AUC versus balanced accuracy
- advanced-model metric heatmap
- implementation complexity versus performance
- feature-selection composition
- top selected features
- MoE component performance
- expert feature count versus performance
- ROC curves
- precision-recall curves
- average performance by model family

## How To Run
### Baseline benchmark
```bash
python scripts/train_models.py
```

### Text-aware feature selection
```bash
python scripts/feature_selection.py
```

### Expanded tabular benchmark
```bash
python scripts/train_state_of_the_art_models.py
```

### Non-neural mixture of experts
```bash
python scripts/train_mixture_of_experts.py
```

### Refresh all results
```bash
python evaluation/plots/refresh_results.py
```

### Generate all evaluation plots
```bash
python evaluation/plots/generate_plots.py
```

## Notes
- the main clinical task remains `PD` versus `AP`
- model ranking emphasizes `balanced_accuracy`, `AP recall`, and `AUC-ROC` instead of raw accuracy alone
- the repository is experimental, but the codebase is now organized around reusable preprocessing, modeling, and evaluation components
