# Repository Guide And Methodology

## Purpose
This repository is an experimental case-study workspace for classifying Parkinson's disease (`PD`) versus atypical parkinsonism (`AP`) using Canadian Open Parkinson Network baseline data. The current codebase is organized to support three things:

1. Fast exploratory data analysis over the CSV bundle in `data/ssc_data/`
2. Reproducible preprocessing with explicit leakage controls
3. Benchmarking several baseline classifiers for the clinically important `PD` vs `AP` task

The original literature context lives in `Parkinsons_Classification_Literature_Review.md`, and the case-study framing lives in `README.md` and `INSTRUCTION.md`.

## Current Repository Layout
```text
ssc-case-study-parkinson-classification/
├── INSTRUCTION.md
├── README.md
├── README2.md
├── Parkinsons_Classification_Literature_Review.md
├── configs/
│   └── default_experiment.yaml
├── data/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── exploratory_data_analysis.ipynb
│   └── ssc_data/
│       ├── enrollement.csv
│       ├── demographic.csv
│       ├── clinical.csv
│       ├── epidemiological.csv
│       ├── medication.csv
│       ├── moca.csv
│       ├── mds-updrs.csv
│       ├── ...
├── results/
│   ├── classification_reports.json
│   ├── experiment_summary.json
│   ├── model_comparison.csv
│   └── model_comparison.md
├── scripts/
│   └── train_models.py
└── requirements.txt
```

## Data Assets
The raw data bundle is baseline-only in the current workspace snapshot, and the tables join cleanly on `Project key`.

Observed baseline cohort counts from `enrollement.csv`:

- `PD`: 2,852
- `AP`: 171
- `Healthy control`: 410
- Missing enrolment group: 108

Important label sources:

- Primary modeling label: `Enrolment Group` in `data/ssc_data/enrollement.csv`
- AP subtype detail: diagnosis-specific fields in `data/ssc_data/clinical.csv`

Important caveat:

- The instructions mention a data dictionary and an Excel workbook, but no usable data dictionary file is present in the current workspace. Because of that, implementation-level mapping is provisional and based on the README definitions plus the nature of each questionnaire or assessment.

## Methodology
### 1. Target definition
The code supports:

- Binary target: `PD` vs `AP`
- Multiclass target: `PD` vs `AP` vs `HC`

The default experiment focuses on the primary case-study task, `PD` vs `AP`, and excludes `HC` from training.

### 2. Feature assembly
`data/data_preprocessing.py` builds a patient-level master table by:

- Loading enrollment metadata and target labels
- Merging selected baseline CSVs on `Project key`
- Prefixing feature names by source table
- Removing operational metadata fields that are not useful predictors

The default feature bundle uses:

- `demographic.csv`
- `clinical.csv`
- `epidemiological.csv`
- `medication.csv`
- `moca.csv`
- `mds-updrs.csv`
- `apathy_scale.csv`
- `bai.csv`
- `bdii.csv`
- `fatigue_severity_scale.csv`
- `pdq_8.csv`
- `pdq_39.csv`
- `scopa.csv`
- `timed_up_go.csv`
- `parkinson_severity_scale.csv`
- `schwab_&_england.csv`
- `ehi.csv`

### 3. Leakage control
Direct diagnosis fields from `clinical.csv` are explicitly removed before modeling, including:

- diagnosed/not diagnosed flags
- diagnosis subtype fields
- diagnosis certainty fields
- diagnosis confirmation fields
- diagnosis dates and diagnosis-age fields

This is important because those fields would otherwise let the model recover the label directly instead of learning clinically relevant patterns.

### 4. Missing data handling
The dataset is highly sparse, which is expected for this case study. The current pipeline handles this by:

- dropping features with more than `65%` missingness
- keeping only non-constant features
- using median imputation for numeric fields
- using most-frequent imputation for categorical fields
- applying one-hot encoding to categorical variables

This is intentionally conservative and easy to audit.

### 5. Model comparison strategy
The default training script benchmarks five baseline models:

- Logistic regression
- Random forest
- Extra trees
- RBF-kernel SVM
- Histogram gradient boosting

The evaluation split is currently:

- stratified random train/test split
- `80/20` split
- fixed seed `42`

The CLI also supports optional site-based holdout evaluation through `--site-holdout`.

### 6. Metric emphasis
Because `AP` is the rare but clinically important class, the primary ranking signal is not plain accuracy alone. The results emphasize:

- balanced accuracy
- AP sensitivity/recall
- PD specificity
- AUC-ROC

## Current Experimental Results
The current default binary experiment used:

- `3023` labeled `PD/AP` samples
- `224` retained features after filtering
- class distribution of `2852 PD` and `171 AP`

Best observed models from `results/model_comparison.md`:

| Model | Accuracy | Balanced Accuracy | AP Recall | AUC |
| --- | --- | --- | --- | --- |
| `svc_rbf` | 0.7008 | 0.6479 | 0.5882 | 0.7110 |
| `logistic_regression` | 0.7752 | 0.6458 | 0.5000 | 0.7613 |
| `hist_gradient_boosting` | 0.9455 | 0.5700 | 0.1471 | 0.8225 |

Interpretation:

- `HistGradientBoosting` achieves the highest raw accuracy and AUC, but it misses many AP cases.
- `SVC` gives the best AP sensitivity in the current benchmark, which may matter more clinically.
- `LogisticRegression` is a strong implementation-friendly baseline because it is simpler and nearly matches the best balanced accuracy.

## File-by-File Purpose
### `data/data_preprocessing.py`
Core data utility module. It provides:

- table catalog metadata
- schema summaries for EDA
- cohort/label extraction
- merged feature-table construction
- missingness summaries
- prepared modeling datasets for binary or multiclass experiments

### `data/exploratory_data_analysis.ipynb`
Notebook entry point for:

- listing each CSV and its role
- inspecting schema and missingness
- confirming target variables
- reviewing candidate feature groups

### `scripts/train_models.py`
Main experiment runner. It:

- reads config from `configs/default_experiment.yaml`
- prepares the dataset
- performs train/test splitting
- trains the model set
- writes metrics and reports into `results/`

### `results/`
Generated benchmark outputs:

- `model_comparison.csv`: machine-readable summary table
- `model_comparison.md`: readable summary for the repo
- `classification_reports.json`: per-model class metrics
- `experiment_summary.json`: dataset and split metadata

## How To Run
Run the default benchmark:

```bash
python scripts/train_models.py
```

Run a multiclass benchmark:

```bash
python scripts/train_models.py --target multiclass
```

Run a site holdout experiment:

```bash
python scripts/train_models.py --site-holdout "<site name>"
```

## Recommended Next Iterations
- Add a dedicated Level 1 and Level 2 feature subset once the formal data dictionary is available.
- Compare random split performance against site holdout performance.
- Add calibrated threshold tuning for AP sensitivity, since recall is more important than raw accuracy for the rare class.
- Explore multiclass performance only after the binary AP-sensitive baseline is stable.
