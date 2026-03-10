# Model Comparison

- Target: `binary`
- Samples: `3023`
- Features after filtering: `224`
- Missingness threshold: `0.65`
- Split: `stratified_random_split`

## Metrics

| model | accuracy | balanced_accuracy | sensitivity_recall_ap | specificity_pd | precision_ap | f1_ap | auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| svc_rbf | 0.7008 | 0.6479 | 0.5882 | 0.7075 | 0.107 | 0.181 | 0.711 |
| logistic_regression | 0.7752 | 0.6458 | 0.5 | 0.7916 | 0.125 | 0.2 | 0.7613 |
| hist_gradient_boosting | 0.9455 | 0.57 | 0.1471 | 0.993 | 0.5556 | 0.2326 | 0.8225 |
| extra_trees | 0.7537 | 0.5653 | 0.3529 | 0.7776 | 0.0863 | 0.1387 | 0.7471 |
| random_forest | 0.7587 | 0.5541 | 0.3235 | 0.7846 | 0.0821 | 0.131 | 0.7475 |

## Notes

- Binary experiments treat `AP` as the positive class because sensitivity to rare atypical parkinsonism is clinically important.
- Direct diagnosis-leakage columns from `clinical.csv` are excluded before modeling.
- Missingness filtering is intentionally conservative because the dataset is highly sparse.