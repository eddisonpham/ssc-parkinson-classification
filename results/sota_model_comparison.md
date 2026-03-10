# State-Of-The-Art Tabular Benchmark

- Target: `binary`
- Samples: `3023`
- Features after filtering: `224`
- Missingness threshold: `0.65`
- Split: `stratified_random_split`
- Models benchmarked: `12`

## Metrics

| model | accuracy | balanced_accuracy | sensitivity_recall_ap | specificity_pd | precision_ap | f1_ap | auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy_ensemble | 0.6264 | 0.7606 | 0.9118 | 0.6095 | 0.122 | 0.2153 | 0.8509 |
| catboost | 0.7653 | 0.6959 | 0.6176 | 0.7741 | 0.14 | 0.2283 | 0.8224 |
| xgboost_hist | 0.8149 | 0.6945 | 0.5588 | 0.8301 | 0.1638 | 0.2533 | 0.7934 |
| balanced_random_forest | 0.757 | 0.6915 | 0.6176 | 0.7653 | 0.1355 | 0.2222 | 0.822 |
| xgboost_deeper | 0.8165 | 0.6815 | 0.5294 | 0.8336 | 0.1593 | 0.2449 | 0.7998 |
| lightgbm_extra_trees | 0.8132 | 0.6521 | 0.4706 | 0.8336 | 0.1441 | 0.2207 | 0.8009 |
| svc_rbf | 0.7008 | 0.6479 | 0.5882 | 0.7075 | 0.107 | 0.181 | 0.711 |
| logistic_regression | 0.7752 | 0.6458 | 0.5 | 0.7916 | 0.125 | 0.2 | 0.7613 |
| lightgbm_gbdt | 0.8132 | 0.6383 | 0.4412 | 0.8354 | 0.1376 | 0.2098 | 0.7875 |
| hist_gradient_boosting | 0.9455 | 0.57 | 0.1471 | 0.993 | 0.5556 | 0.2326 | 0.8225 |
| extra_trees | 0.7537 | 0.5653 | 0.3529 | 0.7776 | 0.0863 | 0.1387 | 0.7483 |
| random_forest | 0.7587 | 0.5541 | 0.3235 | 0.7846 | 0.0821 | 0.131 | 0.7446 |

## Notes

- This benchmark broadens the original baseline with modern gradient-boosting and imbalance-aware ensembles.
- The added advanced models include `XGBoost`, `LightGBM`, `CatBoost`, `BalancedRandomForest`, and `EasyEnsemble`.
- Ranking prioritizes balanced accuracy first because the AP class is rare and clinically important.