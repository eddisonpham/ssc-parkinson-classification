# Non-Neural Mixture Of Experts

- This pipeline is disjoint from the broad model benchmark.
- Each expert is trained on a separate feature family, then a logistic gate combines expert probabilities.

## Expert Layout

- `low_burden_expert` uses `lightgbm` on `104` features from prefixes: `demographic__, epidemiological__, ehi__`
- `clinical_treatment_expert` uses `xgboost` on `65` features from prefixes: `clinical__, medication__`
- `motor_cognition_expert` uses `catboost` on `140` features from prefixes: `moca__, mds_updrs__, timed_up_go__, schwab_and_england__`
- `questionnaire_expert` uses `balanced_random_forest` on `165` features from prefixes: `apathy_scale__, bai__, bdii__, fatigue_severity_scale__, pdq_8__, pdq_39__, scopa__, parkinson_severity_scale__`

## Metrics

| model | feature_count | accuracy | balanced_accuracy | sensitivity_recall_ap | specificity_pd | precision_ap | f1_ap | auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_of_experts_gate | 4 | 0.6595 | 0.6537 | 0.6471 | 0.6602 | 0.1019 | 0.176 | 0.7809 |
| clinical_treatment_expert | 65 | 0.6744 | 0.6477 | 0.6176 | 0.6778 | 0.1024 | 0.1757 | 0.7643 |
| motor_cognition_expert | 140 | 0.6099 | 0.6412 | 0.6765 | 0.606 | 0.0927 | 0.1631 | 0.7104 |
| expert_average_ensemble | 4 | 0.7603 | 0.6103 | 0.4412 | 0.7793 | 0.1064 | 0.1714 | 0.7434 |
| questionnaire_expert | 165 | 0.3504 | 0.5591 | 0.7941 | 0.324 | 0.0654 | 0.1208 | 0.5791 |
| low_burden_expert | 104 | 0.9421 | 0.5268 | 0.0588 | 0.9947 | 0.4 | 0.1026 | 0.6587 |

## Notes

- The gate is a classical logistic regression, not a neural network.
- Experts are intentionally disjoint by feature family to improve interpretability and reduce modality leakage.
- This pipeline uses a looser default missingness threshold than the broad benchmark so each expert can retain more modality-specific signal.
- The average ensemble is included as a simpler non-gated baseline for the same experts.