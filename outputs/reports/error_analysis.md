# Error Analysis

This report summarizes the main error patterns and limitations of the credit card fraud detection model. It is intended as an evaluation artifact for understanding model behavior, not as a production risk-control document.

## Error Types

The positive class is fraud (`Class = 1`) and the negative class is non-fraud (`Class = 0`).

- True positives: fraudulent transactions correctly flagged as fraud.
- True negatives: normal transactions correctly classified as non-fraud.
- False positives: normal transactions incorrectly flagged as fraud.
- False negatives: fraudulent transactions incorrectly classified as non-fraud.

False negatives are especially important in fraud detection because they may represent fraudulent activity that is not intercepted. False positives also matter because they can create manual review workload, customer friction, and unnecessary alerts.

## Confusion Matrix Interpretation

At the selected threshold of `0.80`, the held-out test set produced the following confusion matrix:

|  | Predicted Non-Fraud | Predicted Fraud |
|---|---:|---:|
| Actual Non-Fraud | 56,846 | 18 |
| Actual Fraud | 22 | 76 |

This means the model correctly identified `76` fraud cases and missed `22` fraud cases. It also correctly classified `56,846` normal transactions while incorrectly flagging `18` normal transactions as fraud.

The final fraud-class metrics were:

- Precision: `0.808511`
- Recall: `0.775510`
- F1-score: `0.791667`
- ROC-AUC: `0.981211`
- PR-AUC: `0.820218`

## Threshold Tradeoff

The final decision threshold was tuned on the validation set and selected as `0.80`. This threshold is more conservative than the default `0.50`, meaning the model requires stronger evidence before labeling a transaction as fraud.

At this threshold, precision and recall are reasonably balanced for a portfolio-style evaluation experiment. Lowering the threshold would likely increase fraud recall and reduce false negatives, but it would also increase false positives. Raising the threshold would likely reduce false positives, but it would miss more fraud cases.

The best threshold depends on business context. A real fraud detection system would need an explicit cost model, investigation capacity constraints, customer-impact analysis, and policy requirements before selecting an operational threshold.

## Practical Risk Notes

- False negatives are high-risk because missed fraud can lead to direct financial loss or delayed response.
- False positives can still be costly because they may trigger unnecessary manual review or customer-facing interventions.
- ROC-AUC is useful for ranking quality, but PR-AUC and threshold-specific metrics are more informative under severe class imbalance.
- The current result should be interpreted as offline model evaluation, not as evidence that the model is ready for live financial deployment.

## Limitations

- The dataset is anonymized and lacks richer context such as merchant category, user history, device data, transaction sequence patterns, and account-level behavior.
- The validation approach uses a stratified random split rather than time-based validation, which may not reflect real fraud drift.
- SMOTE creates synthetic minority-class examples, but synthetic samples may not fully capture real fraud behavior.
- The project does not include calibration analysis, confidence intervals, cost-sensitive threshold optimization, or subgroup performance checks.
- The model has not been tested against live distribution shift, adversarial behavior, compliance requirements, or production monitoring needs.

## Future Work

- Add cost-sensitive evaluation with explicit false-positive and false-negative costs.
- Compare threshold choices under different review-capacity assumptions.
- Add calibration analysis to check whether predicted probabilities are reliable.
- Evaluate performance across transaction amount bands or other meaningful feature slices.
- Use cross-validation or time-based validation to test stability.
- Compare SMOTE with class-weighted learning and additional tabular models such as XGBoost or LightGBM.
