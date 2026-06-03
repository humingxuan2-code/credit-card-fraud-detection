# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions under extreme class imbalance.

This repository is designed as a portfolio-ready ML project for research applications, RA opportunities, and job search presentation. It focuses on practical modeling decisions for real-world fraud detection: imbalanced learning, threshold tuning, model comparison, reproducible evaluation, and readable project structure.

## Why This Project

Credit card fraud detection is a classic but still highly practical machine learning problem:

- the positive class is extremely rare
- accuracy alone is misleading
- recall and precision must be balanced carefully
- decision thresholds matter as much as model choice

This project treats fraud detection as a realistic imbalanced classification task rather than only a toy benchmark.

## Project Highlights

- Built a complete fraud detection pipeline from raw CSV to saved model artifact
- Compared `Logistic Regression` and `Random Forest` on a validation split
- Applied `SMOTE` to address class imbalance in the training set
- Tuned the classification threshold on the validation set instead of using a fixed `0.50`
- Evaluated the final model using `Precision`, `Recall`, `F1`, `ROC-AUC`, and `PR-AUC`
- Generated reusable reports and visualizations for GitHub presentation

## Dataset

Source:

`https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

After downloading, place the dataset here:

```text
data/raw/creditcard.csv
```

The dataset contains anonymized PCA-based transaction features, transaction amount, and a binary fraud label:

- `Class = 0`: normal transaction
- `Class = 1`: fraudulent transaction

## Method

### Pipeline

1. Load and validate the raw transaction dataset
2. Split data into train, validation, and test sets with stratification
3. Standardize the `Amount` feature
4. Apply `SMOTE` on the training split only
5. Train multiple candidate models
6. Select the best model by validation `PR-AUC`
7. Tune the decision threshold on the validation set
8. Evaluate on the held-out test set
9. Save the selected model, threshold, feature metadata, and preprocessing objects

### Models Compared

- `LogisticRegression`
- `RandomForestClassifier`

### Data Split

The split ratios are defined in `src/data_preprocessing.py`:

- test set: `20%` of the full dataset
- validation set: `20%` of the remaining train-full split
- final effective split: approximately `64%` train, `16%` validation, and `20%` test

All splits use stratification so that the rare fraud class is represented consistently across train, validation, and test sets.

## Evaluation Strategy

Accuracy is not a reliable primary metric for this task because the dataset is extremely imbalanced. A model that predicts every transaction as non-fraud would still achieve very high accuracy while failing to detect the minority fraud class.

This project emphasizes metrics that better reflect ranking quality, minority-class detection, and decision-threshold behavior:

- `PR-AUC`: measures the precision-recall tradeoff across thresholds and is especially useful when the positive class is rare.
- `ROC-AUC`: measures how well the model separates fraud from non-fraud across thresholds, but it can look optimistic under severe class imbalance.
- `F1-score`: balances fraud precision and fraud recall at a chosen threshold.
- `Confusion matrix`: shows the concrete number of true positives, false positives, false negatives, and true negatives.

Threshold tuning is important because the default `0.50` decision threshold may not match the practical cost of fraud detection. This project tunes the threshold on the validation set and then evaluates the selected threshold on a held-out test set.

## Experimental Result

### Validation Leaderboard

| Model | Validation PR-AUC | Validation F1 | Validation Recall | Best Threshold |
|---|---:|---:|---:|---:|
| Random Forest | 0.7575 | 0.7801 | 0.6962 | 0.80 |
| Logistic Regression | 0.6801 | 0.3699 | 0.8101 | 0.95 |

### Final Test Performance

Selected model: `Random Forest`  
Selected threshold: `0.80`

| Metric | Value |
|---|---:|
| Fraud Precision | 0.8085 |
| Fraud Recall | 0.7755 |
| Fraud F1-score | 0.7917 |
| ROC-AUC | 0.9812 |
| PR-AUC | 0.8202 |

### Final Confusion Matrix

At threshold `0.80`, the held-out test set confusion matrix is:

|  | Predicted Non-Fraud | Predicted Fraud |
|---|---:|---:|
| Actual Non-Fraud | 56,846 | 18 |
| Actual Fraud | 22 | 76 |

### Interpretation

- The model achieves strong separation performance with `ROC-AUC = 0.9812`
- `PR-AUC = 0.8202` indicates good fraud ranking quality under severe imbalance
- Fraud precision and recall are reasonably balanced after threshold tuning
- Random Forest clearly outperforms the logistic baseline on validation `PR-AUC`

## Error Analysis

False positives are normal transactions that the model flags as fraud. In practice, these can create review workload or customer friction. False negatives are fraudulent transactions that the model misses. In fraud detection, false negatives are often more dangerous because they may correspond to direct financial loss or delayed fraud response.

At the selected threshold of `0.80`, the model produced `18` false positives and `22` false negatives on the held-out test set. This threshold favors a relatively conservative fraud prediction: precision is high enough to reduce unnecessary alerts, while recall still captures most fraud cases in the test set.

The precision-recall tradeoff remains central. Lowering the threshold would likely catch more fraud cases but increase false positives. Raising the threshold would likely reduce false positives but miss more fraud cases. The selected threshold should therefore be viewed as an evaluation choice for this project, not a production policy.

Limitations:

- The dataset is anonymized and does not include richer behavioral, temporal, or account-level features.
- The experiment uses a single train/validation/test split rather than cross-validation or time-based validation.
- SMOTE is applied only to the training split, but synthetic oversampling may not fully represent real fraud behavior.
- The model and threshold are not validated against operational cost, alert capacity, regulatory requirements, or live distribution shift.
- This project is intended for reproducible ML evaluation practice, not direct deployment in a real financial risk system.

## Visualizations

### Confusion Matrix

![Confusion Matrix](outputs/figures/confusion_matrix.png)

### Precision-Recall Curve

![PR Curve](outputs/figures/pr_curve.png)

### ROC Curve

![ROC Curve](outputs/figures/roc_curve.png)

## Repository Structure

```text
credit-card-fraud-detection/
+-- data/
|   +-- raw/
|   |   +-- creditcard.csv
|   +-- processed/
+-- notebooks/
+-- src/
|   +-- data_preprocessing.py
|   +-- train.py
|   +-- evaluate.py
|   +-- predict.py
|   +-- utils.py
+-- models/
+-- outputs/
|   +-- figures/
|   +-- reports/
+-- requirements.txt
+-- .gitignore
+-- README.md
```

## Reproducibility / How to Reproduce

Recommended environment:

- Python `3.11`
- A local virtual environment
- The Kaggle Credit Card Fraud Detection dataset

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the dataset from Kaggle:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Place the downloaded CSV file here:

```text
data/raw/creditcard.csv
```

Open the exploratory notebook if you want a compact data analysis walkthrough:

```bash
jupyter notebook notebooks/eda.ipynb
```

Train the model:

```bash
python src/train.py
```

Generate evaluation outputs:

```bash
python src/evaluate.py
```

Use the saved model for prediction:

```bash
python src/predict.py
```

The main outputs are saved under:

- `models/fraud_model.pkl`
- `outputs/reports/`
- `outputs/figures/`

## Output Files

After running the pipeline, the following outputs are generated:

- `models/fraud_model.pkl`
- `outputs/reports/training_report.txt`
- `outputs/reports/evaluation_summary.txt`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/pr_curve.png`
- `outputs/figures/roc_curve.png`
- `notebooks/eda.ipynb`

## Skills Demonstrated

- imbalanced classification
- validation-based model selection
- threshold optimization
- sklearn pipeline-style project organization
- experiment reporting and reproducibility
- GitHub-ready ML project presentation

## AI-assisted Development Note

Codex was used to assist with debugging, refactoring, documentation, and GitHub repository organization, while the model logic, evaluation results, and final conclusions were independently reviewed and verified.

## Future Work

- add `XGBoost` or `LightGBM` for stronger tabular performance
- add cross-validation and confidence intervals
- test cost-sensitive learning against SMOTE-based resampling
- add experiment tracking with MLflow or Weights & Biases
- package the model behind a small API or demo app
