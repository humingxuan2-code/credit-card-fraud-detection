import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix

from data_preprocessing import split_scale_and_validate
from utils import FIGURES_DIR
from utils import MODEL_PATH
from utils import REPORTS_DIR
from utils import ensure_output_dirs


def main():
    ensure_output_dirs()

    _, _, X_test, _, _, y_test = split_scale_and_validate()
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    model_name = artifact["model_name"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fraud_precision = precision_score(y_test, y_pred, zero_division=0)
    fraud_recall = recall_score(y_test, y_pred, zero_division=0)
    fraud_f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight", dpi=200)
    plt.close()

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(FIGURES_DIR / "pr_curve.png", bbox_inches="tight", dpi=200)
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(FIGURES_DIR / "roc_curve.png", bbox_inches="tight", dpi=200)
    plt.close()

    summary_path = REPORTS_DIR / "evaluation_summary.txt"
    summary_text = (
        "Credit Card Fraud Detection Evaluation Summary\n"
        "==============================================\n\n"
        f"Selected model: {model_name}\n"
        f"Selected threshold: {threshold:.2f}\n\n"
        "Final Test Metrics\n"
        "------------------\n"
        f"- Fraud precision: {fraud_precision:.6f}\n"
        f"- Fraud recall: {fraud_recall:.6f}\n"
        f"- Fraud F1-score: {fraud_f1:.6f}\n"
        f"- ROC-AUC: {roc_auc:.6f}\n"
        f"- PR-AUC: {pr_auc:.6f}\n\n"
        "Confusion Matrix\n"
        "----------------\n"
        "Rows represent actual labels and columns represent predicted labels.\n\n"
        "|                      | Predicted Non-Fraud | Predicted Fraud |\n"
        "|----------------------|--------------------:|----------------:|\n"
        f"| Actual Non-Fraud     | {tn:>19,} | {fp:>15,} |\n"
        f"| Actual Fraud         | {fn:>19,} | {tp:>15,} |\n\n"
        "Error Interpretation\n"
        "--------------------\n"
        f"- False positives: {fp:,} normal transactions were flagged as fraud. "
        "These cases may increase manual review workload or create customer friction.\n"
        f"- False negatives: {fn:,} fraudulent transactions were missed. "
        "In a fraud detection setting, these are often more costly because they may "
        "represent undetected fraudulent activity.\n"
        f"- True positives: {tp:,} fraudulent transactions were correctly identified.\n"
        f"- True negatives: {tn:,} normal transactions were correctly classified.\n\n"
        "Threshold Tradeoff\n"
        "------------------\n"
        f"The selected threshold of {threshold:.2f} produces a relatively conservative "
        "fraud prediction rule. It keeps false positives low while still detecting "
        "most fraud cases in the held-out test set. Lowering the threshold would "
        "likely increase recall but also increase false positives. Raising the "
        "threshold would likely improve precision but miss more fraud cases.\n\n"
        "Limitations\n"
        "-----------\n"
        "- The dataset is anonymized and does not include richer behavioral, "
        "account-level, or merchant-level features.\n"
        "- The experiment uses a single stratified train/validation/test split rather "
        "than cross-validation or time-based validation.\n"
        "- SMOTE is applied only to the training split, but synthetic oversampling may "
        "not fully represent real fraud behavior.\n"
        "- The model is evaluated as a reproducible offline ML experiment and should "
        "not be treated as production-ready financial risk infrastructure.\n"
        "- No operational cost matrix, investigation capacity, regulatory constraints, "
        "or live distribution-shift analysis is included.\n\n"
        "Reproducibility Notes\n"
        "---------------------\n"
        "- Dataset path: data/raw/creditcard.csv\n"
        "- Test split: 20% of the full dataset\n"
        "- Validation split: 20% of the train-full split\n"
        "- Effective split: approximately 64% train, 16% validation, and 20% test\n"
        "- Stratified splitting is used for both split stages.\n"
        "- Random seed: 42\n"
        "- Model artifact: models/fraud_model.pkl\n"
        "- Evaluation figures: outputs/figures/\n"
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    print("Evaluation figures saved to outputs/figures/")
    print(f"Selected model: {model_name}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Test ROC-AUC: {roc_auc:.6f}")
    print(f"Test PR-AUC: {pr_auc:.6f}")
    print(f"Fraud F1-score: {fraud_f1:.6f}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
