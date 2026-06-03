import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import split_scale_and_validate
from utils import MODEL_PATH
from utils import REPORTS_DIR
from utils import ensure_output_dirs


def evaluate_scores(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 1.00, 0.05)
    scored_thresholds = [evaluate_scores(y_true, y_prob, threshold) for threshold in thresholds]
    best_result = max(scored_thresholds, key=lambda item: (item["f1"], item["recall"], item["precision"]))
    return best_result, scored_thresholds


def main():
    ensure_output_dirs()

    X_train, X_val, X_test, y_train, y_val, y_test, amount_scaler = split_scale_and_validate(
        return_scaler=True
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    leaderboard = []
    best_model_name = None
    best_model = None
    best_threshold = 0.5
    best_validation_result = None
    best_validation_pr_auc = -1.0
    best_threshold_history = None

    for model_name, model in candidate_models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        validation_pr_auc = average_precision_score(y_val, y_val_prob)
        threshold_result, threshold_history = find_best_threshold(y_val, y_val_prob)

        leaderboard.append(
            {
                "model_name": model_name,
                "validation_pr_auc": validation_pr_auc,
                "validation_f1": threshold_result["f1"],
                "validation_recall": threshold_result["recall"],
                "validation_precision": threshold_result["precision"],
                "best_threshold": threshold_result["threshold"],
            }
        )

        if validation_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = validation_pr_auc
            best_model_name = model_name
            best_model = model
            best_threshold = threshold_result["threshold"]
            best_validation_result = threshold_result
            best_threshold_history = threshold_history

    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    classification_text = classification_report(y_test, y_test_pred, digits=4)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)

    leaderboard_lines = [
        (
            f"- {item['model_name']}: "
            f"val PR-AUC={item['validation_pr_auc']:.6f}, "
            f"val F1={item['validation_f1']:.6f}, "
            f"val Recall={item['validation_recall']:.6f}, "
            f"best threshold={item['best_threshold']:.2f}"
        )
        for item in sorted(leaderboard, key=lambda item: item["validation_pr_auc"], reverse=True)
    ]

    threshold_lines = [
        (
            f"- threshold={item['threshold']:.2f}, "
            f"precision={item['precision']:.6f}, "
            f"recall={item['recall']:.6f}, "
            f"f1={item['f1']:.6f}"
        )
        for item in best_threshold_history
    ]

    report_text = (
        "Credit Card Fraud Detection Training Report\n"
        "==========================================\n\n"
        "Model leaderboard on validation set:\n"
        f"{chr(10).join(leaderboard_lines)}\n\n"
        f"Selected model: {best_model_name}\n"
        f"Selected threshold: {best_threshold:.2f}\n"
        f"Selected model validation precision: {best_validation_result['precision']:.6f}\n"
        f"Selected model validation recall: {best_validation_result['recall']:.6f}\n"
        f"Selected model validation F1: {best_validation_result['f1']:.6f}\n\n"
        "Threshold tuning details for the selected model:\n"
        f"{chr(10).join(threshold_lines)}\n\n"
        "Test set classification report:\n"
        f"{classification_text}\n"
        f"Test ROC-AUC: {roc_auc:.6f}\n"
        f"Test PR-AUC: {pr_auc:.6f}\n"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "threshold": best_threshold,
        "feature_columns": X_train.columns.tolist(),
        "amount_scaler": amount_scaler,
    }
    joblib.dump(model_artifact, MODEL_PATH)

    report_path = REPORTS_DIR / "training_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("Validation leaderboard:")
    for line in leaderboard_lines:
        print(line)
    print()
    print(f"Selected model: {best_model_name}")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(classification_text)
    print(f"Test ROC-AUC: {roc_auc:.6f}")
    print(f"Test PR-AUC: {pr_auc:.6f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training report saved to: {report_path}")


if __name__ == "__main__":
    main()
