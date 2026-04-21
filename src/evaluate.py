import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
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
    summary_path.write_text(
        (
            f"Selected model: {model_name}\n"
            f"Decision threshold: {threshold:.2f}\n"
            f"Saved confusion matrix, PR curve, and ROC curve to outputs/figures.\n"
        ),
        encoding="utf-8",
    )

    print("Evaluation figures saved to outputs/figures/")
    print(f"Selected model: {model_name}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
