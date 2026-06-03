import joblib
import pandas as pd

from utils import MODEL_PATH


def prepare_sample(sample_dict: dict, feature_columns: list[str], amount_scaler):
    sample_df = pd.DataFrame([sample_dict])
    sample_df = sample_df.reindex(columns=feature_columns, fill_value=0.0)

    if "Amount" in sample_df.columns:
        sample_df["Amount"] = amount_scaler.transform(sample_df[["Amount"]]).ravel()

    return sample_df


def predict_sample(sample_dict: dict):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]
    amount_scaler = artifact.get("amount_scaler")

    if amount_scaler is None:
        raise ValueError("Model artifact does not contain amount_scaler. Please rerun src/train.py.")

    sample_df = prepare_sample(sample_dict, feature_columns, amount_scaler)

    fraud_probability = float(model.predict_proba(sample_df)[0][1])
    prediction = int(fraud_probability >= threshold)
    return prediction, fraud_probability


if __name__ == "__main__":
    demo_sample = {"Amount": 100.0}
    prediction, probability = predict_sample(demo_sample)
    print(f"Demo prediction: {prediction}")
    print(f"Fraud probability: {probability:.6f}")
