import joblib
import pandas as pd

from utils import MODEL_PATH


def predict_sample(sample_dict: dict):
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]

    sample_df = pd.DataFrame([sample_dict])
    sample_df = sample_df.reindex(columns=feature_columns, fill_value=0.0)

    fraud_probability = float(model.predict_proba(sample_df)[0][1])
    prediction = int(fraud_probability >= threshold)
    return prediction, fraud_probability


if __name__ == "__main__":
    print("Import predict_sample from this module and pass a transaction dictionary.")
