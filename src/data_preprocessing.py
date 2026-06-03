import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import DATA_PATH


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")
    return df


def prepare_features(df: pd.DataFrame):
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    if "Time" in X.columns:
        X = X.drop("Time", axis=1)

    return X, y


def split_scale_and_validate(
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    return_scaler: bool = False,
):
    df = load_dataset()
    X, y = prepare_features(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    if "Amount" in X_train.columns:
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]]).ravel()
        X_val["Amount"] = scaler.transform(X_val[["Amount"]]).ravel()
        X_test["Amount"] = scaler.transform(X_test[["Amount"]]).ravel()

    if return_scaler:
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler

    return X_train, X_val, X_test, y_train, y_val, y_test
