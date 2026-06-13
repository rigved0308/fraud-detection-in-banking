import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def preprocess(df):
    df = df.copy()

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()

    if "Amount" in X.columns:
        X["Amount"] = scaler.fit_transform(X[["Amount"]])

    if "Time" in X.columns:
        X["Time"] = scaler.fit_transform(X[["Time"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, X, y
