import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


def train_model(X_train, y_train, model_type="XGBoost"):
    if model_type == "XGBoost":
        if XGBOOST_AVAILABLE:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
    elif model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
        "cm": cm,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "avg_precision": avg_precision,
    }


def get_feature_importance(model, feature_names, model_type):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    if model_type == "Logistic Regression" and hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    return None
