import pandas as pd
import streamlit as st

from preprocess import load_data, preprocess
from model import train_model, evaluate_model, get_feature_importance
from charts import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_feature_importance,
    plot_fraud_amount_distribution,
)
from report import build_pdf

st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🔍")

with st.sidebar:
    st.title("🔍 Fraud Detection")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help="Use the Kaggle Credit Card Fraud dataset",
    )
    model_type = st.selectbox(
        "Select Model",
        ["XGBoost", "Random Forest", "Logistic Regression"],
    )
    run = st.button("🚀 Train & Evaluate", use_container_width=True)
    st.markdown("---")
    st.caption("Dataset: Kaggle Credit Card Fraud")

st.title("🔍 Fraud Detection in Banking")
st.markdown("Upload transaction data, train a model, and detect fraudulent transactions.")
st.markdown("---")

if not uploaded_file:
    st.info("Upload a CSV file and click **Train & Evaluate** to begin.")
    st.markdown(
        "**Download dataset:** [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)"
    )
    st.stop()

if not run:
    st.info("Click **Train & Evaluate** in the sidebar to start.")
    st.stop()

with st.spinner("Loading and preprocessing data..."):
    df = load_data(uploaded_file)
    X_train, X_test, y_train, y_test, X, y = preprocess(df)

st.success(
    f"Dataset loaded: **{len(df):,}** transactions | **{int(y.sum()):,}** frauds ({y.mean()*100:.2f}%)"
)

with st.expander("📋 Step 1: Data Overview", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases", f"{int(y.sum()):,}")
    c3.metric("Legitimate Cases", f"{int((y == 0).sum()):,}")
    c4.metric("Fraud Rate", f"{y.mean()*100:.3f}%")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("### 📊 Step 2: Data Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_dist = plot_class_distribution(y)
    st.pyplot(fig_dist)

with col2:
    if "Amount" in df.columns:
        fig_amt = plot_fraud_amount_distribution(df)
        st.pyplot(fig_amt)

with st.spinner(f"Training {model_type} model..."):
    model = train_model(X_train, y_train, model_type)
    results = evaluate_model(model, X_test, y_test)

st.markdown("### 🤖 Step 3: Model Evaluation")
r = results["report"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC AUC", f"{results['roc_auc']:.4f}")
c2.metric("Fraud Precision", f"{r['1']['precision']:.4f}")
c3.metric("Fraud Recall", f"{r['1']['recall']:.4f}")
c4.metric("Fraud F1 Score", f"{r['1']['f1-score']:.4f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Confusion Matrix", "ROC Curve", "Precision-Recall", "Feature Importance"]
)

with tab1:
    fig_cm = plot_confusion_matrix(results["cm"])
    st.pyplot(fig_cm)
    tn, fp, fn, tp = results["cm"].ravel()
    st.caption(f"True Positives: {tp} | False Positives: {fp} | True Negatives: {tn} | False Negatives: {fn}")

with tab2:
    fig_roc = plot_roc_curve(results["fpr"], results["tpr"], results["roc_auc"])
    st.pyplot(fig_roc)

with tab3:
    fig_pr = plot_precision_recall(
        results["precision"], results["recall"], results["avg_precision"]
    )
    st.pyplot(fig_pr)

fig_imp = None
with tab4:
    importance = get_feature_importance(model, X.columns, model_type)
    if importance is not None:
        fig_imp = plot_feature_importance(importance)
        st.pyplot(fig_imp)
    else:
        st.info("Feature importance not available for this model.")

st.markdown("### 💾 Step 4: Export")

results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = results["y_pred"]
results_df["Fraud_Probability"] = results["y_prob"]

csv_bytes = results_df.to_csv(index=False).encode("utf-8")

summary = {
    "Model": model_type,
    "Total Transactions": len(df),
    "Fraud Cases": int(y.sum()),
    "Fraud Rate": f"{y.mean()*100:.3f}%",
    "ROC AUC": round(results["roc_auc"], 4),
    "Fraud Precision": round(r["1"]["precision"], 4),
    "Fraud Recall": round(r["1"]["recall"], 4),
    "Fraud F1": round(r["1"]["f1-score"], 4),
}

pdf_figs = {
    "class_dist": fig_dist,
    "confusion": fig_cm,
    "roc": fig_roc,
    "pr": fig_pr,
    "importance": fig_imp,
}

pdf_bytes = build_pdf(
    model_type=model_type,
    summary=summary,
    report=r,
    figs=pdf_figs,
)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "📥 Download Predictions CSV",
        csv_bytes,
        "fraud_predictions.csv",
        "text/csv",
        use_container_width=True,
    )

with d2:
    st.download_button(
        "📄 Download PDF Report",
        pdf_bytes,
        "fraud_report.pdf",
        "application/pdf",
        use_container_width=True,
    )

st.caption("Fraud Detection | Python · XGBoost · scikit-learn · Streamlit")
