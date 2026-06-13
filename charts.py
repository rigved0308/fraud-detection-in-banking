import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")


def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=y, ax=ax, palette="viridis")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["Legitimate", "Fraud"])
    fig.tight_layout()
    return fig


def plot_fraud_amount_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    fraud_df = df[df["Class"] == 1]
    sns.histplot(fraud_df["Amount"], bins=40, kde=True, ax=ax, color="crimson")
    ax.set_title("Fraud Transaction Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall(precision, recall, avg_precision):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"AP = {avg_precision:.4f}", color="green")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance, top_n=15):
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.head(top_n).sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig
