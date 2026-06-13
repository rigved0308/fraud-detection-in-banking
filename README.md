Intern ID: CITS724


# Fraud Detection in Banking

A Streamlit web application for detecting fraudulent banking transactions using machine learning models on the Kaggle Credit Card Fraud dataset.

## Live App

[Streamlit App](https://fraud-detection-in-banking-d4t9mwez9qqdmnvjxd7le4.streamlit.app)

## GitHub Repository

[rigved0308/fraud-detection-in-banking](https://github.com/rigved0308/fraud-detection-in-banking)

## Overview

This project allows users to upload a transaction dataset in CSV format, train a machine learning model, evaluate fraud-detection performance, visualize the results, and download predictions and a PDF report.

The application is built with Streamlit and currently supports:

- XGBoost
- Random Forest
- Logistic Regression

## Features

- Upload CSV fraud transaction dataset
- Train and evaluate multiple ML models
- View dataset summary and fraud rate
- Visualize:
  - Class distribution
  - Fraud amount distribution
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Feature importance
- Download:
  - Predictions CSV
  - PDF performance report

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle:

[Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Expected Dataset Format

The uploaded CSV should contain a `Class` column where:

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

The dataset may also include columns such as:

- `Time`
- `Amount`
- PCA-transformed features such as `V1` to `V28`

## Project Structure

```bash
fraud-detection-in-banking/
│
├── app.py
├── model.py
├── preprocess.py
├── charts.py
├── report.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rigved0308/fraud-detection-in-banking.git
cd fraud-detection-in-banking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

The app usually opens at:

```bash
http://localhost:8501
```

## Requirements

Example dependencies used in this project:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
fpdf2
```

## How to Use

1. Open the app.
2. Upload the Kaggle fraud dataset CSV.
3. Select a model from the sidebar.
4. Click **Train & Evaluate**.
5. Review metrics, plots, and model output.
6. Download the predictions CSV and PDF report.

## Models Used

### XGBoost
A gradient boosting algorithm that performs well on structured tabular datasets.

### Random Forest
An ensemble model based on multiple decision trees for robust classification.

### Logistic Regression
A simple and interpretable baseline model for binary classification tasks.

## Output Metrics

The app displays key fraud-detection metrics, including:

- ROC AUC
- Fraud Precision
- Fraud Recall
- Fraud F1 Score
- Confusion Matrix
- Precision-Recall Curve

## Deployment

This project is deployed on Streamlit Cloud.

To deploy:

1. Push the code to GitHub.
2. Open Streamlit Cloud.
3. Create a new app linked to the repository.
4. Select `app.py` as the main file.
5. Deploy the app.

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- fpdf2

## Future Improvements

- Add model comparison in a single dashboard
- Add class imbalance handling with SMOTE
- Add threshold tuning for fraud probability
- Add sample dataset mode
- Add better validation for uploaded files

## License

This project is intended for educational, academic, and internship demonstration purposes.
