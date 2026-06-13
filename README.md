Intern ID: CITS724

# Fraud Detection in Banking

A Streamlit-based machine learning web app for detecting fraudulent banking transactions using the Kaggle Credit Card Fraud dataset.

## Features

- Upload a CSV dataset
- Train and evaluate fraud detection models
- Supports:
  - XGBoost
  - Random Forest
  - Logistic Regression
- Visualizes:
  - Class distribution
  - Fraud amount distribution
  - Confusion matrix
  - ROC curve
  - Precision-Recall curve
  - Feature importance
- Download:
  - Predictions as CSV
  - PDF report

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset must contain a `Class` column where:
- `0` = Legitimate transaction
- `1` = Fraudulent transaction

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
git clone https://github.com/your-username/fraud-detection-in-banking.git
cd fraud-detection-in-banking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

The app will open in your browser, usually at:

```bash
http://localhost:8501
```

## Requirements

Example `requirements.txt`:

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
2. Upload the fraud detection CSV dataset.
3. Select a model.
4. Click **Train & Evaluate**.
5. Review charts and evaluation metrics.
6. Download predictions and PDF report.

## Models Used

### 1. XGBoost
A powerful boosting algorithm often used for tabular classification tasks.

### 2. Random Forest
An ensemble learning model based on multiple decision trees.

### 3. Logistic Regression
A simple and effective baseline model for binary classification.

## Outputs

The app provides:

- Fraud detection metrics
- ROC AUC score
- Precision, Recall, and F1 score
- Confusion matrix
- Downloadable predictions CSV
- Downloadable PDF summary report

## Deployment

This project can be deployed easily on **Streamlit Cloud**.

Steps:
1. Push the project to GitHub.
2. Go to Streamlit Cloud.
3. Create a new app linked to the GitHub repository.
4. Set `app.py` as the main file.
5. Deploy.

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- XGBoost
- matplotlib
- seaborn
- fpdf2

## Future Improvements

- Add sample dataset support
- Add SMOTE or class balancing
- Add threshold tuning
- Add model comparison dashboard
- Add fraud probability filtering

## License

This project is for educational and internship/demo purposes.
