# Credit Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Credit%20Risk-111827?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-D7B56D?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Style-10B981?style=for-the-badge)

A premium end-to-end machine learning platform for the **Home Credit Default Risk** dataset.  
This project predicts loan default risk, compares multiple machine learning models, explains predictions using SHAP, and delivers a luxury dark-mode Streamlit dashboard for credit analysts, risk teams, and executive decision-makers.

---

## Executive Overview

The **Credit Risk Intelligence Platform** is designed as a complete analytics and machine learning solution for credit default prediction. It combines exploratory data analysis, data preprocessing, feature engineering, model training, model evaluation, explainable AI, and an interactive business dashboard.

The platform helps lenders understand customer risk, identify high-risk loan applications, compare model performance, and generate transparent prediction explanations that support responsible credit decision-making.

---

## Author

**Wajiha Babar**  
Data Analyst | Data Scientist | Machine Learning Enthusiast

---

## Key Highlights

- End-to-end credit default risk prediction pipeline
- Professional exploratory data analysis notebook
- Automated data cleaning and feature engineering
- Logistic Regression with SMOTE
- Random Forest model
- XGBoost model
- Model comparison using Accuracy, Precision, Recall, F1 Score, and ROC AUC
- Confusion matrix and ROC curve visualizations
- Global and local SHAP explainability
- Premium executive Streamlit dashboard
- Modular Python architecture for maintainability
- Reusable preprocessing and prediction utilities
- Safer feature alignment between training and dashboard prediction
- Business-focused risk insights and recommendations

---

## Business Problem

Financial institutions need to evaluate whether a loan applicant is likely to repay or default. Traditional rule-based systems can miss complex risk patterns hidden inside customer demographics, income, credit exposure, employment history, and external credit scores.

This project solves that problem by building a machine learning system that predicts default probability and explains the main drivers behind each prediction.

---

## Dataset

This project uses the **Home Credit Default Risk** dataset.

Target variable:

| Target | Meaning |
|---|---|
| `0` | Customer repaid the loan |
| `1` | Customer defaulted on the loan |

Main training file:

```text
data/application_train.csv
```

Due to file-size limitations, large Kaggle dataset files should not be committed to GitHub. Download the dataset separately and place the required CSV files inside the `data/` directory.

Recommended dataset location:

```text
data/application_train.csv
data/application_test.csv
data/bureau.csv
data/bureau_balance.csv
data/previous_application.csv
data/POS_CASH_balance.csv
data/installments_payments.csv
data/credit_card_balance.csv
```

---

## Project Structure

```text
.
├── data/
│   └── HomeCredit_columns_description.csv
│
├── outputs/
│   ├── figures/
│   │   ├── age_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance_top_20.png
│   │   ├── roc_curves.png
│   │   ├── shap_summary_plot.png
│   │   └── shap_waterfall_plot.png
│   │
│   ├── models/
│   │   └── model_metadata.json
│   │
│   └── reports/
│       ├── metrics_report.json
│       ├── model_metrics.csv
│       └── individual_prediction_explanations.csv
│
├── Home_Credit_Default_Risk_Analysis.ipynb
├── streamlit_app.py
├── train_models.py
├── utils.py
├── model_utils.py
├── dashboard_utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Core Deliverables

### 1. Jupyter Notebook

The notebook provides a complete analytical workflow:

- Dataset loading
- Exploratory data analysis
- Missing value inspection
- Target distribution analysis
- Numerical and categorical feature analysis
- Feature engineering
- Model training
- Model comparison
- SHAP explainability
- Saved charts and reports

Notebook file:

```text
Home_Credit_Default_Risk_Analysis.ipynb
```

---

### 2. Model Training Pipeline

The training script builds and compares multiple machine learning models.

Script:

```text
train_models.py
```

Models included:

| Model | Purpose |
|---|---|
| Logistic Regression + SMOTE | Interpretable baseline model with class imbalance handling |
| Random Forest | Tree-based ensemble for non-linear patterns |
| XGBoost | High-performance gradient boosting model |

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- ROC Curve

---

### 3. Premium Streamlit Dashboard

The Streamlit application provides an executive-level interface for risk analytics.

App file:

```text
streamlit_app.py
```

Dashboard modules:

| Section | Description |
|---|---|
| Executive Dashboard | Portfolio KPIs, default rate, model strength, and risk overview |
| Customer Prediction | Single-customer default probability and risk category |
| Risk Analysis | Portfolio segmentation, exposure distribution, and risk pattern analysis |
| Explainable AI | SHAP summary, waterfall, and local explanation |
| Model Performance | Metrics leaderboard, confusion matrix, ROC curve, and feature importance |
| Business Insights | Executive recommendations for risk policy and credit strategy |

---

## Technical Features

### Data Processing

The project includes reusable functions for:

- Loading raw data
- Cleaning missing values
- Handling special values such as `DAYS_EMPLOYED = 365243`
- Creating engineered features
- Aligning model input columns
- Preprocessing dashboard form inputs

### Feature Engineering

Example engineered features:

- Credit-to-income ratio
- Annuity-to-income ratio
- Employment duration
- Customer age
- Affordability indicators
- Risk burden indicators

### Model Reliability Fixes

The project includes important safeguards:

- Converts age and employment years into correct integer day values
- Prevents `int64` assignment issues
- Stores and reuses training feature order
- Automatically adds missing prediction columns
- Automatically removes extra prediction columns
- Aligns Streamlit dashboard inputs with trained model features
- Handles SHAP output shape differences for binary classifiers
- Provides explicit artifact loading errors
- Uses reusable preprocessing functions
- Improves dashboard stability with safer missing-column handling

---

## Explainable AI

The project uses SHAP to make model predictions more transparent.

SHAP outputs include:

- Global feature importance
- SHAP summary plot
- SHAP bar plot
- Local prediction explanation
- SHAP waterfall plot
- Risk-increasing and risk-reducing drivers

This allows analysts to understand not only the prediction result, but also why the model produced that result.

---

## Business Value

The platform supports credit risk teams by helping them:

- Identify high-risk loan applications
- Understand customer-level default drivers
- Compare model tradeoffs
- Support risk-based credit policies
- Improve underwriting decisions
- Reduce blind reliance on black-box predictions
- Explain model decisions transparently
- Monitor portfolio-level risk trends

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Wajiha-Babar/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
```

### 2. Create a virtual environment

For Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Run the Jupyter notebook

```bash
jupyter notebook Home_Credit_Default_Risk_Analysis.ipynb
```

### Retrain models

```bash
python train_models.py
```

### Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

After running the command, open the local Streamlit URL in your browser:

```text
http://localhost:8501
```

---

## Outputs

The project generates artifacts in the `outputs/` directory.

### Figures

```text
outputs/figures/
```

Includes:

- Target distribution
- Age distribution
- Income distribution
- Credit amount distribution
- Correlation heatmap
- Confusion matrices
- ROC curves
- Feature importance plots
- SHAP plots

### Models

```text
outputs/models/
```

Includes:

- Best trained model
- Model metadata
- Training feature list

### Reports

```text
outputs/reports/
```

Includes:

- Model metrics
- Metrics report
- Individual prediction explanations

---

## Dashboard Preview

The dashboard is designed with a premium executive interface and includes:

- Dark luxury theme
- Executive KPI cards
- Portfolio risk visualizations
- Customer default probability gauge
- Risk segmentation
- SHAP explanation tables
- Model validation views
- Business recommendation panel

---

## Deployment Notes

### GitHub

Large raw dataset files should be excluded from version control.

Recommended `.gitignore` rules:

```gitignore
.venv/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml

data/*.csv
!data/HomeCredit_columns_description.csv

outputs/models/*.joblib
outputs/models/*.pkl
outputs/models/*.pickle
```

### Streamlit Cloud

To deploy on Streamlit Cloud:

1. Push the repository to GitHub.
2. Open Streamlit Cloud.
3. Create a new app from this repository.
4. Select `streamlit_app.py` as the main app file.
5. Confirm that `requirements.txt` contains all required dependencies.
6. Provide trained model artifacts under `outputs/models/`.
7. Make sure required dataset files are available through a supported storage method.

---

## Requirements

Main Python libraries used:

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- streamlit
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model Evaluation Summary

The platform compares models using multiple performance metrics rather than relying on a single score.

| Metric | Meaning |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | How many predicted defaults were actually defaults |
| Recall | How many actual defaults were successfully detected |
| F1 Score | Balance between precision and recall |
| ROC AUC | Model's ability to separate default and non-default cases |

For credit risk, ROC AUC and recall are especially important because missing risky applicants can create financial loss.

---

## Risk Interpretation

The dashboard classifies applicants into risk levels:

| Risk Level | Interpretation |
|---|---|
| Low Risk | Lower estimated probability of default |
| Medium Risk | Requires careful review or risk-based pricing |
| High Risk | Requires enhanced affordability checks and manual underwriting |

The model output should be used as a decision-support tool, not as the only approval or rejection mechanism.

---

## Responsible AI Notice

This project is built for educational and analytical purposes.  
Credit decisions should always include human oversight, compliance review, fairness checks, and institution-specific policy rules.

The machine learning model should not be used as the sole basis for real-world lending decisions without proper validation, governance, and regulatory compliance.

---

## Future Enhancements

Planned improvements may include:

- Full pipeline automation
- MLflow experiment tracking
- Advanced hyperparameter tuning
- Additional feature engineering from bureau and previous application files
- Drift monitoring
- Fairness analysis
- Streamlit Cloud deployment
- Docker support
- API endpoint for prediction service
- Interactive applicant search
- Batch prediction upload feature

---

## Repository

GitHub Repository:

```text
https://github.com/Wajiha-Babar/Home-Credit-Default-Risk
```

---

## License

This project is intended for educational, portfolio, and analytical demonstration purposes.

---

## Acknowledgement

Dataset source: Home Credit Default Risk dataset.

This project demonstrates how machine learning, explainable AI, and executive dashboarding can be combined to support modern credit risk analytics.
