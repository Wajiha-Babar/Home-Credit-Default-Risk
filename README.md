# Credit Risk Intelligence Platform

Premium end-to-end machine learning project for the Home Credit Default Risk dataset. It predicts loan default, compares multiple models, explains predictions with SHAP, and ships a dark enterprise Streamlit dashboard for risk analysts and executives.

## Project Structure

```text
.
|-- data/
|   `-- application_train.csv
|-- outputs/
|   |-- figures/
|   |-- models/
|   `-- reports/
|-- Home_Credit_Default_Risk_Analysis.ipynb
|-- streamlit_app.py
|-- train_models.py
|-- utils.py
|-- model_utils.py
|-- dashboard_utils.py
|-- requirements.txt
`-- README.md
```

## Deliverables

- Professional Jupyter notebook
- Modular production-style Python utilities
- Exploratory data analysis with saved figures
- Data cleaning and feature engineering
- Logistic Regression + SMOTE, Random Forest, and XGBoost comparison
- Metrics report with accuracy, precision, recall, F1, and ROC AUC
- Confusion matrices and ROC curves
- SHAP global and local explanations
- Premium Streamlit dashboard for executive risk analytics

## Key Fixes

- Prevents `int64` assignment errors by converting age and employment years to integer day values.
- Saves and reuses training feature order.
- Adds missing prediction columns automatically.
- Removes extra prediction columns automatically.
- Aligns dashboard input to model training columns.
- Handles SHAP output shape differences for binary classifiers.
- Uses explicit artifact loading errors and reusable preprocessing functions.

## Dataset

The main notebook uses `data/application_train.csv`.

Target variable:

- `TARGET = 0`: loan repaid
- `TARGET = 1`: loan defaulted

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Home_Credit_Default_Risk_Analysis.ipynb
```

Retrain models and regenerate comparison artifacts:

```bash
python train_models.py
```

Launch the premium dashboard:

```bash
streamlit run streamlit_app.py
```

## Outputs

The notebook creates:

- `outputs/figures/`: EDA, model, and SHAP plots
- `outputs/models/`: best trained model and metadata
- `outputs/reports/`: model metrics and individual explanations

## Business Value

The platform helps lenders identify high-risk applications, understand drivers of default, compare model tradeoffs, explain decisions transparently, and support risk-based credit policy.

## Deployment

### GitHub

1. Keep `data/application_train.csv` out of Git if repository size is a concern.
2. Commit source files, notebook, requirements, README, and generated reports/figures.
3. Add large model/data artifacts with Git LFS if needed.

### Streamlit Cloud

1. Push the repository to GitHub.
2. In Streamlit Cloud, create a new app from the repository.
3. Set the app file to `streamlit_app.py`.
4. Ensure `requirements.txt` includes `plotly`, `imbalanced-learn`, `xgboost`, and `shap`.
5. Upload or provision the trained artifacts under `outputs/models/`.
