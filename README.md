# bati-credit-risk-model
Credit Risk Scoring Model for Bati Bank’s Buy-Now-Pay-Later (BNPL) service using alternative eCommerce data. This project includes proxy default definition, feature engineering, risk probability modeling, credit scoring, and loan recommendation aligned with Basel II regulatory standards.

📘## Credit Scoring Business Understanding

### 1. Why Interpretable, Documented Models Are Essential Under Basel II
The Basel II Accord emphasizes accurate and transparent credit risk measurement. Under its Internal Ratings-Based (IRB) approach, banks must ensure credit risk models are well-documented, rigorously validated, and interpretable. This helps satisfy regulatory requirements and builds trust among stakeholders by making credit decisions explainable and auditable.

### 2. Importance and Risk of Using a Proxy Variable
Because a direct “default” label is unavailable, a proxy default variable is created using fraud indicators and behavioral signals. This proxy enables supervised learning but carries risks such as label bias, false positives rejecting creditworthy customers, and false negatives increasing default risk. Continuous refinement and monitoring of this proxy are critical.

### 3. Model Trade-Offs: Simplicity vs. Performance
Simple models like Logistic Regression with Weight of Evidence (WoE) offer transparency and ease of validation but may lack complex pattern detection. Complex models such as Gradient Boosting provide higher predictive accuracy but require explainability tools (e.g., SHAP) and face stricter regulatory scrutiny. Balancing interpretability and performance is essential in regulated financial environments.

| Aspect                    | Simple Models (e.g., LR + WoE)       | Complex Models (e.g., XGBoost) |
| ------------------------- | ------------------------------------ | ------------------------------ |
| **Interpretability**      | High – easy to explain to regulators | Low – requires SHAP/LIME       |
| **Transparency**          | Strong – variable impact is clear    | Weaker – complex interactions  |
| **Regulatory Compliance** | Easier to validate                   | More scrutiny required         |
| **Predictive Power**      | May miss non-linear patterns         | Often higher accuracy          |
| **Deployment**            | Lightweight and fast                 | Heavier and harder to audit    |

In this project, we may start with a simple baseline model for compliance and stakeholder trust, then evaluate more complex models using explainable AI techniques for improvement, all while preserving auditability and fairness.

credit-risk-model/
├── .github/workflows/ci.yml           # GitHub Actions for CI/CD – Good DevOps practice
├── data/                              # Store datasets (✅ should be in .gitignore)
│   ├── raw/                           # Raw input data
│   └── processed/                     # Cleaned and transformed data
├── notebooks/
│   └── 1.0-eda.ipynb                  # Jupyter notebook for initial EDA
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Feature engineering, data cleaning, etc.
│   ├── train.py                      # Model training logic
│   ├── predict.py                    # Inference script
│   └── api/
│       ├── main.py                   # FastAPI app entry point
│       └── pydantic_models.py        # Request/response models
├── tests/
│   └── test_data_processing.py       # Unit tests (you can expand this)
├── Dockerfile                        # For containerizing the app
├── docker-compose.yml                # For multi-service orchestration
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Should include `/data/`, `.env`, etc.
└── README.md                         # Project documentation
## 📌 Project Overview


Credit risk modeling is essential for financial institutions to assess the likelihood that a borrower will default on a loan. This project builds an end-to-end pipeline for credit risk prediction, including:

- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Model training and evaluation
- Inference via RESTful API
- Dockerized deployment and CI/CD integration

### 📥 Clone the Repository


git clone https://github.com/temeawoke/bati-credit-risk-model.git
cd credit-risk-model

🛠️ Install Dependencies
pip install -r requirements.txt



Task 3 and 4

# Credit Risk Probability Model using Alternative Data

This project builds a machine learning pipeline to predict high-risk (proxy default) customers using transaction data without explicit credit risk labels. It simulates how alternative financial behavior can be used to assess creditworthiness.

---

## 📂 Project Structure

# Generate a tree structure as a markdown-formatted string
project_structure 

credit-risk-model/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                      # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Script for feature engineering
│   ├── train.py               # Script for model training
│   ├── predict.py             # Script for inference
│   └── api/
│       ├── main.py            # FastAPI application
│       └── pydantic_models.py # Pydantic models for API
├── tests/
│   └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md


# Append this structure to the README.md
readme_path = "/mnt/data/README.md"
with open(readme_path, "a", encoding="utf-8") as f:
    f.write("\n## 📁 Project Structure\n")
    f.write(project_structure)

readme_path


## 🎯 Objective

* Build a pipeline to create a **proxy credit risk target** using behavioral clustering.
* Engineer meaningful features from raw data.
* Train and compare ML models.
* Log experiments using **MLflow**.

---

## 🔍 Data Summary

The dataset contains mobile money transaction logs with fields like:

* `TransactionStartTime`
* `CustomerId`
* `Amount`
* `ProductId`, `ChannelId`, etc.

There is **no direct `credit_risk` or `default` column**, so we create one via clustering.

---

## ⚙️ Pipeline Overview

### Task 3: Feature Engineering

* Extract datetime features (hour, day, month, year)
* Aggregate stats per `CustomerId`: total, mean, count, std of amount
* One-hot encode categorical variables
* Standardize numeric variables

### Task 4: Proxy Target Engineering

* Calculate RFM (Recency, Frequency, Monetary) per customer
* Cluster customers with KMeans into 3 segments
* Label lowest engagement group as `is_high_risk = 1`

### Task 5: Model Training

* Train & compare:

  * Logistic Regression
  * Random Forest
* Hyperparameter tuning via `GridSearchCV`
* Log metrics + artifacts with **MLflow**

---

## 📈 Run the Project

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Feature Engineering & Target Label Creation

```python
from src.data.feature_engineering import run_feature_engineering
X, y = run_feature_engineering("data/raw/your_file.csv", "models/")
```

### 3. Train Models & Track Experiments


python src/models/train_model.py


### 4. View MLflow UI


mlflow ui --port 5000


Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)



## ✅ Requirements


mlflow
scikit-learn
pandas
numpy
joblib
pytest


## 🧪 Testing


pytest tests/




## 📌 Notes

* This is a **proxy modeling** setup — not for real-world credit scoring.
* Ethical use of customer segmentation is critical.
* Future work could include time-series modeling or model explainability (e.g. SHAP).


## 👤 Author

Temesgen Awoke — Built as part of 10 Academy Week 5 Challenge

