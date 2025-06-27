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
