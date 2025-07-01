# src/models/train_model.py

import os
import mlflow
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.feature_engineering import run_feature_engineering


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

def train_and_log_model(X_train, X_test, y_train, y_test, model, params, model_name):
    with mlflow.start_run(run_name=model_name):
        clf = GridSearchCV(model, params, cv=3, scoring='f1', n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_params(clf.best_params_)
        model_path = f"models/{model_name}.pkl"
        joblib.dump(clf.best_estimator_, model_path)
        mlflow.log_artifact(model_path)
        print(f"Logged {model_name} with metrics: {metrics}")


def main():
    # Load and process data
    X, y = run_feature_engineering("data/raw/data.csv", "models/")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Set MLflow experiment
    mlflow.set_experiment("credit-risk-model")

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_params = {"C": [0.01, 0.1, 1, 10]}
    train_and_log_model(X_train, X_test, y_train, y_test, lr_model, lr_params, "LogisticRegression")

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_params = {"n_estimators": [50, 100], "max_depth": [5, 10]}
    train_and_log_model(X_train, X_test, y_train, y_test, rf_model, rf_params, "RandomForest")


if __name__ == "__main__":
    main()
