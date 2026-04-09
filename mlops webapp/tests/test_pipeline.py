"""Tests basiques pour le pipeline de preprocessing et le modèle."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import load_data, preprocess, split_and_scale

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Loan_Data.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def test_load_data():
    df = load_data(DATA_PATH)
    assert df.shape[0] > 0
    assert "default" in df.columns
    assert "customer_id" in df.columns


def test_preprocess():
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    assert "customer_id" not in X.columns
    assert "default" not in X.columns
    assert "debt_to_income" in X.columns
    assert "loan_to_income" in X.columns
    assert "debt_per_credit_line" in X.columns
    assert len(X) == len(y)


def test_split_and_scale():
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    # Scaled data should have roughly zero mean
    assert abs(X_train.mean().mean()) < 0.1


def test_model_exists():
    assert os.path.exists(os.path.join(MODELS_DIR, "best_model.joblib"))
    assert os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib"))
    assert os.path.exists(os.path.join(MODELS_DIR, "feature_names.json"))


def test_model_prediction():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    # Créer un sample d'entrée
    sample = pd.DataFrame([{
        "credit_lines_outstanding": 1,
        "loan_amt_outstanding": 4000.0,
        "total_debt_outstanding": 8000.0,
        "income": 70000.0,
        "years_employed": 5,
        "fico_score": 640,
        "debt_to_income": 8000.0 / 70001.0,
        "loan_to_income": 4000.0 / 70001.0,
        "debt_per_credit_line": 8000.0 / 2.0,
    }])

    X_scaled = scaler.transform(sample)
    pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)

    assert pred[0] in [0, 1]
    assert 0 <= proba[0][1] <= 1
