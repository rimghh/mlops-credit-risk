"""
Pré-traitement des données de prêts bancaires.
- Chargement du CSV
- Suppression de customer_id
- Feature engineering
- Split train/test
- Standardisation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame):
    # Supprimer l'identifiant client (pas utile pour la prédiction)
    df = df.drop(columns=["customer_id"])

    # Feature engineering
    df["debt_to_income"] = df["total_debt_outstanding"] / (df["income"] + 1)
    df["loan_to_income"] = df["loan_amt_outstanding"] / (df["income"] + 1)
    df["debt_per_credit_line"] = df["total_debt_outstanding"] / (df["credit_lines_outstanding"] + 1)

    # Séparation features / target
    X = df.drop(columns=["default"])
    y = df["default"]

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Sauvegarder le scaler pour le déploiement
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_data("data/Loan_Data.csv")
    print(f"Dataset: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Taux de défaut: {df['default'].mean():.2%}")

    X, y = preprocess(df)
    print(f"Features: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
