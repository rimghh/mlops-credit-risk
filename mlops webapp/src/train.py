"""
Entraînement de 3 modèles de classification avec tracking MLflow.
- Régression Logistique
- Arbre de Décision
- Random Forest

Chaque modèle = 1 experiment MLflow.
Chaque itération (hyperparamètres) = 1 run.
Le meilleur modèle est sauvegardé dans models/.
"""

import os
import sys
import json
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Ajout du chemin src au PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_data, preprocess, split_and_scale

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "Loan_Data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRACKING_URI = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train_and_log(model, experiment_name, params, X_train, X_test, y_train, y_test, run_name):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        report = classification_report(y_test, model.predict(X_test))
        mlflow.log_text(report, "classification_report.txt")

        print(f"  [{run_name}] AUC={metrics['roc_auc']:.4f}  F1={metrics['f1_score']:.4f}")

    return model, metrics


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Chargement et pré-traitement ──────────────────────────────
    print("=" * 60)
    print("  Loan Default Prediction — Model Training")
    print("=" * 60)

    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Sauvegarder scaler + feature names pour le déploiement
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(X_train.columns.tolist(), f)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Taux de défaut: {y.mean():.2%}\n")

    best_model, best_score, best_name = None, -1, ""

    # ── 2. Régression Logistique ─────────────────────────────────────
    print("[Experiment] Logistic Regression")
    for i, cfg in enumerate([
        {"C": 0.01, "max_iter": 1000, "solver": "lbfgs"},
        {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
        {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        {"C": 10.0, "max_iter": 1000, "solver": "lbfgs"},
    ], 1):
        m, met = train_and_log(
            LogisticRegression(**cfg, random_state=42),
            "Logistic_Regression", cfg,
            X_train, X_test, y_train, y_test,
            f"lr_run_{i}_C{cfg['C']}",
        )
        if met["f1_score"] > best_score:
            best_score, best_model, best_name = met["f1_score"], m, f"LR_C{cfg['C']}"

    # ── 3. Arbre de Décision ─────────────────────────────────────────
    print("\n[Experiment] Decision Tree")
    for i, cfg in enumerate([
        {"max_depth": 3, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 5},
        {"max_depth": 10, "min_samples_split": 10},
        {"max_depth": None, "min_samples_split": 2},
    ], 1):
        m, met = train_and_log(
            DecisionTreeClassifier(**cfg, random_state=42),
            "Decision_Tree", cfg,
            X_train, X_test, y_train, y_test,
            f"dt_run_{i}_depth{cfg['max_depth']}",
        )
        if met["f1_score"] > best_score:
            best_score, best_model, best_name = met["f1_score"], m, f"DT_depth{cfg['max_depth']}"

    # ── 4. Random Forest ─────────────────────────────────────────────
    print("\n[Experiment] Random Forest")
    for i, cfg in enumerate([
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 5},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    ], 1):
        m, met = train_and_log(
            RandomForestClassifier(**cfg, random_state=42, n_jobs=-1),
            "Random_Forest", cfg,
            X_train, X_test, y_train, y_test,
            f"rf_run_{i}_n{cfg['n_estimators']}_depth{cfg['max_depth']}",
        )
        if met["f1_score"] > best_score:
            best_score, best_model, best_name = met["f1_score"], m, f"RF_n{cfg['n_estimators']}"

    # ── 5. Sauvegarde du meilleur modèle ─────────────────────────────
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    info = {"model_name": best_name, "f1_score": best_score,
            "features": X_train.columns.tolist()}
    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  MEILLEUR MODÈLE: {best_name}  (F1 = {best_score:.4f})")
    print(f"{'=' * 60}")
    print(f"  Artifacts sauvegardés dans {MODELS_DIR}/")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
