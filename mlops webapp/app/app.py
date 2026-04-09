"""
Application Streamlit — Prédiction de Défaut de Prêt Personnel
Modes : saisie manuelle OU upload CSV pour prédiction batch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

REQUIRED_COLUMNS = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODELS_DIR, "model_info.json")) as f:
        model_info = json.load(f)
    return model, scaler, feature_names, model_info


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering identique au preprocessing."""
    df = df.copy()
    df["debt_to_income"] = df["total_debt_outstanding"] / (df["income"] + 1)
    df["loan_to_income"] = df["loan_amt_outstanding"] / (df["income"] + 1)
    df["debt_per_credit_line"] = df["total_debt_outstanding"] / (df["credit_lines_outstanding"] + 1)
    return df


def predict_single(model, scaler, feature_names):
    """Mode saisie manuelle — un seul client."""
    st.sidebar.header("Caractéristiques du Client")

    credit_lines = st.sidebar.slider("Lignes de crédit en cours", 0, 10, 1)
    loan_amt = st.sidebar.number_input("Montant du prêt en cours ($)", 0.0, 50000.0, 4000.0, step=500.0)
    total_debt = st.sidebar.number_input("Dette totale en cours ($)", 0.0, 100000.0, 8000.0, step=500.0)
    income = st.sidebar.number_input("Revenu annuel ($)", 1000.0, 300000.0, 70000.0, step=1000.0)
    years_employed = st.sidebar.slider("Années d'emploi", 0, 40, 5)
    fico_score = st.sidebar.slider("Score FICO", 300, 850, 640)

    input_data = pd.DataFrame([{
        "credit_lines_outstanding": credit_lines,
        "loan_amt_outstanding": loan_amt,
        "total_debt_outstanding": total_debt,
        "income": income,
        "years_employed": years_employed,
        "fico_score": fico_score,
    }])
    input_data = add_features(input_data)[feature_names]

    if st.sidebar.button("🔍 Prédire", use_container_width=True):
        X_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_names)
        proba = model.predict_proba(X_scaled)[0][1]
        prediction = model.predict(X_scaled)[0]

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilité de défaut", f"{proba:.1%}")
        with col2:
            if prediction == 1:
                st.error("⚠️ RISQUE ÉLEVÉ — Défaut probable")
            else:
                st.success("✅ RISQUE FAIBLE — Pas de défaut prévu")

        st.markdown("### Niveau de risque")
        if proba < 0.3:
            st.progress(proba, text="Risque faible")
        elif proba < 0.6:
            st.progress(proba, text="Risque modéré")
        else:
            st.progress(min(proba, 1.0), text="Risque élevé")

        st.markdown("### Résumé du profil client")
        display_data = pd.DataFrame([{
            "Lignes de crédit": credit_lines,
            "Montant prêt ($)": f"{loan_amt:,.2f}",
            "Dette totale ($)": f"{total_debt:,.2f}",
            "Revenu ($)": f"{income:,.2f}",
            "Années emploi": years_employed,
            "Score FICO": fico_score,
        }])
        st.dataframe(display_data.T.rename(columns={0: "Valeur"}), use_container_width=True)


def predict_batch(model, scaler, feature_names):
    """Mode upload CSV — prédiction batch sur tout un fichier."""
    st.markdown("### 📂 Upload de fichier CSV")
    st.info(
        f"Le fichier doit contenir les colonnes : `{'`, `'.join(REQUIRED_COLUMNS)}`\n\n"
        "Les colonnes optionnelles (`customer_id`, `default`) seront ignorées pour la prédiction."
    )

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"**{len(df)} lignes chargées**")

        # Validation des colonnes
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"❌ Colonnes manquantes : `{'`, `'.join(missing)}`")
            return

        st.markdown("#### Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("🚀 Lancer les prédictions", use_container_width=True):
            # Feature engineering
            df_feat = add_features(df)
            X = df_feat[feature_names]
            X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names, index=df.index)

            # Prédictions
            df["probabilite_defaut"] = model.predict_proba(X_scaled)[:, 1]
            df["prediction"] = model.predict(X_scaled)
            df["risque"] = df["probabilite_defaut"].apply(
                lambda p: "🔴 Élevé" if p >= 0.6 else ("🟡 Modéré" if p >= 0.3 else "🟢 Faible")
            )

            # Résumé
            st.markdown("---")
            st.markdown("#### 📊 Résultats")

            col1, col2, col3 = st.columns(3)
            n_default = (df["prediction"] == 1).sum()
            n_safe = (df["prediction"] == 0).sum()
            avg_proba = df["probabilite_defaut"].mean()

            with col1:
                st.metric("Clients à risque", f"{n_default} / {len(df)}")
            with col2:
                st.metric("Clients sains", f"{n_safe} / {len(df)}")
            with col3:
                st.metric("Probabilité moyenne", f"{avg_proba:.1%}")

            # Tableau des résultats
            st.markdown("#### Détail des prédictions")
            display_cols = [c for c in df.columns if c != "prediction"]
            st.dataframe(
                df[display_cols].sort_values("probabilite_defaut", ascending=False),
                use_container_width=True,
                height=400,
            )

            # Téléchargement
            csv_result = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv_result,
                file_name="predictions_defaut.csv",
                mime="text/csv",
                use_container_width=True,
            )


def main():
    st.set_page_config(page_title="Prédiction Défaut de Prêt", page_icon="🏦", layout="wide")

    st.title("🏦 Prédiction de Défaut de Prêt Personnel")
    st.markdown(
        "Cette application prédit la **probabilité de défaut de paiement** d'un client "
        "en fonction de ses caractéristiques financières."
    )

    model, scaler, feature_names, model_info = load_artifacts()
    st.caption(f"Modèle actif : **{model_info['model_name']}** — F1 = {model_info['f1_score']:.4f}")

    # Onglets : saisie manuelle vs upload CSV
    tab1, tab2 = st.tabs(["👤 Client unique", "📂 Upload CSV (batch)"])

    with tab1:
        predict_single(model, scaler, feature_names)

    with tab2:
        predict_batch(model, scaler, feature_names)

    st.markdown("---")
    st.caption("Projet MLOps — Prédiction de défaut de prêt | Banque de détail")


if __name__ == "__main__":
    main()
