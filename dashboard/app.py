import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("🔮 Plateforme de Rétention Client")

# ✅ Chemins absolus
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_churn.csv")
df = pd.read_csv(DATA_PATH)

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(" Modèle de prédiction", [
    "random_forest", "logistic", "xgboost", "mlp"
], format_func=lambda x: {
    "random_forest": " Random Forest",
    "logistic":      " Régression Logistique",
    "xgboost":       " Gradient Boosting",
    "mlp":           " Réseau de Neurones (MLP)"
}[x])

st.sidebar.markdown("---")
st.sidebar.header(" Profil Client")

gender = st.sidebar.selectbox("Genre", [0, 1], format_func=lambda x: "Homme" if x == 0 else "Femme")
age    = st.sidebar.slider("Âge", 18, 80, 35)
tenure_months      = st.sidebar.slider("Ancienneté (mois)", 0, 72, 12)
monthly_fee        = st.sidebar.number_input("Frais mensuels (€)", 0, 500, 50)
payment_failures   = st.sidebar.slider("Échecs de paiement", 0, 10, 0)
support_tickets    = st.sidebar.slider("Tickets support", 0, 20, 1)
nps_score          = st.sidebar.slider("NPS Score", -100, 100, 50)
csat_score         = st.sidebar.slider("CSAT Score", 0, 10, 7)
monthly_logins     = st.sidebar.slider("Connexions/mois", 0, 100, 10)
last_login_days_ago = st.sidebar.slider("Dernière connexion (jours)", 0, 365, 7)
contract_type      = st.sidebar.selectbox("Type de contrat", [0, 1, 2],
                        format_func=lambda x: ["Mensuel", "Annuel", "Deux ans"][x])

# ── Prédiction ─────────────────────────────────────────────
if st.sidebar.button("🔍 Prédire le Churn"):
    payload = {
        "model_name":            model_choice,
        "gender":                gender,
        "age":                   age,
        "country":               0,
        "city":                  0,
        "customer_segment":      0,
        "tenure_months":         tenure_months,
        "signup_channel":        0,
        "contract_type":         contract_type,
        "monthly_logins":        monthly_logins,
        "weekly_active_days":    3,
        "avg_session_time":      30,
        "features_used":         10,
        "usage_growth_rate":     0,
        "last_login_days_ago":   last_login_days_ago,
        "monthly_fee":           monthly_fee,
        "total_revenue":         monthly_fee * tenure_months,
        "payment_method":        0,
        "payment_failures":      payment_failures,
        "discount_applied":      0,
        "price_increase_last_3m": 0,
        "support_tickets":       support_tickets,
        "avg_resolution_time":   24,
        "complaint_type":        0,
        "csat_score":            csat_score,
        "escalations":           0,
        "email_open_rate":       30,
        "marketing_click_rate":  10,
        "nps_score":             nps_score,
        "survey_response":       0,
        "referral_count":        0
    }

    response = requests.post("http://localhost:8000/predict", json=payload)
    result   = response.json()

    st.header("📊 Résultat de la Prédiction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilité de Churn", f"{result['churn_probability']:.2%}")
    col2.metric("Décision", "⚠️ Churn" if result['churn'] else "✅ Fidèle")
    col3.metric("Niveau de Risque", result['risk_level'])

    prob  = result['churn_probability']
    color = "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
    st.markdown(f"""
        <div style='background:{color}; padding:15px; border-radius:10px;
                    text-align:center; color:white; font-size:22px; font-weight:bold'>
            Risque : {prob:.2%} — {result['risk_level']}
        </div>
    """, unsafe_allow_html=True)

    st.info(f"🤖 Modèle utilisé : **{model_choice}**")

# ── Vue globale ────────────────────────────────────────────
st.markdown("---")
st.header("📈 Vue Globale du Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Total clients",   len(df))
col2.metric("Clients churnés", int(df['churn'].sum()))
col3.metric("Taux de churn",   f"{df['churn'].mean():.2%}")

col4, col5 = st.columns(2)

with col4:
    fig, ax = plt.subplots(figsize=(5, 3))
    df['churn'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
    ax.set_title("Distribution du Churn")
    ax.set_xticklabels(["Fidèle", "Churné"], rotation=0)
    st.pyplot(fig)

with col5:
    fig, ax = plt.subplots(figsize=(5, 3))
    df.groupby('contract_type')['churn'].mean().plot(kind='bar', ax=ax, color='coral')
    ax.set_title("Taux de churn par type de contrat")
    ax.set_ylabel("Taux de churn")
    ax.set_xlabel("Type de contrat")
    plt.xticks(rotation=0)
    st.pyplot(fig)

col6, col7 = st.columns(2)

with col6:
    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column='nps_score', by='churn', ax=ax)
    ax.set_title("NPS Score selon le Churn")
    ax.set_xlabel("Churn")
    plt.suptitle("")
    st.pyplot(fig)

with col7:
    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column='monthly_fee', by='churn', ax=ax)
    ax.set_title("Frais mensuels selon le Churn")
    ax.set_xlabel("Churn")
    plt.suptitle("")
    st.pyplot(fig)