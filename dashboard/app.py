import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("Plateforme de Rétention Client")

# ─────────────────────────────────────────────────────────────
# CHEMINS ABSOLUS
# On utilise __file__ pour construire les chemins indépendamment
# de l'endroit depuis lequel on lance le script
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "customer_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Chargement du dataset brut pour les visualisations globales
df = pd.read_csv(DATA_PATH)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Configuration et saisie du profil client
# L'utilisateur choisit le modèle et renseigne les paramètres
# du client à analyser
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

# Sélection du modèle de prédiction
model_choice = st.sidebar.selectbox("Modèle de prédiction", [
    "random_forest", "logistic", "xgboost", "mlp"
], format_func=lambda x: {
    "random_forest": "Random Forest",
    "logistic":      "Régression Logistique",
    "xgboost":       "Gradient Boosting",
    "mlp":           "Réseau de Neurones (MLP)"
}[x])

st.sidebar.markdown("---")
st.sidebar.header("Profil Client")

# Critères du client saisis manuellement par l'utilisateur métier
gender              = st.sidebar.selectbox("Genre", [0, 1], format_func=lambda x: "Homme" if x == 0 else "Femme")
age                 = st.sidebar.slider("Age", 18, 80, 35)
tenure_months       = st.sidebar.slider("Ancienneté (mois)", 0, 72, 12)
monthly_fee         = st.sidebar.number_input("Frais mensuels (euros)", 0, 500, 50)
payment_failures    = st.sidebar.slider("Echecs de paiement", 0, 10, 0)
support_tickets     = st.sidebar.slider("Tickets support", 0, 20, 1)
nps_score           = st.sidebar.slider("NPS Score", -100, 100, 50)
csat_score          = st.sidebar.slider("CSAT Score", 0, 10, 7)
monthly_logins      = st.sidebar.slider("Connexions par mois", 0, 100, 10)
last_login_days_ago = st.sidebar.slider("Derniere connexion (jours)", 0, 365, 7)
contract_type       = st.sidebar.selectbox("Type de contrat", [0, 1, 2],
                        format_func=lambda x: ["Mensuel", "Annuel", "Deux ans"][x])

# ─────────────────────────────────────────────────────────────
# PREDICTION EN TEMPS REEL
# Quand l'utilisateur clique sur le bouton, on envoie les
# données à l'API FastAPI via une requête POST /predict
# L'API retourne la probabilité de churn et le niveau de risque
# ─────────────────────────────────────────────────────────────
if st.sidebar.button("Prédire le Churn"):

    # Construction du payload envoyé à l'API
    # Les variables non saisies par l'utilisateur sont fixées
    # à leurs valeurs moyennes/par défaut
    payload = {
        "model_name":             model_choice,
        "gender":                 gender,
        "age":                    age,
        "country":                0,
        "city":                   0,
        "customer_segment":       0,
        "tenure_months":          tenure_months,
        "signup_channel":         0,
        "contract_type":          contract_type,
        "monthly_logins":         monthly_logins,
        "weekly_active_days":     3,
        "avg_session_time":       30,
        "features_used":          10,
        "usage_growth_rate":      0,
        "last_login_days_ago":    last_login_days_ago,
        "monthly_fee":            monthly_fee,
        "total_revenue":          monthly_fee * tenure_months,  # calculé automatiquement
        "payment_method":         0,
        "payment_failures":       payment_failures,
        "discount_applied":       0,
        "price_increase_last_3m": 0,
        "support_tickets":        support_tickets,
        "avg_resolution_time":    24,
        "complaint_type":         0,
        "csat_score":             csat_score,
        "escalations":            0,
        "email_open_rate":        30,
        "marketing_click_rate":   10,
        "nps_score":              nps_score,
        "survey_response":        0,
        "referral_count":         0
    }

    # Appel à l'API REST
    response = requests.post("http://localhost:8000/predict", json=payload)
    result   = response.json()

    st.header("Résultat de la Prédiction")

    # Affichage des 3 métriques principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilité de Churn", f"{result['churn_probability']:.2%}")
    col2.metric("Décision", "Churn" if result['churn'] else "Fidèle")
    col3.metric("Niveau de Risque", result['risk_level'])

    # Jauge colorée selon le niveau de risque
    # Vert = faible, Orange = moyen, Rouge = haut
    prob  = result['churn_probability']
    color = "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"
    st.markdown(f"""
        <div style='background:{color}; padding:15px; border-radius:10px;
                    text-align:center; color:white; font-size:22px; font-weight:bold'>
            Risque : {prob:.2%} — {result['risk_level']}
        </div>
    """, unsafe_allow_html=True)
    st.info(f"Modèle utilisé : {model_choice}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 1 : KPI METIER
# On charge le modèle Random Forest pour scorer tous les clients
# du dataset et calculer les indicateurs clés business
# ─────────────────────────────────────────────────────────────
st.header("KPI Métier")

# Chargement du modèle et du scaler
model_rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Encodage du dataset complet pour le scoring
from sklearn.preprocessing import LabelEncoder
df_encoded = df.drop(columns=['customer_id'])
cat_cols   = df_encoded.select_dtypes(include='object').columns
for col in cat_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Calcul des probabilités de churn pour chaque client
X        = df_encoded.drop(columns=['churn'])
X_scaled = scaler.transform(X)
probas   = model_rf.predict_proba(X_scaled)[:, 1]

# Ajout des colonnes calculées au dataframe
df['churn_proba']     = probas
df['revenu_a_risque'] = df['total_revenue'] * df['churn_proba']

# Calcul des KPI globaux
total_revenu_risque  = df['revenu_a_risque'].sum()
clients_haut_risque  = (df['churn_proba'] > 0.7).sum()
clients_moyen_risque = ((df['churn_proba'] > 0.4) & (df['churn_proba'] <= 0.7)).sum()

# Affichage des KPI sous forme de métriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total clients",        len(df))
col2.metric("Clients haut risque",  clients_haut_risque)
col3.metric("Clients moyen risque", clients_moyen_risque)
col4.metric("Revenu a risque",      f"{total_revenu_risque:,.0f} euros")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 2 : ANALYSE GLOBALE
# Visualisations orientées métier pour comprendre la distribution
# du churn selon les variables clés
# ─────────────────────────────────────────────────────────────
st.header("Analyse Globale")

col4, col5 = st.columns(2)

with col4:
    # Distribution globale churn vs non-churn
    fig, ax = plt.subplots(figsize=(5, 3))
    df['churn'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
    ax.set_title("Distribution du Churn")
    ax.set_xticklabels(["Fidèle", "Churné"], rotation=0)
    st.pyplot(fig)

with col5:
    # Taux de churn selon le type de contrat
    # Permet d'identifier quel contrat est le plus risqué
    fig, ax = plt.subplots(figsize=(5, 3))
    df.groupby('contract_type')['churn'].mean().plot(kind='bar', ax=ax, color='coral')
    ax.set_title("Taux de churn par type de contrat")
    ax.set_ylabel("Taux de churn")
    ax.set_xlabel("Type de contrat")
    plt.xticks(rotation=0)
    st.pyplot(fig)

col6, col7 = st.columns(2)

with col6:
    # NPS Score selon churn : un NPS bas est souvent précurseur de résiliation
    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column='nps_score', by='churn', ax=ax)
    ax.set_title("NPS Score selon le Churn")
    ax.set_xlabel("Churn")
    plt.suptitle("")
    st.pyplot(fig)

with col7:
    # Frais mensuels selon churn : les clients qui paient plus churnent-ils plus ?
    fig, ax = plt.subplots(figsize=(5, 3))
    df.boxplot(column='monthly_fee', by='churn', ax=ax)
    ax.set_title("Frais mensuels selon le Churn")
    ax.set_xlabel("Churn")
    plt.suptitle("")
    st.pyplot(fig)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 3 : COMPARAISON DES MODELES
# Évaluation des 4 modèles sur le test set avec les métriques
# adaptées au déséquilibre des classes :
# F1-Score, Recall, ROC-AUC, PR-AUC
# ─────────────────────────────────────────────────────────────
st.header("Comparaison des Modèles")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, recall_score, average_precision_score

# Reconstruction du test set avec le même random_state que l'entraînement
y = df_encoded['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

model_names = ["logistic", "random_forest", "xgboost", "mlp"]
comparison  = []

# Évaluation de chaque modèle sur le test set
for name in model_names:
    m       = joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))
    y_pred  = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:, 1]
    comparison.append({
        "Modele":   name,
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "Recall":   round(recall_score(y_test, y_pred), 4),
        "ROC-AUC":  round(roc_auc_score(y_test, y_proba), 4),
        "PR-AUC":   round(average_precision_score(y_test, y_proba), 4),
    })

# Tableau comparatif
df_comparison = pd.DataFrame(comparison)
st.dataframe(df_comparison.set_index("Modele"))

# Graphe comparatif des métriques
fig, ax = plt.subplots(figsize=(10, 4))
x     = np.arange(len(model_names))
width = 0.2
ax.bar(x - width*1.5, df_comparison["F1-Score"], width, label="F1-Score", color="steelblue")
ax.bar(x - width/2,   df_comparison["ROC-AUC"],  width, label="ROC-AUC",  color="tomato")
ax.bar(x + width/2,   df_comparison["PR-AUC"],   width, label="PR-AUC",   color="green")
ax.bar(x + width*1.5, df_comparison["Recall"],   width, label="Recall",   color="orange")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1)
ax.set_title("Comparaison des modèles")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 4 : FEATURE IMPORTANCE
# Identification des variables les plus influentes dans
# la décision du Random Forest
# Permet au responsable CRM de comprendre POURQUOI un client
# est classé à risque
# ─────────────────────────────────────────────────────────────
st.header("Variables les plus influentes (Random Forest)")

feature_names = X.columns.tolist()
importances   = model_rf.feature_importances_

# Tri des variables par importance décroissante (top 10)
indices = np.argsort(importances)[::-1][:10]

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(
    [feature_names[i] for i in indices][::-1],
    importances[indices][::-1],
    color="steelblue"
)
ax.set_title("Top 10 variables les plus influentes")
ax.set_xlabel("Importance")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 5 : TOP CLIENTS A RISQUE
# Liste des 10 clients avec la probabilité de churn la plus
# élevée — permet de prioriser les actions de rétention
# ─────────────────────────────────────────────────────────────
st.header("Top 10 Clients a Haut Risque")

# Sélection et tri des colonnes pertinentes pour la décision
df_risk = df[['churn_proba', 'revenu_a_risque', 'monthly_fee',
              'tenure_months', 'contract_type']].copy()
df_risk = df_risk.sort_values('churn_proba', ascending=False).head(10)
df_risk.columns = ['Probabilite Churn', 'Revenu a Risque (euros)',
                   'Frais Mensuels (euros)', 'Anciennete (mois)', 'Type Contrat']

# Formatage des colonnes pour la lisibilité
df_risk['Probabilite Churn']       = df_risk['Probabilite Churn'].apply(lambda x: f"{x:.2%}")
df_risk['Revenu a Risque (euros)'] = df_risk['Revenu a Risque (euros)'].apply(lambda x: f"{x:,.0f}")
st.dataframe(df_risk)