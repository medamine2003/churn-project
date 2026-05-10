# 🔮 Churn Prediction Project

Système intelligent de prédiction du churn client basé sur le Machine Learning.
Projet M1 Dev. Manager Full Stack — EFREI 2025-26

---

## 📁 Structure du projet

churn-project/
├── data/                    ← dataset (customer_churn.csv)
├── models/                  ← modèles entraînés (.pkl)
├── notebooks/
│   └── eda.ipynb            ← analyse exploratoire + entraînement
├── src/
│   ├── preprocessing.py     ← nettoyage + encodage + split
│   ├── train.py             ← entraînement des modèles
│   ├── evaluate.py          ← évaluation + métriques
│   └── shap_analysis.py     ← interprétabilité SHAP
├── api/
│   └── main.py              ← API REST FastAPI
├── dashboard/
│   └── app.py               ← Dashboard Streamlit
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1. Cloner le projet
git clone https://github.com/TON_USERNAME/churn-project.git
cd churn-project

### 2. Créer et activer l'environnement virtuel
python -m venv venv

Windows :
venv\Scripts\activate

Mac/Linux :
source venv/bin/activate

### 3. Installer les dépendances
pip install -r requirements.txt

### 4. Télécharger le dataset
Télécharger customer_churn.csv depuis :
https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset
Placer le fichier dans le dossier data/

---

## 🚀 Lancement

### Étape 1 — Entraîner les modèles
Ouvrir notebooks/eda.ipynb et exécuter toutes les cellules dans l'ordre.
Les modèles seront sauvegardés automatiquement dans models/

### Étape 2 — Lancer l'API REST
uvicorn api.main:app --reload

API disponible sur        : http://localhost:8000
Documentation Swagger     : http://localhost:8000/docs

### Étape 3 — Lancer le Dashboard
Ouvrir un second terminal, activer le venv, puis :
streamlit run dashboard/app.py

Dashboard disponible sur : http://localhost:8501

---

## 🤖 Modèles disponibles

| Modèle          | Description                                    |
|-----------------|------------------------------------------------|
| logistic        | Régression Logistique — baseline interprétable |
| random_forest   | Random Forest — capture les non-linéarités     |
| xgboost         | Gradient Boosting — haute performance          |
| mlp             | Réseau de Neurones — interactions complexes    |

---

## 🌐 API REST

### Endpoints

| Méthode | Endpoint   | Description              |
|---------|------------|--------------------------|
| GET     | /health    | Vérifie l'état de l'API  |
| POST    | /predict   | Prédit le churn d'un client |

### Exemple de réponse /predict
{
  "model_used": "random_forest",
  "churn": 0,
  "churn_probability": 0.09,
  "risk_level": "Faible"
}

---

## 📊 Gestion du déséquilibre des classes

Le dataset est déséquilibré : 90% non-churn / 10% churn.
Deux techniques ont été comparées :

| Technique             | Description                                        |
|-----------------------|----------------------------------------------------|
| Baseline              | Sans rééquilibrage — point de référence            |
| SMOTE                 | Synthetic Minority Over-sampling                   |
| Random Over-Sampling  | Duplication aléatoire de la classe minoritaire     |

Stratégie retenue : SMOTE
Métriques utilisées : F1-Score, Recall, ROC-AUC, PR-AUC

---

## 📦 Dépendances principales

pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
seaborn
joblib
streamlit
fastapi
uvicorn
requests
jupyter
ipykernel
pydantic

---

##  Équipe
Projet réalisé par Mohamed Amine Aissaoui et Brakissa Diomandé dans le cadre du module Data Science 