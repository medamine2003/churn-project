# Bibliothèque d'interprétabilité des modèles
import shap

# Chargement des modèles sauvegardés
import joblib

# Bibliothèque pour le calcul numérique
import numpy as np

# Bibliothèque de visualisation
import matplotlib.pyplot as plt

# Fonction personnalisée de préparation des données
from src.preprocessing import load_and_prepare


# ==========================================================
# Fonction principale d'analyse SHAP
# ==========================================================

def run_shap():

    # Chargement et préparation des données
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    # Chargement du modèle Random Forest déjà entraîné
    model = joblib.load("models/random_forest.pkl")

    # ==========================================================
    # Sélection d'un échantillon
    # ==========================================================

    # On limite à 200 lignes pour accélérer les calculs SHAP
    # (SHAP peut être très coûteux en temps de calcul)
    X_sample = X_test[:200]

    # ==========================================================
    # Création de l'explainer SHAP
    # ==========================================================

    # TreeExplainer est optimisé pour les modèles basés sur les arbres
    # comme Random Forest, XGBoost, LightGBM, etc.
    explainer = shap.TreeExplainer(model)

    # Calcul des valeurs SHAP
    # Les SHAP values indiquent l'impact de chaque variable
    # sur la prédiction du modèle
    shap_values = explainer.shap_values(X_sample)

    # ==========================================================
    # Vérification du format des SHAP values
    # ==========================================================

    # Affichage du type de l'objet retourné
    print("Type shap_values :", type(shap_values))

    # Affichage de la dimension du tableau
    print("Shape :", np.array(shap_values).shape)

    # ==========================================================
    # Gestion des différents formats SHAP
    # ==========================================================

    # Cas fréquent en classification binaire :
    # shap_values = liste contenant :
    #   - classe 0
    #   - classe 1
    #
    # Ici on récupère les SHAP values de la classe 1
    # (par exemple : churn = oui)

    if isinstance(shap_values, list):

        # Classe positive = churn
        sv = shap_values[1]

    else:

        # Cas où SHAP retourne directement un tableau
        sv = shap_values

    # ==========================================================
    # Visualisation globale : Summary Plot
    # ==========================================================

    print("SHAP Summary Plot (global)")

    # Graphique principal SHAP :
    # - importance globale des variables
    # - impact positif/négatif sur la prédiction
    # - distribution des valeurs
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_names
    )

    # ==========================================================
    # Visualisation : Bar Plot
    # ==========================================================

    print("SHAP Bar Plot")

    # Version simplifiée :
    # affiche uniquement l'importance moyenne des variables
    shap.summary_plot(
        sv,
        X_sample,
        feature_names=feature_names,
        plot_type="bar"
    )


# ==========================================================
# Point d'entrée du script
# ==========================================================

if __name__ == "__main__":

    # Lancement de l'analyse SHAP
    run_shap()