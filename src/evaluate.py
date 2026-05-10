# ==========================================================
# Import des bibliothèques
# ==========================================================

# Chargement des modèles sauvegardés
import joblib

# Calcul numérique
import numpy as np

# Visualisation graphique
import matplotlib.pyplot as plt
import seaborn as sns

# Métriques d'évaluation des modèles
from sklearn.metrics import (

    # Rapport détaillé de classification
    classification_report,

    # Matrice de confusion
    confusion_matrix,

    # Aire sous la courbe ROC
    roc_auc_score,

    # Coordonnées de la courbe ROC
    roc_curve,

    # Score F1
    f1_score,

    # Courbe Precision-Recall
    precision_recall_curve,

    # Aire sous la courbe Precision-Recall
    average_precision_score,

    # Recall
    recall_score
)

# Fonction personnalisée de préparation des données
from src.preprocessing import load_and_prepare

# Gestion des chemins système
import os


# ==========================================================
# Définition des chemins du projet
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(BASE_DIR, "models")


# ==========================================================
# Fonction principale d'évaluation des modèles
# ==========================================================

def evaluate_all():

    # Chargement des données préparées
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    # Liste des modèles à évaluer
    model_names = [
        "logistic",
        "random_forest",
        "xgboost",
        "mlp"
    ]

    # Dictionnaire pour stocker les résultats
    results = {}

    # ==========================================================
    # Boucle sur chaque modèle
    # ==========================================================

    for name in model_names:

        # Chargement du modèle sauvegardé
        model = joblib.load(
            os.path.join(MODELS_DIR, f"{name}.pkl")
        )

        # ==========================================================
        # Prédictions
        # ==========================================================

        # Prédictions des classes (0 ou 1)
        y_pred = model.predict(X_test)

        # Probabilités de la classe positive
        # [:,1] = probabilité de la classe 1
        y_proba = model.predict_proba(X_test)[:, 1]

        # ==========================================================
        # Calcul des métriques
        # ==========================================================

        results[name] = {

            # F1-score
            "f1": f1_score(y_test, y_pred),

            # Recall
            "recall": recall_score(y_test, y_pred),

            # ROC-AUC
            "roc_auc": roc_auc_score(y_test, y_proba),

            # Precision-Recall AUC
            "pr_auc": average_precision_score(y_test, y_proba),

            # Rapport détaillé
            "report": classification_report(y_test, y_pred),

            # Matrice de confusion
            "cm": confusion_matrix(y_test, y_pred),

            # Probabilités prédites
            "y_proba": y_proba
        }

        # ==========================================================
        # Affichage des résultats
        # ==========================================================

        print(f"\n{'='*40}")

        print(f"Modèle : {name}")

        print(results[name]["report"])

        print(f"ROC-AUC : {results[name]['roc_auc']:.4f}")

        print(f"PR-AUC  : {results[name]['pr_auc']:.4f}")

        print(f"Recall  : {results[name]['recall']:.4f}")

    # Retour des résultats
    return results, y_test


# ==========================================================
# Affichage des matrices de confusion
# ==========================================================

def plot_confusion_matrices(results, y_test):

    # Création de 4 graphiques côte à côte
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Boucle sur chaque modèle
    for ax, (name, res) in zip(axes, results.items()):

        # Heatmap de la matrice de confusion
        sns.heatmap(
            res["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        # Titres et labels
        ax.set_title(f"{name}")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    # Titre global
    plt.suptitle(
        "Matrices de confusion",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ==========================================================
# Affichage des courbes ROC
# ==========================================================

def plot_roc_curves(results, y_test):

    plt.figure(figsize=(8, 6))

    # Boucle sur les modèles
    for name, res in results.items():

        # Calcul des coordonnées ROC
        fpr, tpr, _ = roc_curve(
            y_test,
            res["y_proba"]
        )

        # Tracé de la courbe
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={res['roc_auc']:.2f})"
        )

    # Ligne aléatoire de référence
    plt.plot([0,1], [0,1], 'k--')

    # Labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Courbes ROC")

    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# Affichage des courbes Precision-Recall
# ==========================================================

def plot_pr_curves(results, y_test):

    plt.figure(figsize=(8, 6))

    # Boucle sur les modèles
    for name, res in results.items():

        # Calcul precision / recall
        precision, recall, _ = precision_recall_curve(
            y_test,
            res["y_proba"]
        )

        # Tracé de la courbe
        plt.plot(
            recall,
            precision,
            label=f"{name} (PR-AUC={res['pr_auc']:.2f})"
        )

    # Labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Courbes Precision-Recall")

    plt.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# Comparaison globale des métriques
# ==========================================================

def plot_comparison(results):

    # Noms des modèles
    names = list(results.keys())

    # Extraction des métriques
    f1_scores = [
        results[n]["f1"]
        for n in names
    ]

    auc_scores = [
        results[n]["roc_auc"]
        for n in names
    ]

    pr_scores = [
        results[n]["pr_auc"]
        for n in names
    ]

    recall_scores = [
        results[n]["recall"]
        for n in names
    ]

    # Position des barres
    x = np.arange(len(names))

    # Largeur des barres
    width = 0.2

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 5))

    # ==========================================================
    # Barres des différentes métriques
    # ==========================================================

    ax.bar(
        x - width*1.5,
        f1_scores,
        width,
        label="F1-Score",
        color="steelblue"
    )

    ax.bar(
        x - width/2,
        auc_scores,
        width,
        label="ROC-AUC",
        color="tomato"
    )

    ax.bar(
        x + width/2,
        pr_scores,
        width,
        label="PR-AUC",
        color="green"
    )

    ax.bar(
        x + width*1.5,
        recall_scores,
        width,
        label="Recall",
        color="orange"
    )

    # Labels des modèles
    ax.set_xticks(x)
    ax.set_xticklabels(names)

    # Limite de l'axe Y
    ax.set_ylim(0, 1)

    # Titre
    ax.set_title(
        "Comparaison des modèles — métriques adaptées au déséquilibre"
    )

    # Légende
    ax.legend()

    plt.tight_layout()
    plt.show()


# ==========================================================
# Point d'entrée du script
# ==========================================================

if __name__ == "__main__":

    # Évaluation des modèles
    results, y_test = evaluate_all()

    # Affichage des matrices de confusion
    plot_confusion_matrices(results, y_test)

    # Affichage des courbes ROC
    plot_roc_curves(results, y_test)

    # Affichage des courbes Precision-Recall
    plot_pr_curves(results, y_test)

    # Comparaison globale des modèles
    plot_comparison(results)