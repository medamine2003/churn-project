# Import des modèles de machine learning utilisés
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Outils pour la validation croisée
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Techniques de rééquilibrage des classes
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Fonction personnalisée pour charger et préparer les données
from src.preprocessing import load_and_prepare

# Bibliothèque pour sauvegarder les modèles entraînés
import joblib

# Bibliothèques utilitaires
import numpy as np
import os


# Définition des chemins du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# Fonction principale d'entraînement
def train_all():

    # Chargement et préparation des données
    # X_train / X_test = variables explicatives
    # y_train / y_test = variable cible
    # feature_names = noms des colonnes
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    # Affichage de la distribution initiale des classes
    print(f"Distribution originale — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

    # ==========================================================
    # Comparaison des techniques de rééquilibrage
    # ==========================================================

    # Dictionnaire contenant :
    # - les données originales
    # - les données équilibrées avec SMOTE
    # - les données équilibrées par duplication aléatoire
    samplers = {

        # Données sans rééquilibrage
        "baseline": (X_train, y_train),

        # Génération artificielle de nouvelles données minoritaires
        "smote": SMOTE(random_state=42).fit_resample(X_train, y_train),

        # Duplication aléatoire des exemples minoritaires
        "oversample": RandomOverSampler(random_state=42).fit_resample(X_train, y_train),
    }

    # Validation croisée stratifiée :
    # conserve la proportion des classes dans chaque fold
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Modèle de référence pour comparer les techniques
    baseline_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    print("\n📊 Comparaison des techniques de rééquilibrage (F1) :")

    # Boucle sur chaque stratégie de rééquilibrage
    for sampler_name, (X_res, y_res) in samplers.items():

        # Évaluation par validation croisée
        cv_scores = cross_val_score(
            baseline_model,
            X_res,
            y_res,
            cv=skf,
            scoring="f1"
        )

        # Affichage du score F1 moyen et de son écart-type
        print(
            f"  {sampler_name:12} → "
            f"F1 moyen : {np.mean(cv_scores):.4f} "
            f"(+/- {np.std(cv_scores):.4f})"
        )

    # ==========================================================
    # Stratégie retenue : SMOTE
    # ==========================================================

    # On récupère les données rééquilibrées avec SMOTE
    X_resampled, y_resampled = samplers["smote"]

    print(f"\n✅ Stratégie retenue : SMOTE")

    # Vérification de la nouvelle distribution des classes
    print(
        f"Distribution après SMOTE — "
        f"0: {(y_resampled==0).sum()}, "
        f"1: {(y_resampled==1).sum()}"
    )

    # ==========================================================
    # Définition des modèles à entraîner
    # ==========================================================

    models = {

        # Régression logistique
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        # Forêt aléatoire
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        # Gradient Boosting (appelé xgboost ici)
        # Attention : ce n'est PAS la vraie librairie XGBoost
        "xgboost": GradientBoostingClassifier(
            random_state=42
        ),

        # Réseau de neurones multicouche
        "mlp": MLPClassifier(
            max_iter=300,
            random_state=42
        )
    }

    # ==========================================================
    # Entraînement et sauvegarde des modèles
    # ==========================================================

    for name, model in models.items():

        # Validation croisée sur les données équilibrées
        cv_scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=skf,
            scoring="f1"
        )

        # Affichage des performances
        print(
            f"\n{name} — "
            f"CV F1: {np.mean(cv_scores):.4f} "
            f"(+/- {np.std(cv_scores):.4f})"
        )

        # Entraînement du modèle sur toutes les données équilibrées
        model.fit(X_resampled, y_resampled)

        # Sauvegarde du modèle au format .pkl
        joblib.dump(
            model,
            os.path.join(MODELS_DIR, f"{name}.pkl")
        )

        print(f"✅ {name} entraîné et sauvegardé")

    # Retour des modèles et du jeu de test
    return models, X_test, y_test


# Point d'entrée du script
if __name__ == "__main__":

    # Lancement de l'entraînement
    train_all()