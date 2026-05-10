# Bibliothèque de manipulation de données
import pandas as pd

# Outils de prétraitement des données
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fonction de séparation train/test
from sklearn.model_selection import train_test_split

# Sauvegarde des objets Python (ex : scaler)
import joblib

# Gestion des chemins et dossiers
import os


# ==========================================================
# Définition des chemins du projet
# ==========================================================

# Dossier racine du projet
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Chemin du fichier CSV contenant les données
DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "customer_churn.csv"
)

# Dossier de sauvegarde des modèles
MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ==========================================================
# Fonction principale de préparation des données
# ==========================================================

def load_and_prepare():

    # ==========================================================
    # Chargement des données
    # ==========================================================

    # Lecture du fichier CSV avec pandas
    df = pd.read_csv(DATA_PATH)

    # ==========================================================
    # Nettoyage des données
    # ==========================================================

    # Suppression de la colonne customer_id
    # car elle ne contient pas d'information utile
    # pour l'entraînement du modèle
    df = df.drop(columns=['customer_id'])

    # ==========================================================
    # Encodage des variables catégorielles
    # ==========================================================

    # Sélection des colonnes de type texte/object
    cat_cols = df.select_dtypes(include='object').columns

    # Transformation des catégories en nombres
    # Exemple :
    # "Male" -> 0
    # "Female" -> 1
    for col in cat_cols:

        # Encodage de chaque colonne catégorielle
        df[col] = LabelEncoder().fit_transform(df[col])

    # ==========================================================
    # Séparation des variables explicatives et cible
    # ==========================================================

    # Variables d'entrée (features)
    X = df.drop(columns=['churn'])

    # Variable cible
    y = df['churn']

    # ==========================================================
    # Séparation entraînement / test
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        # Données
        X,
        y,

        # 20% des données pour le test
        test_size=0.2,

        # Conservation de la proportion des classes
        stratify=y,

        # Reproductibilité des résultats
        random_state=42
    )

    # ==========================================================
    # Standardisation des données
    # ==========================================================

    # Création du scaler
    # StandardScaler :
    # moyenne = 0
    # écart-type = 1
    scaler = StandardScaler()

    # Apprentissage des statistiques sur le train
    # puis transformation
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformation du jeu de test
    # IMPORTANT :
    # on utilise le scaler appris sur le train
    X_test_scaled = scaler.transform(X_test)

    # ==========================================================
    # Sauvegarde du scaler
    # ==========================================================

    # Création du dossier models s'il n'existe pas
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Sauvegarde du scaler au format .pkl
    joblib.dump(
        scaler,
        os.path.join(MODELS_DIR, "scaler.pkl")
    )

    print(f"✅ Scaler sauvegardé dans : {MODELS_DIR}")

    # ==========================================================
    # Retour des données préparées
    # ==========================================================

    return (

        # Données d'entraînement normalisées
        X_train_scaled,

        # Données de test normalisées
        X_test_scaled,

        # Labels d'entraînement
        y_train,

        # Labels de test
        y_test,

        # Noms des variables
        X.columns.tolist()
    )