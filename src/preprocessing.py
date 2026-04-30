import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ✅ Chemin basé sur l'emplacement du fichier preprocessing.py lui-même
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "customer_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['customer_id'])

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"✅ Scaler sauvegardé dans : {MODELS_DIR}")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()