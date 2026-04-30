from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from src.preprocessing import load_and_prepare
import joblib
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_all():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()
    # ✅ Le scaler est déjà sauvegardé dans preprocessing.py

    models = {
        "logistic":      LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost":       GradientBoostingClassifier(random_state=42),
        "mlp":           MLPClassifier(max_iter=300, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"✅ {name} entraîné et sauvegardé")

    return models, X_test, y_test

if __name__ == "__main__":
    train_all()