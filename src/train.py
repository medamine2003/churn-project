from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from src.preprocessing import load_and_prepare
import joblib
import numpy as np
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def train_all():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    print(f"Distribution originale — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")

    # ── Comparaison des 2 techniques ──────────────────────
    samplers = {
        "baseline":   (X_train, y_train),
        "smote":      SMOTE(random_state=42).fit_resample(X_train, y_train),
        "oversample": RandomOverSampler(random_state=42).fit_resample(X_train, y_train),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced")

    print("\n📊 Comparaison des techniques de rééquilibrage (F1) :")
    for sampler_name, (X_res, y_res) in samplers.items():
        cv_scores = cross_val_score(baseline_model, X_res, y_res, cv=skf, scoring="f1")
        print(f"  {sampler_name:12} → F1 moyen : {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # ── Stratégie retenue : SMOTE ─────────────────────────
    X_resampled, y_resampled = samplers["smote"]
    print(f"\n✅ Stratégie retenue : SMOTE")
    print(f"Distribution après SMOTE — 0: {(y_resampled==0).sum()}, 1: {(y_resampled==1).sum()}")

    # ── Entraînement des 4 modèles ────────────────────────
    models = {
        "logistic":      LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "xgboost":       GradientBoostingClassifier(random_state=42),
        "mlp":           MLPClassifier(max_iter=300, random_state=42)
    }

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring="f1")
        print(f"\n{name} — CV F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        model.fit(X_resampled, y_resampled)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"✅ {name} entraîné et sauvegardé")

    return models, X_test, y_test

if __name__ == "__main__":
    train_all()