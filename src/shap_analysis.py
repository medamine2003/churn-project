import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import load_and_prepare

def run_shap():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    model = joblib.load("models/random_forest.pkl")

    #  200 lignes pour aller vite
    X_sample = X_test[:200]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    #  Fix : vérifier la forme de shap_values
    print("Type shap_values :", type(shap_values))
    print("Shape :", np.array(shap_values).shape)

    # Si shap_values est une liste de 2 arrays → prendre [1] (classe churn)
    # Si shap_values est un seul array → le prendre directement
    if isinstance(shap_values, list):
        sv = shap_values[1]  # classe 1 = churn
    else:
        sv = shap_values

    print(" SHAP Summary Plot (global)")
    shap.summary_plot(sv, X_sample, feature_names=feature_names)

    print(" SHAP Bar Plot")
    shap.summary_plot(sv, X_sample, feature_names=feature_names, plot_type="bar")

if __name__ == "__main__":
    run_shap()