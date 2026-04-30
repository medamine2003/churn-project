import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)
from src.preprocessing import load_and_prepare

def evaluate_all():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()

    model_names = ["logistic", "random_forest", "xgboost", "mlp"]
    results = {}

    for name in model_names:
        model = joblib.load(f"models/{name}.pkl")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "f1":      f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "report":  classification_report(y_test, y_pred),
            "cm":      confusion_matrix(y_test, y_pred),
            "y_proba": y_proba
        }

        print(f"\n{'='*40}")
        print(f"Modèle : {name}")
        print(results[name]["report"])
        print(f"ROC-AUC : {results[name]['roc_auc']:.4f}")

    return results, y_test

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (name, res) in zip(axes, results.items()):
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
    plt.suptitle("Matrices de confusion", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.2f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbes ROC")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison(results):
    names = list(results.keys())
    f1_scores  = [results[n]["f1"]      for n in names]
    auc_scores = [results[n]["roc_auc"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, f1_scores,  width, label="F1-Score",  color="steelblue")
    ax.bar(x + width/2, auc_scores, width, label="ROC-AUC",   color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des modèles")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results, y_test = evaluate_all()
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_comparison(results)