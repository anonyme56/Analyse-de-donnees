"""
Etape 5 - modelisation predictive + evaluation.

Objectif : estimer la probabilite de resiliation (churn) d'un client.
On compare 3 modeles : regression logistique, arbre de decision, random
forest. Metriques : AUC, F1, precision, rappel, + matrice de confusion.
On regarde aussi le sur-apprentissage (ecart train/test) et un proxy tres
simple de "cout ecoresponsable" (temps d'entrainement, nombre de parametres
/ profondeur du modele) pour comparer les modeles sur un autre axe que la
seule performance predictive.
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)

RACINE = Path(__file__).resolve().parent.parent
DF = pd.read_csv(RACINE / "data" / "clients_ecoprime_propre.csv")
DOSSIER_FIG = RACINE / "outputs" / "figures" / "modelisation"
DOSSIER_FIG.mkdir(parents=True, exist_ok=True)

VAR_QUANTI = [
    "age", "anciennete_mois", "nombre_produits", "score_credit",
    "salaire_annuel_net", "solde_moyen_compte", "montant_moyen_transaction",
    "nb_transactions_mois", "nb_connexions_appli_mois",
    "jours_depuis_derniere_connexion", "nb_reclamations_12mois",
    "score_satisfaction",
]
VAR_QUALI_BINAIRE = ["carte_credit", "membre_actif", "utilise_appli_mobile"]
VAR_QUALI_NOMINALE = ["sexe", "situation_familiale", "region", "type_compte", "canal_contact_prefere"]


def preparer_matrice():
    df = DF.dropna(subset=["score_satisfaction"]).copy()
    X = df[VAR_QUANTI + VAR_QUALI_BINAIRE].copy()
    X = pd.concat([X, pd.get_dummies(df[VAR_QUALI_NOMINALE], drop_first=True)], axis=1)
    y = df["churn"]
    return X, y


def evaluer_modele(nom, modele, X_train, X_test, y_train, y_test, resultats):
    t0 = time.perf_counter()
    modele.fit(X_train, y_train)
    duree = time.perf_counter() - t0

    proba_test = modele.predict_proba(X_test)[:, 1]
    pred_test = modele.predict(X_test)
    proba_train = modele.predict_proba(X_train)[:, 1]

    auc_test = roc_auc_score(y_test, proba_test)
    auc_train = roc_auc_score(y_train, proba_train)

    resultats[nom] = {
        "auc_test": round(auc_test, 4),
        "auc_train": round(auc_train, 4),
        "ecart_train_test": round(auc_train - auc_test, 4),
        "f1": round(f1_score(y_test, pred_test), 4),
        "precision": round(precision_score(y_test, pred_test), 4),
        "rappel": round(recall_score(y_test, pred_test), 4),
        "temps_entrainement_s": round(duree, 4),
    }
    return proba_test, pred_test


def modeliser():
    X, y = preparer_matrice()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1742, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    resultats = {}
    courbes_roc = {}

    # logistique
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")
    proba, pred = evaluer_modele("regression_logistique", log_reg, X_train_std, X_test_std, y_train, y_test, resultats)
    courbes_roc["regression_logistique"] = roc_curve(y_test, proba)
    matrice_log = confusion_matrix(y_test, pred)

    # arbre de decision
    arbre = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=1742)
    proba, pred = evaluer_modele("arbre_de_decision", arbre, X_train, X_test, y_train, y_test, resultats)
    courbes_roc["arbre_de_decision"] = roc_curve(y_test, proba)

    # random forest
    foret = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=1742, n_jobs=-1
    )
    proba, pred = evaluer_modele("random_forest", foret, X_train, X_test, y_train, y_test, resultats)
    courbes_roc["random_forest"] = roc_curve(y_test, proba)
    matrice_foret = confusion_matrix(y_test, pred)

    # validation croisee (5-fold) sur la regression logistique pour verifier la stabilite
    scores_cv = cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        scaler.fit_transform(X), y, cv=5, scoring="roc_auc"
    )
    resultats["regression_logistique"]["auc_cv5_moyenne"] = round(scores_cv.mean(), 4)
    resultats["regression_logistique"]["auc_cv5_ecart_type"] = round(scores_cv.std(), 4)

    # variables les plus importantes (random forest)
    importances = pd.Series(foret.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

    # --- figures ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nom, (fpr, tpr, _) in courbes_roc.items():
        ax.plot(fpr, tpr, label=f"{nom} (AUC={resultats[nom]['auc_test']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("taux de faux positifs")
    ax.set_ylabel("taux de vrais positifs")
    ax.set_title("Courbes ROC - comparaison des modeles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "courbes_roc.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    importances.sort_values().plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_title("Importance des variables (random forest)")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "importance_variables.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m, titre in zip(axes, [matrice_log, matrice_foret], ["Regression logistique", "Random forest"]):
        ax.imshow(m, cmap="Blues")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Non", "Oui"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Non", "Oui"])
        ax.set_xlabel("predit"); ax.set_ylabel("reel")
        ax.set_title(titre)
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "matrices_confusion.png", dpi=110)
    plt.close(fig)

    with open(RACINE / "outputs" / "resultats_modeles.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    with open(RACINE / "outputs" / "importance_variables.txt", "w", encoding="utf-8") as f:
        f.write(importances.round(4).to_string())

    print(json.dumps(resultats, indent=2, ensure_ascii=False))
    return resultats


if __name__ == "__main__":
    modeliser()
