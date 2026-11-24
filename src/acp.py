"""
Etape 4 - ACP (reduction de dimension).

On standardise les variables quantitatives, on lance une ACP, on regarde
la variance expliquee par composante, et on projette les individus sur les
deux premiers axes en coloriant par churn pour voir si une structure se
degage. On regarde aussi les contributions des variables aux axes (loadings).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DF = pd.read_csv(RACINE / "data" / "clients_ecoprime_propre.csv")
DOSSIER_FIG = RACINE / "outputs" / "figures" / "acp"
DOSSIER_FIG.mkdir(parents=True, exist_ok=True)

VAR_QUANTI = [
    "age", "anciennete_mois", "nombre_produits", "score_credit",
    "salaire_annuel_net", "solde_moyen_compte", "montant_moyen_transaction",
    "nb_transactions_mois", "nb_connexions_appli_mois",
    "jours_depuis_derniere_connexion", "nb_reclamations_12mois",
]


def lancer_acp():
    X = DF[VAR_QUANTI].dropna()
    churn = DF.loc[X.index, "churn"]

    X_std = StandardScaler().fit_transform(X)

    acp = PCA(n_components=len(VAR_QUANTI))
    composantes = acp.fit_transform(X_std)

    variance_expliquee = acp.explained_variance_ratio_
    variance_cumulee = np.cumsum(variance_expliquee)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(1, len(variance_expliquee) + 1), variance_expliquee, color="#3b6ea5", label="par axe")
    ax.plot(range(1, len(variance_expliquee) + 1), variance_cumulee, color="#c0392b", marker="o", label="cumulee")
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("axe")
    ax.set_ylabel("variance expliquee")
    ax.set_title("Eboulis des valeurs propres (ACP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "eboulis.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    couleurs = np.where(churn == 1, "#c0392b", "#3b6ea5")
    ax.scatter(composantes[:, 0], composantes[:, 1], c=couleurs, alpha=0.35, s=14)
    ax.set_xlabel(f"axe 1 ({variance_expliquee[0]*100:.1f}% de variance)")
    ax.set_ylabel(f"axe 2 ({variance_expliquee[1]*100:.1f}% de variance)")
    ax.set_title("Projection des clients sur les 2 premiers axes (rouge = churn)")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "projection_axe1_axe2.png", dpi=110)
    plt.close(fig)

    loadings = pd.DataFrame(
        acp.components_[:3].T, index=VAR_QUANTI, columns=["axe1", "axe2", "axe3"]
    ).sort_values("axe1", key=abs, ascending=False)

    chemin_txt = RACINE / "outputs" / "acp_resultats.txt"
    with open(chemin_txt, "w", encoding="utf-8") as f:
        f.write("Variance expliquee par axe :\n")
        for i, v in enumerate(variance_expliquee, start=1):
            f.write(f"  axe{i} : {v*100:.1f}% (cumule {variance_cumulee[i-1]*100:.1f}%)\n")
        nb_axes_80 = np.argmax(variance_cumulee >= 0.8) + 1
        f.write(f"\nIl faut {nb_axes_80} axes pour depasser 80% de variance expliquee.\n")
        f.write("\nContributions des variables aux 3 premiers axes (loadings) :\n")
        f.write(loadings.round(3).to_string())

    print(f"resultats ACP -> {chemin_txt}")


if __name__ == "__main__":
    lancer_acp()
