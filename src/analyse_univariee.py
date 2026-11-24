"""
Etape 2 - analyse univariee.

Pour chaque variable on regarde sa distribution toute seule :
- quanti -> moyenne / mediane / variance / quartiles + histogramme + boxplot
- quali -> tableau de frequences + diagramme en barres

Les figures partent dans outputs/figures/univarie, un resume texte est
ecrit dans outputs/analyse_univariee.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DF = pd.read_csv(RACINE / "data" / "clients_ecoprime_propre.csv")
DOSSIER_FIG = RACINE / "outputs" / "figures" / "univarie"
DOSSIER_FIG.mkdir(parents=True, exist_ok=True)

VAR_QUANTI = [
    "age", "anciennete_mois", "nombre_produits", "score_credit",
    "salaire_annuel_net", "solde_moyen_compte", "montant_moyen_transaction",
    "nb_transactions_mois", "nb_connexions_appli_mois",
    "jours_depuis_derniere_connexion", "nb_reclamations_12mois",
]

VAR_QUALI = [
    "sexe", "situation_familiale", "region", "type_compte", "carte_credit",
    "membre_actif", "utilise_appli_mobile", "score_satisfaction",
    "canal_contact_prefere", "churn",
]

plt.rcParams["figure.autolayout"] = True


def analyser():
    lignes = []

    lignes.append("=== Variables quantitatives ===\n")
    for col in VAR_QUANTI:
        s = DF[col]
        desc = s.describe()
        lignes.append(
            f"\n{col} : moyenne={desc['mean']:.2f}, mediane={s.median():.2f}, "
            f"variance={s.var():.2f}, ecart-type={desc['std']:.2f}, "
            f"Q1={desc['25%']:.2f}, Q3={desc['75%']:.2f}, min={desc['min']:.2f}, max={desc['max']:.2f}"
        )

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        axes[0].hist(s.dropna(), bins=30, color="#3b6ea5", edgecolor="white")
        axes[0].set_title(f"Distribution de {col}")
        axes[1].boxplot(s.dropna(), vert=False)
        axes[1].set_title(f"Boxplot de {col}")
        fig.savefig(DOSSIER_FIG / f"{col}.png", dpi=110)
        plt.close(fig)

    lignes.append("\n\n=== Variables qualitatives ===\n")
    for col in VAR_QUALI:
        freq = DF[col].value_counts(dropna=False)
        pct = (freq / len(DF) * 100).round(1)
        lignes.append(f"\n{col} :\n")
        for modalite, effectif in freq.items():
            lignes.append(f"  {modalite} : {effectif} ({pct[modalite]}%)\n")

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        freq.sort_values().plot(kind="barh", ax=ax, color="#5a8f5a")
        ax.set_title(f"Repartition de {col}")
        fig.savefig(DOSSIER_FIG / f"{col}.png", dpi=110)
        plt.close(fig)

    chemin_txt = RACINE / "outputs" / "analyse_univariee.txt"
    with open(chemin_txt, "w", encoding="utf-8") as f:
        f.writelines(l if l.endswith("\n") else l + "\n" for l in lignes)
    print(f"analyse univariee ecrite -> {chemin_txt}")
    print(f"figures -> {DOSSIER_FIG}")


if __name__ == "__main__":
    analyser()
