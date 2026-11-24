"""
Etape 3 - analyse bivariee.

- correlations Pearson (lineaire) et Spearman (monotone) entre variables
  quantitatives, avec heatmap
- test du Chi2 d'independance entre variables qualitatives et le churn
- t-test / ANOVA pour comparer une variable quantitative selon le churn
  (ou une autre variable qualitative a plusieurs modalites)
- quelques nuages de points et diagrammes croises
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DF = pd.read_csv(RACINE / "data" / "clients_ecoprime_propre.csv")
DOSSIER_FIG = RACINE / "outputs" / "figures" / "bivarie"
DOSSIER_FIG.mkdir(parents=True, exist_ok=True)

VAR_QUANTI = [
    "age", "anciennete_mois", "nombre_produits", "score_credit",
    "salaire_annuel_net", "solde_moyen_compte", "montant_moyen_transaction",
    "nb_transactions_mois", "nb_connexions_appli_mois",
    "jours_depuis_derniere_connexion", "nb_reclamations_12mois",
]
VAR_QUALI = [
    "sexe", "situation_familiale", "region", "type_compte",
    "carte_credit", "membre_actif", "utilise_appli_mobile",
    "canal_contact_prefere",
]


def analyser():
    lignes = []

    # --- correlations entre quantitatives ---
    corr_pearson = DF[VAR_QUANTI].corr(method="pearson")
    corr_spearman = DF[VAR_QUANTI].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_pearson, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlations de Pearson entre variables quantitatives")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "heatmap_pearson.png", dpi=110)
    plt.close(fig)

    # paires les plus correlees (hors diagonale)
    paires = corr_pearson.where(~np.eye(len(VAR_QUANTI), dtype=bool)).abs().unstack().sort_values(ascending=False)
    paires = paires[~paires.duplicated()].head(6)
    lignes.append("Paires de variables quantitatives les plus correlees (Pearson, valeur absolue) :\n")
    for (a, b), v in paires.items():
        lignes.append(f"  {a} / {b} : r = {corr_pearson.loc[a, b]:.3f}\n")

    # --- chi2 entre qualitatives et churn ---
    lignes.append("\nTests du Chi2 (variable qualitative vs churn) :\n")
    for col in VAR_QUALI:
        table = pd.crosstab(DF[col], DF["churn"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        signif = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        lignes.append(f"  {col} : chi2={chi2:.2f}, ddl={dof}, p={p:.4f} -> {signif}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    pd.crosstab(DF["membre_actif"], DF["churn"], normalize="index").plot(kind="bar", stacked=True, ax=axes[0])
    axes[0].set_title("Churn selon membre_actif")
    pd.crosstab(DF["type_compte"], DF["churn"], normalize="index").plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Churn selon type_compte")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "churn_vs_qualitatives.png", dpi=110)
    plt.close(fig)

    # --- t-test / ANOVA quantitatif vs churn ---
    lignes.append("\nComparaison de moyennes (t-test) quantitatif vs churn :\n")
    for col in VAR_QUANTI:
        grp_oui = DF.loc[DF["churn"] == 1, col].dropna()
        grp_non = DF.loc[DF["churn"] == 0, col].dropna()
        t, p = stats.ttest_ind(grp_oui, grp_non, equal_var=False)
        signif = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        lignes.append(
            f"  {col} : moy(churn=Oui)={grp_oui.mean():.2f}, moy(churn=Non)={grp_non.mean():.2f}, "
            f"t={t:.2f}, p={p:.4f} -> {signif}\n"
        )

    # ANOVA nb_reclamations selon region (juste pour illustrer un cas multi-groupes)
    groupes = [g["nb_reclamations_12mois"].values for _, g in DF.groupby("region", observed=True)]
    f_stat, p_anova = stats.f_oneway(*groupes)
    lignes.append(
        f"\nANOVA nb_reclamations_12mois selon region : F={f_stat:.2f}, p={p_anova:.4f}\n"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(DF["nb_reclamations_12mois"], DF["score_satisfaction"], alpha=0.25, s=14)
    ax.set_xlabel("nb_reclamations_12mois")
    ax.set_ylabel("score_satisfaction")
    ax.set_title("Reclamations vs satisfaction")
    fig.savefig(DOSSIER_FIG / "nuage_reclamations_satisfaction.png", dpi=110)
    plt.close(fig)

    chemin_txt = RACINE / "outputs" / "analyse_bivariee.txt"
    with open(chemin_txt, "w", encoding="utf-8") as f:
        f.writelines(lignes)
    print(f"analyse bivariee ecrite -> {chemin_txt}")


if __name__ == "__main__":
    analyser()
