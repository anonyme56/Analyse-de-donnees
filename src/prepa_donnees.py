"""
Etape 1 - preparation des donnees (typage + qualite).

- typage informatique (dtype pandas) et typage statistique (quanti continue /
  discrete, quali nominale / ordinale)
- reperage des valeurs manquantes
- reperage des valeurs aberrantes (methode IQR pour les variables continues,
  bornes metier pour l'age et le solde)
- export d'un csv "propre" utilise par les etapes suivantes + un fichier
  texte qui recapitule ce qui a ete trouve / corrige
"""

import pandas as pd
import numpy as np
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_BRUT = RACINE / "data" / "clients_ecoprime.csv"
CHEMIN_PROPRE = RACINE / "data" / "clients_ecoprime_propre.csv"
CHEMIN_RAPPORT = RACINE / "outputs" / "qualite_donnees.txt"

# typage statistique declare a la main (utile pour les etapes suivantes)
TYPAGE_STATISTIQUE = {
    "id_client": "identifiant",
    "age": "quantitative discrete",
    "sexe": "qualitative nominale",
    "situation_familiale": "qualitative nominale",
    "region": "qualitative nominale",
    "anciennete_mois": "quantitative discrete",
    "type_compte": "qualitative nominale",
    "nombre_produits": "quantitative discrete",
    "carte_credit": "qualitative nominale (binaire)",
    "membre_actif": "qualitative nominale (binaire)",
    "score_credit": "quantitative continue",
    "salaire_annuel_net": "quantitative continue",
    "solde_moyen_compte": "quantitative continue",
    "montant_moyen_transaction": "quantitative continue",
    "nb_transactions_mois": "quantitative discrete",
    "utilise_appli_mobile": "qualitative nominale (binaire)",
    "nb_connexions_appli_mois": "quantitative discrete",
    "jours_depuis_derniere_connexion": "quantitative discrete",
    "nb_reclamations_12mois": "quantitative discrete",
    "score_satisfaction": "qualitative ordinale (1 a 5)",
    "canal_contact_prefere": "qualitative nominale",
    "date_dernier_contact": "date",
    "churn": "qualitative nominale (binaire, cible)",
}

COLONNES_CONTINUES = [
    "score_credit", "salaire_annuel_net", "solde_moyen_compte",
    "montant_moyen_transaction",
]


def bornes_iqr(serie, k=3.0):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def preparer():
    df = pd.read_csv(CHEMIN_BRUT, parse_dates=["date_dernier_contact"])

    lignes_rapport = []
    lignes_rapport.append(f"Nombre de lignes / colonnes de depart : {df.shape}\n")

    # 1. typage informatique explicite
    df["sexe"] = df["sexe"].astype("category")
    df["situation_familiale"] = df["situation_familiale"].astype("category")
    df["region"] = df["region"].astype("category")
    df["type_compte"] = df["type_compte"].astype("category")
    df["canal_contact_prefere"] = df["canal_contact_prefere"].astype("category")
    for c in ["carte_credit", "membre_actif", "utilise_appli_mobile", "churn"]:
        df[c] = df[c].map({"Oui": 1, "Non": 0}).astype("Int64")

    # 2. valeurs manquantes
    manquants = df.isna().sum()
    manquants = manquants[manquants > 0]
    lignes_rapport.append("Valeurs manquantes par colonne :\n" + manquants.to_string() + "\n")

    # traitement : imputation simple (mediane / mode), documentee
    df["score_credit"] = df["score_credit"].fillna(df["score_credit"].median())
    df["salaire_annuel_net"] = df["salaire_annuel_net"].fillna(df["salaire_annuel_net"].median())
    df["score_satisfaction"] = df["score_satisfaction"].fillna(df["score_satisfaction"].median())
    df["canal_contact_prefere"] = df["canal_contact_prefere"].cat.add_categories(["Inconnu"]).fillna("Inconnu")
    df["situation_familiale"] = df["situation_familiale"].cat.add_categories(["Inconnu"]).fillna("Inconnu")

    # 3. valeurs aberrantes
    lignes_rapport.append("Valeurs aberrantes detectees :\n")

    nb_age_ab = ((df["age"] < 18) | (df["age"] > 100)).sum()
    lignes_rapport.append(f"- age hors [18, 100] : {nb_age_ab} lignes -> plafonnees a 100\n")
    df["age"] = df["age"].clip(upper=100)

    for col in COLONNES_CONTINUES:
        basse, haute = bornes_iqr(df[col])
        nb_hors = ((df[col] < basse) | (df[col] > haute)).sum()
        lignes_rapport.append(f"- {col} hors IQR*3 [{basse:.1f}, {haute:.1f}] : {nb_hors} lignes -> plafonnees\n")
        df[col] = df[col].clip(lower=basse, upper=haute)

    lignes_rapport.append("\nTypage statistique retenu :\n")
    for col, typ in TYPAGE_STATISTIQUE.items():
        lignes_rapport.append(f"- {col} : {typ}\n")

    CHEMIN_RAPPORT.parent.mkdir(exist_ok=True)
    with open(CHEMIN_RAPPORT, "w", encoding="utf-8") as f:
        f.writelines(lignes_rapport)

    df.to_csv(CHEMIN_PROPRE, index=False, encoding="utf-8")
    print(f"donnees preparees -> {CHEMIN_PROPRE}")
    print(f"rapport qualite -> {CHEMIN_RAPPORT}")
    return df


if __name__ == "__main__":
    preparer()
