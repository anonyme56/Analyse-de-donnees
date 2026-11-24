"""
Genere le jeu de donnees clients pour EcoPrime Banque (banque fictive).

Contexte : on n'a pas eu acces au jeu de donnees interne fourni par l'ecole
(pas telecharge / pas retenu), donc on simule un portefeuille clients plausible
pour une banque de detail, avec des correlations volontaires entre certaines
variables et la resiliation (churn), pour pouvoir ensuite tester les etapes
d'analyse et de modelisation dessus. Les valeurs manquantes et aberrantes sont
injectees a la fin pour se rapprocher d'un vrai jeu de donnees "sale".
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 1742
N = 2200

DOSSIER_DATA = Path(__file__).resolve().parent.parent / "data"
DOSSIER_DATA.mkdir(exist_ok=True)


def generer():
    rng = np.random.default_rng(SEED)

    id_client = np.arange(100000, 100000 + N)

    age = rng.normal(42, 13, N).clip(18, 88).round().astype(int)

    sexe = rng.choice(["F", "M"], size=N, p=[0.49, 0.51])

    situation_familiale = rng.choice(
        ["Celibataire", "Marie", "Divorce", "Veuf"], size=N, p=[0.38, 0.42, 0.15, 0.05]
    )

    regions = [
        "Ile-de-France", "Auvergne-Rhone-Alpes", "Nouvelle-Aquitaine",
        "Occitanie", "Hauts-de-France", "Grand Est", "Bretagne",
        "Pays de la Loire", "PACA", "Normandie",
    ]
    poids_regions = [0.22, 0.13, 0.09, 0.09, 0.09, 0.08, 0.07, 0.08, 0.09, 0.06]
    region = rng.choice(regions, size=N, p=poids_regions)

    anciennete_mois = rng.gamma(shape=2.2, scale=28, size=N).clip(1, 420).round().astype(int)

    type_compte = rng.choice(
        ["Courant", "Premium", "Professionnel"], size=N, p=[0.68, 0.22, 0.10]
    )

    nombre_produits = rng.choice([1, 2, 3, 4], size=N, p=[0.32, 0.38, 0.22, 0.08])

    carte_credit = rng.choice(["Oui", "Non"], size=N, p=[0.71, 0.29])

    membre_actif = rng.choice(["Oui", "Non"], size=N, p=[0.63, 0.37])

    # score credit correle un peu avec l'age et l'anciennete
    score_credit = (
        520
        + 0.9 * age
        + 0.35 * anciennete_mois
        + rng.normal(0, 55, N)
    ).clip(300, 850).round().astype(int)

    salaire_annuel_net = (
        18000
        + 380 * age
        + rng.normal(0, 9000, N)
        + np.where(type_compte == "Professionnel", 9000, 0)
        + np.where(type_compte == "Premium", 14000, 0)
    ).clip(9500, 260000).round(2)

    solde_moyen_compte = (
        salaire_annuel_net * rng.uniform(0.02, 0.22, N)
        - rng.normal(300, 900, N)
    ).round(2)

    montant_moyen_transaction = (30 + salaire_annuel_net / 2200 + rng.normal(0, 18, N)).clip(5, None).round(2)

    nb_transactions_mois = rng.poisson(lam=14, size=N).clip(0, 90)

    utilise_appli_mobile = rng.choice(["Oui", "Non"], size=N, p=[0.74, 0.26])

    nb_connexions_appli_mois = np.where(
        utilise_appli_mobile == "Oui",
        rng.poisson(lam=9, size=N),
        rng.poisson(lam=0.6, size=N),
    )

    jours_depuis_derniere_connexion = rng.exponential(scale=18, size=N).clip(0, 400).round().astype(int)

    nb_reclamations_12mois = rng.poisson(lam=0.9, size=N).clip(0, 12)

    # satisfaction depend (mollement) des reclamations
    satisfaction_brute = 4.2 - 0.55 * nb_reclamations_12mois + rng.normal(0, 0.9, N)
    score_satisfaction = satisfaction_brute.clip(1, 5).round().astype(int)

    canal_contact_prefere = rng.choice(
        ["Email", "SMS", "Appel", "Agence"], size=N, p=[0.40, 0.24, 0.16, 0.20]
    )

    date_dernier_contact = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 640, N), unit="D"
    )

    # --- construction du churn (variable cible) ---
    def normaliser(x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-9)

    logit = (
        -1.35
        + 0.85 * normaliser(nb_reclamations_12mois)
        - 0.75 * normaliser(score_satisfaction)
        - 0.55 * normaliser(anciennete_mois)
        - 0.40 * (membre_actif == "Non").astype(float)
        - 0.30 * normaliser(nombre_produits)
        + 0.45 * normaliser(jours_depuis_derniere_connexion)
        - 0.20 * (utilise_appli_mobile == "Oui").astype(float)
        + rng.normal(0, 0.55, N)
    )
    proba_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, N) < proba_churn).astype(int)
    churn_label = np.where(churn == 1, "Oui", "Non")

    df = pd.DataFrame({
        "id_client": id_client,
        "age": age,
        "sexe": sexe,
        "situation_familiale": situation_familiale,
        "region": region,
        "anciennete_mois": anciennete_mois,
        "type_compte": type_compte,
        "nombre_produits": nombre_produits,
        "carte_credit": carte_credit,
        "membre_actif": membre_actif,
        "score_credit": score_credit,
        "salaire_annuel_net": salaire_annuel_net,
        "solde_moyen_compte": solde_moyen_compte,
        "montant_moyen_transaction": montant_moyen_transaction,
        "nb_transactions_mois": nb_transactions_mois,
        "utilise_appli_mobile": utilise_appli_mobile,
        "nb_connexions_appli_mois": nb_connexions_appli_mois,
        "jours_depuis_derniere_connexion": jours_depuis_derniere_connexion,
        "nb_reclamations_12mois": nb_reclamations_12mois,
        "score_satisfaction": score_satisfaction,
        "canal_contact_prefere": canal_contact_prefere,
        "date_dernier_contact": date_dernier_contact,
        "churn": churn_label,
    })

    # --- valeurs manquantes injectees sur quelques colonnes, taux variables ---
    for col, taux in [
        ("score_credit", 0.03),
        ("salaire_annuel_net", 0.025),
        ("score_satisfaction", 0.015),
        ("canal_contact_prefere", 0.02),
        ("situation_familiale", 0.01),
    ]:
        idx_nan = rng.choice(N, size=int(N * taux), replace=False)
        df.loc[idx_nan, col] = np.nan

    # --- quelques valeurs aberrantes volontaires (erreurs de saisie plausibles) ---
    idx_aberrant = rng.choice(N, size=12, replace=False)
    df["age"] = df["age"].astype("int64")
    df.loc[idx_aberrant[:4], "age"] = rng.integers(120, 145, 4)  # age impossible
    df.loc[idx_aberrant[4:8], "salaire_annuel_net"] = rng.integers(1_000_000, 3_000_000, 4)
    df.loc[idx_aberrant[8:], "solde_moyen_compte"] = -rng.integers(50000, 90000, 4)

    return df


if __name__ == "__main__":
    donnees = generer()
    chemin = DOSSIER_DATA / "clients_ecoprime.csv"
    donnees.to_csv(chemin, index=False, encoding="utf-8")
    print(f"{len(donnees)} clients generes -> {chemin}")
    print(f"taux de churn brut : {(donnees['churn'] == 'Oui').mean():.3f}")
