"""
Tableau de bord interactif (Streamlit) - EcoPrime Banque.

Lancer avec : streamlit run src/dashboard.py

Trois profils d'utilisateurs simules via des onglets : conseiller (vue
client par client), manager (vue portefeuille / KPI), et un onglet marches
financiers (partie 2). L'idee n'est pas de faire un outil "parfait" mais de
montrer qu'on sait brancher les resultats des etapes precedentes sur une
interface filtrable.
"""

import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DF = pd.read_csv(RACINE / "data" / "clients_ecoprime_propre.csv", parse_dates=["date_dernier_contact"])

st.set_page_config(page_title="EcoPrime Banque - tableau de bord", layout="wide")
st.title("EcoPrime Banque - suivi resiliation clients")

onglet_manager, onglet_conseiller, onglet_marches = st.tabs(
    ["Vue manager", "Vue conseiller", "Marches financiers"]
)

with onglet_manager:
    st.subheader("Vue d'ensemble du portefeuille")

    col_region, col_type = st.columns(2)
    regions = ["Toutes"] + sorted(DF["region"].unique().tolist())
    region_choisie = col_region.selectbox("Region", regions)
    types_compte = ["Tous"] + sorted(DF["type_compte"].unique().tolist())
    type_choisi = col_type.selectbox("Type de compte", types_compte)

    df_filtre = DF.copy()
    if region_choisie != "Toutes":
        df_filtre = df_filtre[df_filtre["region"] == region_choisie]
    if type_choisi != "Tous":
        df_filtre = df_filtre[df_filtre["type_compte"] == type_choisi]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clients", len(df_filtre))
    k2.metric("Taux de resiliation", f"{df_filtre['churn'].mean()*100:.1f}%")
    k3.metric("Anciennete moyenne (mois)", f"{df_filtre['anciennete_mois'].mean():.0f}")
    k4.metric("Reclamations moy. / client", f"{df_filtre['nb_reclamations_12mois'].mean():.2f}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df_filtre.groupby("region", observed=True)["churn"].mean().sort_values().plot(kind="barh", ax=ax, color="#3b6ea5")
        ax.set_xlabel("taux de resiliation")
        st.pyplot(fig)
    with col_g2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df_filtre["nb_reclamations_12mois"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#5a8f5a")
        ax.set_xlabel("nb reclamations (12 mois)")
        st.pyplot(fig)

    with st.expander("Resultats des modeles predictifs"):
        chemin_json = RACINE / "outputs" / "resultats_modeles.json"
        if chemin_json.exists():
            st.json(json.loads(chemin_json.read_text(encoding="utf-8")))
        else:
            st.write("Lancer src/modelisation.py pour generer les resultats.")

with onglet_conseiller:
    st.subheader("Recherche client")
    id_saisi = st.number_input("Identifiant client", min_value=int(DF["id_client"].min()), max_value=int(DF["id_client"].max()), value=int(DF["id_client"].iloc[0]))
    ligne = DF[DF["id_client"] == id_saisi]
    if ligne.empty:
        st.warning("Identifiant introuvable.")
    else:
        c = ligne.iloc[0]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Anciennete (mois)", int(c["anciennete_mois"]))
        col_b.metric("Score satisfaction", f"{c['score_satisfaction']}/5" if pd.notna(c["score_satisfaction"]) else "n.c.")
        col_c.metric("Reclamations (12 mois)", int(c["nb_reclamations_12mois"]))
        st.write(f"Statut resiliation : {'Oui' if c['churn'] == 1 else 'Non'}")
        st.write(f"Canal prefere : {c['canal_contact_prefere']} - type de compte : {c['type_compte']}")

        chemin_genai = RACINE / "outputs" / "partie3_exemples_genai.json"
        if chemin_genai.exists():
            exemples = json.loads(chemin_genai.read_text(encoding="utf-8"))
            correspondance = [e for e in exemples if e["id_client"] == id_saisi]
            if correspondance:
                st.info(correspondance[0]["message_client"])

with onglet_marches:
    st.subheader("Tesla / Stellantis")
    chemin_cotations = RACINE / "data" / "cotations_tesla_stellantis.csv"
    if chemin_cotations.exists():
        cotes = pd.read_csv(chemin_cotations, index_col="date", parse_dates=True)
        st.line_chart(cotes)
        st.caption("Cours de cloture ajuste (source Yahoo Finance, via yfinance).")
    else:
        st.write("Lancer src/partie2_marches.py pour recuperer les cotations.")
