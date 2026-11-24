"""
Partie 2 - extraction et visualisation de donnees financieres.

On recupere les cotations de Tesla (TSLA) et Stellantis (STLA) sur Yahoo
Finance (yfinance), on calcule quelques indicateurs (rendement journalier,
volatilite glissante, performance cumulee), et on compare a titre illustratif
l'evolution mensuelle du taux de resiliation interne avec la volatilite des
marches sur la meme periode. Le but n'est pas de prouver un lien de cause a
effet (rien ne dit qu'un client de banque suit la bourse Tesla...) mais de
montrer comment on articule une source interne et une source externe dans un
meme pipeline, ce qui est demande par le sujet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE / "data"
DOSSIER_FIG = RACINE / "outputs" / "figures" / "marches"
DOSSIER_FIG.mkdir(parents=True, exist_ok=True)

TICKERS = {"TSLA": "Tesla", "STLA": "Stellantis"}
PERIODE = "2y"


def telecharger_cotations():
    cotations = {}
    for ticker, nom in TICKERS.items():
        d = yf.download(ticker, period=PERIODE, auto_adjust=True, progress=False)
        cloture = d["Close"].squeeze()
        cloture.name = nom
        cotations[nom] = cloture
    df = pd.concat(cotations.values(), axis=1)
    df.columns = list(cotations.keys())
    df.index.name = "date"
    df.to_csv(DOSSIER_DATA / "cotations_tesla_stellantis.csv")
    return df


def indicateurs(cotes):
    rendements = cotes.pct_change().dropna()
    volatilite_20j = rendements.rolling(20).std() * np.sqrt(252)
    perf_cumulee = (1 + rendements).cumprod() - 1
    return rendements, volatilite_20j, perf_cumulee


def taux_churn_mensuel():
    df = pd.read_csv(DOSSIER_DATA / "clients_ecoprime_propre.csv", parse_dates=["date_dernier_contact"])
    df["mois"] = df["date_dernier_contact"].dt.to_period("M").dt.to_timestamp()
    taux = df.groupby("mois")["churn"].mean().rename("taux_churn_mensuel")
    return taux


def construire():
    cotes = telecharger_cotations()
    rendements, volatilite, perf_cumulee = indicateurs(cotes)
    taux_churn = taux_churn_mensuel()

    # graphe 1 : performance cumulee des deux titres
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in perf_cumulee.columns:
        ax.plot(perf_cumulee.index, perf_cumulee[col] * 100, label=col)
    ax.set_ylabel("performance cumulee (%)")
    ax.set_title(f"Performance cumulee Tesla / Stellantis sur {PERIODE}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "performance_cumulee.png", dpi=110)
    plt.close(fig)

    # graphe 2 : volatilite glissante
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for col in volatilite.columns:
        ax.plot(volatilite.index, volatilite[col], label=col)
    ax.set_ylabel("volatilite annualisee (fenetre 20j)")
    ax.set_title("Volatilite glissante Tesla / Stellantis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "volatilite_glissante.png", dpi=110)
    plt.close(fig)

    # graphe 3 : mise en regard avec le taux de churn interne mensuel
    volat_mensuelle = volatilite.resample("MS").mean()
    jointure = volat_mensuelle.join(taux_churn, how="inner")

    fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
    ax1.plot(jointure.index, jointure["taux_churn_mensuel"] * 100, color="#c0392b", marker="o", label="taux de churn interne (%)")
    ax1.set_ylabel("taux de churn interne (%)", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(jointure.index, jointure["Tesla"], color="#3b6ea5", label="volatilite Tesla")
    ax2.plot(jointure.index, jointure["Stellantis"], color="#5a8f5a", label="volatilite Stellantis")
    ax2.set_ylabel("volatilite annualisee")
    ax1.set_title("Taux de churn interne vs volatilite des marches (echelle double)")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIG / "churn_vs_volatilite.png", dpi=110)
    plt.close(fig)

    correlation = jointure.corr(numeric_only=True)["taux_churn_mensuel"].drop("taux_churn_mensuel")

    with open(RACINE / "outputs" / "partie2_resultats.txt", "w", encoding="utf-8") as f:
        f.write("Rendement journalier moyen et volatilite annualisee moyenne :\n")
        for col in cotes.columns:
            f.write(
                f"  {col} : rendement moyen/jour = {rendements[col].mean()*100:.3f}%, "
                f"volatilite annualisee moyenne = {volatilite[col].mean():.3f}\n"
            )
        f.write(f"\nPerformance cumulee sur la periode ({PERIODE}) :\n")
        for col in cotes.columns:
            f.write(f"  {col} : {perf_cumulee[col].iloc[-1]*100:.1f}%\n")
        f.write("\nCorrelation (mensuelle) entre taux de churn interne et volatilite des marches :\n")
        for nom, val in correlation.items():
            f.write(f"  {nom} : r = {val:.3f}\n")
        f.write(
            "\nNote : correlation calculee sur peu de points mensuels, a interpreter avec prudence, "
            "elle sert surtout a illustrer la demarche d'articulation donnees internes / externes.\n"
        )

    print("Partie 2 terminee, resultats dans outputs/partie2_resultats.txt")


if __name__ == "__main__":
    construire()
