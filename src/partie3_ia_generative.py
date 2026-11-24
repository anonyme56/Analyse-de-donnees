"""
Partie 3 (facultative) - prototype IA generative.

Deux cas d'usage :
1) message personnalise pour un client identifie comme "a risque" (churn
   probable) par le modele de la partie 1
2) mini resume financier pour un conseiller, a partir des chiffres du client

Choix assume : pas d'appel a une API de LLM externe ici (pas de cle
disponible / question de cout et de confidentialite des donnees clients
simulees quand meme). On construit un generateur a base de gabarits +
regles, qui reste un prototype demontrable, et on note dans le rapport a
quel endroit on brancherait un vrai modele de langage (ex: appel a une API
de type OpenAI/Mistral avec le meme prompt, en remplacement de la fonction
`generer_message`).

L'evaluation reste volontairement simple : longueur du message, presence
des elements attendus (nom, raison du risque, action proposee), et un score
de lisibilite approxime (longueur moyenne de phrase).
"""

import json
import pandas as pd
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DF = pd.read_csv(RACINE / "data" / "clients_ecoprime_propre.csv")

RAISONS_RISQUE = {
    "reclamations": "plusieurs reclamations recentes",
    "inactivite": "une baisse d'activite sur les derniers mois",
    "anciennete": "un client encore recent, periode ou le risque de depart est plus eleve",
}


def choisir_raison(client):
    if client["nb_reclamations_12mois"] >= 2:
        return RAISONS_RISQUE["reclamations"]
    if client["jours_depuis_derniere_connexion"] >= 30:
        return RAISONS_RISQUE["inactivite"]
    return RAISONS_RISQUE["anciennete"]


def generer_message(client):
    raison = choisir_raison(client)
    civilite = "Madame" if client["sexe"] == "F" else "Monsieur"
    if client["nombre_produits"] <= 1:
        action = "un point telephonique pour revoir son offre et lui presenter les produits complementaires adaptes"
    else:
        action = "un rendez-vous en agence pour faire le point sur sa satisfaction et ses besoins actuels"

    message = (
        f"{civilite}, client depuis {client['anciennete_mois']} mois (compte {client['type_compte'].lower()}), "
        f"presente un risque de resiliation lie a {raison}. "
        f"Action recommandee : proposer {action}. "
        f"Canal de contact prefere : {client['canal_contact_prefere']}."
    )
    return message


def generer_resume_financier(client):
    resume = (
        f"Client {client['id_client']} : anciennete {client['anciennete_mois']} mois, "
        f"{client['nombre_produits']} produit(s), solde moyen {client['solde_moyen_compte']:.0f} EUR, "
        f"salaire annuel estime {client['salaire_annuel_net']:.0f} EUR, "
        f"score de satisfaction {client['score_satisfaction']}/5, "
        f"{client['nb_reclamations_12mois']} reclamation(s) sur 12 mois."
    )
    return resume


def evaluer_message(message, elements_attendus):
    nb_mots = len(message.split())
    nb_phrases = message.count(".") + message.count("!") + 1
    longueur_moyenne_phrase = nb_mots / nb_phrases
    couverture = sum(1 for e in elements_attendus if e.lower() in message.lower()) / len(elements_attendus)
    return {
        "nb_mots": nb_mots,
        "longueur_moyenne_phrase": round(longueur_moyenne_phrase, 1),
        "couverture_elements_attendus": round(couverture, 2),
    }


def demo():
    # on prend les 5 clients avec le plus de reclamations comme cas "a risque"
    clients_a_risque = DF.sort_values("nb_reclamations_12mois", ascending=False).head(5)

    resultats = []
    for _, client in clients_a_risque.iterrows():
        message = generer_message(client)
        resume = generer_resume_financier(client)
        evaluation = evaluer_message(
            message,
            elements_attendus=["risque", "action", "canal", str(client["id_client"])[:3]],
        )
        resultats.append({
            "id_client": int(client["id_client"]),
            "message_client": message,
            "resume_conseiller": resume,
            "evaluation": evaluation,
        })

    with open(RACINE / "outputs" / "partie3_exemples_genai.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print(f"{len(resultats)} exemples generes -> outputs/partie3_exemples_genai.json")
    for r in resultats[:2]:
        print("-", r["message_client"])


if __name__ == "__main__":
    demo()
