# Projet analyse de donnees + IA generative - EcoPrime Banque

Projet realise en binome pour le module Analyse de Donnees (bloc 2, RNCP
Expert en ingenierie des donnees). Le rapport complet est dans
`rapport.md`.

## Contexte

On a choisi un cas de churn bancaire (probabilite de resiliation d'un
client) plutot que le jeu de donnees fourni par l'ecole, faute d'y avoir eu
acces au moment de demarrer. Le jeu de donnees utilise ici est simule
(voir `src/generation_donnees.py`) pour representer un portefeuille client
d'une banque de detail fictive, "EcoPrime Banque", avec des relations
volontaires entre certaines variables et la resiliation.

## Structure

```
data/                    jeux de donnees (bruts, propres, cotations)
src/                     scripts python, a lancer dans l'ordre ci-dessous
notebooks/               version notebook du projet (analyse_ecoprime.ipynb)
outputs/                 figures + resultats texte/JSON generes par les scripts
rapport.md               rapport complet du projet
```

Le notebook `notebooks/analyse_ecoprime.ipynb` reprend les memes etapes que
les scripts, dans l'ordre, avec les graphiques affiches directement dans les
cellules. C'est la version a montrer en demo / soutenance ; les scripts dans
`src/` restent la version "production" utilisee pour generer les fichiers
dans `outputs/` et pour le dashboard.

## Comment relancer le projet

```
pip install -r requirements.txt

python src/generation_donnees.py      # genere data/clients_ecoprime.csv
python src/prepa_donnees.py           # nettoyage -> data/clients_ecoprime_propre.csv
python src/analyse_univariee.py
python src/analyse_bivariee.py
python src/acp.py
python src/modelisation.py
python src/partie2_marches.py         # necessite une connexion internet (yfinance)
python src/partie3_ia_generative.py

streamlit run src/dashboard.py        # tableau de bord interactif
```

Chaque script relit les fichiers produits par le precedent, donc l'ordre
compte (surtout generation -> preparation -> le reste).
