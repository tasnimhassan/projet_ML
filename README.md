Projet de Machine Learning
L’objectif est de détecter automatiquement les attaques réseau (intrusions) à partir du dataset NSL-KDD, en entraînant plusieurs modèles de Machine Learning et en comparant leurs performances.

Le projet est inspirée de la structure Cookiecutter Data Science.

Structure du projet

nsl-kdd-project/
│
├ data/
│   ├ raw/                      données brutes
│   │   ├ NSL_KDD_Train.csv
│   │   └ NSL_KDD_Test.csv
│   ├ processed/                 données nettoyées
│
├ notebooks/
│   ├ 01_exploration.ipynb      analyse exploratoire du dataset
│   └ 02_modelisation.ipynb     essais et modèles
│
├ src/
│   ├ data/
│   │   └ load_data.py       chargement des données
│   ├ preprocessing/
│   │   └ preprocess.py      renommage, encodage, splits, scaling
│   ├ models/
│   │   ├ train_logreg.py
│   │   ├ train_tree.py
│   │   └ train_knn.py       entraînement des modèles
│   ├ evaluation/
│   │   └ evaluate.py        métriques et scores
│   └ visualization/
│       └ plots.py           graphiques et matrices de confusion
│
├ models/                    modèles sauvegardés 
│
├ reports/
│   ├ figures/               images, graphiques
│   └ rapport.pdf            rapport final
│
├ main.py                    pipeline complet et exécutable
├ requirements.txt           dépendances
└ README.md                  documentation du projet

Installer les dépendances:
pip install -r requirements.txt


Placer les fichiers de données
Déposer les fichiers dans:
data/raw/NSL_KDD_Train.csv  
data/raw/NSL_KDD_Test.csv

Pour lancer tout le pipeline:
python main.py

AUTEUR : 
Projet réalisé par : SAWERA AKHTAR & TESLINE HASSAN OKIEH
MASTER INFORMATIQUE  BIG DATA