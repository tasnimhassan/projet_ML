Guide d'utilisation :
URL du Dépôt : https://github.com/tasnimhassan/projet_ML.git

Branche de travail principale : main

Étape 1 : Préparation et Clonage du Dépôt

-Ouvrir le Terminal : Ouvrez votre terminal .
-Cloner le code : Naviguez vers le répertoire où vous souhaitez stocker le projet, puis exécutez la commande git clone :
-git clone https://github.com/tasnimhassan/projet_ML.git
Accéder au dossier : Entrez dans le nouveau dossier du projet :

Etape 2:

tester le projet avec :
python main.py ou py main.py 

Ensuite Dans ce fichiers vous allez voir les images et les graphique du code enregistrés :
nsl-kdd-project\reports\figures


1. Introduction du Projet 

1.1 Contexte

La sécurité des réseaux est essentielle aujourd'hui. Les attaques informatiques deviennent de plus en plus complexes. Compter sur l'humain pour les détecter n'est plus suffisant. C'est pourquoi nous utilisons les Systèmes de Détection d'Intrusion basés sur le Machine Learning (ML). Ces systèmes permettent d'identifier automatiquement les comportements suspects sur un réseau.

1.2 Notre Objectif

L'objectif de ce projet est d'entraîner et de comparer plusieurs modèles de Machine Learning. Le but est de créer un outil pour détecter automatiquement les intrusions sur un réseau informatique. Nous utilisons le célèbre jeu de données NSL-KDD pour cet entraînement.

1.3 Architecture du Pipeline

Préparation des données (nettoyage).
Création de la variable cible (ce qu'on cherche à prédire).
Sélection et entraînement des modèles ML.
Évaluation de la performance des modèles.
Sauvegarde des meilleurs modèles et création des graphiques.

1.4 Le Dataset NSL-KDD
C'est une version améliorée et plus équilibrée d'un ancien dataset . Il contient :
41 caractéristiques qui décrivent chaque connexion réseau.
Une étiquette qui dit si c'est normal ou si c'est une attaque.
Quatre types d'attaques (DOS, Probe, R2L, U2R).
j'ai donner egalement des titres aux colonnes car ils y'avaient pas.

1.5 Structure du projet

nsl-kdd-project/

├ data/
│   ├ raw/                      données brutes
│   │   ├ NSL_KDD_Train.csv
│   │   └ NSL_KDD_Test.csv
│   ├ processed/                 données nettoyées

 

├ src/
│   ├ data/
│   │   └ load_data.py       chargement des données
│   ├ preprocessing/
│   │   └ preprocess.py      
│   ├ models/
│   │   ├ train_logreg.py
│   │   ├ train_tree.py
│   │   └ train_knn.py       entraînement des modèles
    ├ evaluation/
    │   └ evaluate.py        métriques et scores
│   └ visualization/
│       └ plots.py           graphiques et matrices de confusion
│
├ models/                    modèles sauvegardés 
│
├ reports/
│   ├ figures/               images, graphiques
│   └ rapport.pdf            rapport 
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


2. Problématique ? 

2.1 Le Défi Principal
La question centrale est simple : Comment réussir à détecter efficacement une attaque réseau en partant de données brutes ?
2.2 Limites des Anciens Systèmes
Les systèmes de sécurité plus anciens ont des problèmes :
Ils utilisent des règles fixes qui doivent être mises à jour souvent.
Ils ont du mal à s'adapter aux nouvelles menaces.
Ils ne peuvent pas identifier des attaques jamais vues auparavant.
2.3 L'Avantage du Machine Learning
Le Machine Learning résout ce problème. Il apprend tout seul les schémas anormaux en se basant sur les données historiques du réseau.

3. Préparation et Méthodologie 

3.1 Les Étapes de Préparation

La méthode de travail est structurée :
Nettoyage : Charger les données, renommer les colonnes, supprimer les doublons.
Encodage : Transformer les variables non numériques (catégorielles, comme le protocole) en nombres.
Cible Binaire : Créer la variable à prédire : 0 pour Normal, 1 pour Attaque.
Normalisation : Mettre les données à la même échelle (avec StandardScaler).
Séparation : Diviser les données en un ensemble d'entraînement (80%) et un ensemble de test (20%).

3.2 Outils

j'ai utilisé le langage Python avec la librairie scikit-learn pour tous les algorithmes de ML.
Les graphiques ont été faits avec Matplotlib et Seaborn.
L'organisation du projet est modulaire  pour un code propre.

4. Modèles et Résultats 
4.1 Les Modèles Testés
j'ai choisi trois algorithmes :
Régression Logistique : Notre modèle de base (baseline), simple et rapide.
Arbre de Décision (Decision Tree) : Un modèle qui est facile à interpréter.
KNN (K plus proches voisins) : Un algorithme très connu pour être performant sur ce type de données.
4.2 Les Résultats 
Nous avons utilisé des mesures importantes (Accuracy, Precision, Recall, F1-Score) pour bien évaluer les modèles, car les données sont déséquilibrées.



5. Conclusion 
Ce projet m'a permis de construire et de mettre en œuvre un processus complet de Machine Learning pour la détection d'intrusions. Le dataset NSL-KDD a été analysé et préparé.
Le modèle arbre de Décision (Decision Tree) est le modèle le plus efficace pour cette tâche de classification. il montre qu'il est excellent pour détecter les attaques réseau.
Ce travail m'a permis d'acquérir de solides compétences dans les domaines suivants :
Préparation de données complexes.
Organisation d'un projet de Data Science structuré.
Test et évaluation d'algorithmes de Machine Learning.




AUTEUR : 
Projet réalisé par : TESLINE HASSAN OKIEH 
MASTER INFORMATIQUE  BIG DATA

