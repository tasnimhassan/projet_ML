import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("🔍 Détection d’attaques réseau (NSL-KDD)")
st.write("Interface simple permettant de visualiser les performances du modèle et comprendre les résultats.")

# -----------------------------------------------------
# Charger le modèle
# -----------------------------------------------------
try:
    model = pickle.load(open("models/logreg_model.pkl", "rb"))
    st.success("Modèle chargé avec succès.")
except:
    st.error("Erreur : fichier models/logreg_model.pkl introuvable.")
    st.stop()

# -----------------------------------------------------
# Explication simple
# -----------------------------------------------------
st.header("🧠 C’est quoi une attaque ?")
st.write("""
Chaque connexion réseau peut être :
- **Normale** → pas de danger  
- **Attaque** → tentative de piratage, scan, déni de service (DoS), etc.

Le modèle apprend à différencier **normal** vs **attaque** à partir de 41 caractéristiques du dataset NSL-KDD.
""")

# -----------------------------------------------------
# Affichage des performances
# -----------------------------------------------------
st.header("📊 Performances du modèle")

st.write("""
Voici les performances obtenues pendant l'évaluation du modèle sur les données test.
Ces mesures permettent de comprendre à quel point le modèle détecte correctement les attaques.
""")

# Valeurs d'exemple (tirées de ton main.py)
accuracy = 0.95
precision = 0.96
recall = 0.94
f1 = 0.95

st.metric("Accuracy", f"{accuracy*100:.2f}%")
st.metric("Precision", f"{precision*100:.2f}%")
st.metric("Recall", f"{recall*100:.2f}%")
st.metric("F1-score", f"{f1*100:.2f}%")

# -----------------------------------------------------
# Matrice de confusion
# -----------------------------------------------------
st.header("🧩 Matrice de confusion")

try:
    # Exemple de matrice (tu peux charger celle générée)
    cm = np.array([[16183, 591], [864, 13856]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.xlabel("Prédiction")
    plt.ylabel("Réel")
    st.pyplot(fig)

except:
    st.warning("Impossible d'afficher la matrice de confusion.")

# -----------------------------------------------------
# Liste des features (explication simplifiée)
# -----------------------------------------------------
st.header("📘 Comprendre les caractéristiques (features)")

st.write("""
Le modèle utilise **41 informations** à propos de chaque connexion, par exemple :

- `duration` → durée de la connexion  
- `protocol_type` → protocole utilisé (TCP, UDP, ICMP)  
- `service` → type de service (http, ftp, smtp…)  
- `src_bytes` → bytes envoyés par la source  
- `dst_bytes` → bytes reçus  
- `count` → nombre de connexions similaires  
- `srv_count` → nombre de connexions vers le même service  
- etc.

L’utilisateur **n’a pas besoin de connaître tout ça** pour comprendre si une attaque est détectée.
""")

st.success("Interface minimaliste prête ✨")
