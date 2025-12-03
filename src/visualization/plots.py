import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_confusion_matrix(cm, title, filename):
    # créer le dossier si nécessaire
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap="Reds",
                xticklabels=['0', '1'],
                yticklabels=['0', '1'],
                ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # chemin final
    filepath = os.path.join(output_dir, filename + ".png")

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()   # pour éviter les conflits

    print(f"Graphique sauvegardé : {filepath}")
