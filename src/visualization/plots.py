import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 25},
                cmap=sns.color_palette(['#f8eded', '#b84747'], as_cmap=True),
                cbar=False, yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.show()
