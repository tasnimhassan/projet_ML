import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def eval_model(model, x_test, y_test, title=""):
    predictions = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 25},
                cmap=sns.color_palette(['#f8eded', '#b84747'], as_cmap=True),
                cbar=False, yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)

    ax.set_title('Confusion Matrix ' + title, size=15, pad=20)
    ax.set_xlabel('Predicted Values', size=18)
    ax.set_ylabel('Actual Values', size=18)

    # Tes métriques EXACTES
    print("Accuracy:", metrics.accuracy_score(y_test, predictions))
    print("Precision:", metrics.precision_score(y_test, predictions))
    print("Recall:", metrics.recall_score(y_test, predictions))
    print("Specificity:", metrics.recall_score(y_test, predictions, pos_label=0))
    print("F1 Score:", metrics.f1_score(y_test, predictions))
