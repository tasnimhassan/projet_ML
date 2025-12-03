import os
os.makedirs("figures", exist_ok=True)

from src.data.load_data import load_train_test
from src.preprocessing.preprocess import (
    encode_categorical, create_target,
    split_data, scale_data
)
from src.models.train_logreg import train_logistic_regression
from src.models.train_tree import train_decision_tree
from src.models.train_knn import train_knn
from src.evaluation.evaluate import evaluate_model
from src.visualization.plots import plot_confusion_matrix

# Import des fonctions EDA
from src.visualization.eda import (
    basic_info,
    value_counts_info,
    plot_pie_target,
    plot_dst_bytes_by_target,
    plot_correlation_heatmap
)

print("Loading data...")

# Charger les données
df_train, df_test = load_train_test(
    "data/raw/NSL_KDD_Train.csv",
    "data/raw/NSL_KDD_Test.csv"
)

# Préparation du dataset (seulement df_train)
df = encode_categorical(df_train)
df = create_target(df)

# EDA
basic_info(df)
value_counts_info(df)
plot_pie_target(df)
plot_dst_bytes_by_target(df)
plot_correlation_heatmap(df)

# Split et normalisation
x_train, x_test, y_train, y_test = split_data(df)
x_train, x_test = scale_data(x_train, x_test)

print("\nTraining models...")

# Entraînement des modèles
logreg = train_logistic_regression(x_train, y_train)
dtree = train_decision_tree(x_train, y_train)
knn = train_knn(x_train, y_train)

print("\nEvaluating...")

# Évaluation
res_log = evaluate_model(logreg, x_test, y_test)
res_tree = evaluate_model(dtree, x_test, y_test)
res_knn = evaluate_model(knn, x_test, y_test)

print("\n  Logistic Regression  ")
print(res_log)

print("\n  Decision Tree  ")
print(res_tree)

print("\n  KNN  ")
print(res_knn)

# Matrices de confusion
plot_confusion_matrix(res_log["confusion_matrix"], "Logistic Regression", "logreg_matrix")
plot_confusion_matrix(res_tree["confusion_matrix"], "Decision Tree", "tree_matrix")
plot_confusion_matrix(res_knn["confusion_matrix"], "KNN", "knn_matrix")

# Sauvegarde du modèle pour Streamlit
import pickle
pickle.dump(logreg, open("models/logreg_model.pkl", "wb"))
print("\nModèle sauvegardé dans models/logreg_model.pkl")
