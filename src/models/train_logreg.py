from sklearn.linear_model import LogisticRegression
import pickle

def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Sauvegarde du modèle pour Streamlit
    pickle.dump(model, open("models/logreg_model.pkl", "wb"))

    return model
