from sklearn.tree import DecisionTreeClassifier
import pickle

def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier(max_depth=15)
    model.fit(x_train, y_train)

    # Sauvegarde pour Streamlit 
    pickle.dump(model, open("models/tree_model.pkl", "wb"))

    return model
