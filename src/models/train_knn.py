from sklearn.neighbors import KNeighborsClassifier
import pickle

def train_knn(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

  
    pickle.dump(model, open("models/knn_model.pkl", "wb"))

    return model
