from sklearn.neighbors import KNeighborsClassifier

def train_knn(x_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    return knn
