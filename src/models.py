from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def logistic_regression(x_train, y_train):
    logr = linear_model.LogisticRegression(max_iter=1500)
    logr.fit(x_train,y_train)
    return logr

def decision_tree(x_train, y_train):
    dtree = DecisionTreeClassifier(max_depth=15)
    dtree.fit(x_train,y_train)
    return dtree

def knn_model(x_train, y_train):
    kcl = KNeighborsClassifier(n_neighbors=5)
    kcl.fit(x_train,y_train)
    return kcl
