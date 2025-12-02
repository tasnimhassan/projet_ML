from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(x_train, y_train):
    tree = DecisionTreeClassifier(max_depth=15)
    tree.fit(x_train, y_train)
    return tree
