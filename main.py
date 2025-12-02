from src.preprocessing import preprocess
from src.models import logistic_regression, decision_tree, knn_model
from src.evaluation import eval_model

x_train, x_test, y_train, y_test, df = preprocess()

logr = logistic_regression(x_train, y_train)
eval_model(logr, x_test, y_test, "Logistic Regression")

dt = decision_tree(x_train, y_train)
eval_model(dt, x_test, y_test, "Decision Tree")

kn = knn_model(x_train, y_train)
eval_model(kn, x_test, y_test, "KNN")
df = pd.read_csv("data/raw/NSL_KDD_Train.csv")
