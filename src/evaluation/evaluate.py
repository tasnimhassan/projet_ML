from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, x_test, y_test):
    preds = model.predict(x_test)
    cm = confusion_matrix(y_test, preds)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "specificity": recall_score(y_test, preds, pos_label=0, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "confusion_matrix": cm
    }
