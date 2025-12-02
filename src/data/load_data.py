import pandas as pd

def load_train_test(train_path, test_path):
    df_train = pd.read_csv(train_path, encoding="utf-8")
    df_test = pd.read_csv(test_path, encoding="utf-8")
    return df_train, df_test
