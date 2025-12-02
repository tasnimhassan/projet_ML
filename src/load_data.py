import pandas as pd

def load_data():
    df_train = pd.read_csv('data/raw/NSL_KDD_Train.csv', encoding="utf-8")
    df_test = pd.read_csv('data/raw/NSL_KDD_Test.csv', encoding="utf-8")

    print("Train data loaded:", df_train.shape)
    print("Test data loaded:", df_test.shape)

    return df_train, df_test
