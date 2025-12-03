import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------
# Encodage des colonnes catégorielles : protocol, service, flag
# ---------------------------------------------------------
def encode_categorical(df):
    enc_protocol = LabelEncoder()
    enc_service = LabelEncoder()
    enc_flag = LabelEncoder()

    df['protocol_type'] = enc_protocol.fit_transform(df['protocol_type'])
    df['service'] = enc_service.fit_transform(df['service'])
    df['flag'] = enc_flag.fit_transform(df['flag'])

    return df

# ---------------------------------------------------------
# Création de la cible (0 = normal, 1 = attaque)
# ---------------------------------------------------------
def create_target(df):
    df['Target'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    return df

# ---------------------------------------------------------
# Split train/test
# ---------------------------------------------------------
def split_data(df):
    x = df.drop(['Target', 'label'], axis=1)
    y = df['Target']
    return train_test_split(x, y, test_size=0.25, random_state=42)

# ---------------------------------------------------------
# Normalisation
# ---------------------------------------------------------
def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # IMPORTANT : pas fit_transform
    return x_train, x_test
