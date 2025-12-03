import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def rename_columns(df):
    columns = ([
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root',
        'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
        'is_host_login','is_guest_login','count','srv_count','serror_rate',
        'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
        'diff_srv_rate','srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','attack_features'
    ])
    df.columns = columns
    return df

def encode_categorical(df):
    LEnc = LabelEncoder()
    df['protocol_type'] = LEnc.fit_transform(df['protocol_type'])
    df['service'] = LEnc.fit_transform(df['service'])
    df['flag'] = LEnc.fit_transform(df['flag'])
    return df

def create_target(df):
    df['Target'] = df.attack_features.map(lambda a: 0 if a == 'normal' else 1)
    return df

def split_data(df):
    x = df.drop(['Target', 'attack_features'], axis=1)
    y = df['Target']
    return train_test_split(x, y, test_size=0.25, random_state=42)

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.fit_transform(x_test)
import pandas as pd

def clean_data(df):
    # 1) Renommer les colonnes automatiquement
    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # 2) Supprimer les lignes dupliquées
    df = df.drop_duplicates()

    # 3) Supprimer / remplir les valeurs manquantes
    df = df.fillna(0)

    # 4) Convertir les colonnes numériques quand c'est possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    return df
