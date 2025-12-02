import numpy
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess():
    df = pd.read_csv('data/raw/NSL_KDD_Train.csv', encoding="utf-8")
    df = pd.read_csv('data/raw/NSL_KDD_Test.csv', encoding="utf-8")

    print("Train data loaded:", df.shape) 
    print("Test data loaded:", df.shape)

    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
                'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
                'root_shell','su_attempted','num_root','num_file_creations','num_shells',
                'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
                'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
                'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
                'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
                'attack_features'])

    df.columns = columns

    object_features = df.select_dtypes(include=['object'])
    object_cols = object_features.columns
    numeric_features = df.select_dtypes(include=['int64', 'float64'])
    numeric_col = numeric_features.columns
    print('Number of Numeric Features: ', len(numeric_col))
    print('Number of Object Features: ', len(object_cols))

    data_attack = df.attack_features.map(lambda a: 0 if a == 'normal' else 1)
    train, test_df = train_test_split(df, test_size=0.2 , random_state=0)
    df['Target'] = data_attack

    LEnc = LabelEncoder()
    df['protocol_type'] = LEnc.fit_transform(df['protocol_type'])
    df['service'] = LEnc.fit_transform(df['service'])
    df['flag'] = LEnc.fit_transform(df['flag'])

    x = df.drop(['Target','attack_features'], axis=1)
    y = df['Target'].copy()

    x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.25, random_state=42)

    Normalize = StandardScaler()
    x_train = Normalize.fit_transform(x_train)
    x_test = Normalize.fit_transform(x_test)

    return x_train, x_test, y_train, y_test, df
