import numpy
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix




df= pd.read_csv('/kaggle/input/data-test/NSL_KDD_Test.csv', encoding="utf-8")


df = pd.read_csv('/kaggle/input/data-train/NSL_KDD_Train.csv', encoding="utf-8")


print("Train data loaded:", df.shape) 
print("Test data loaded:", df.shape)


df


print(df)


df.head()


columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins'
            ,'logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells'
,'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate'
            ,'rerror_rate','srv_rerror_rate'
,'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate'
            ,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
,'attack_features'])

df.columns = columns
df.head()


object_features = df.select_dtypes(include=['object'])
object_cols = object_features.columns
numeric_features = df.select_dtypes(include=['int64', 'float64'])
numeric_col = numeric_features.columns
print('Number of Numeric Features: ', len(numeric_col))
print('Number of Object Features: ', len(object_cols))


df.dtypes


data_attack = df.attack_features.map(lambda a: 0 if a == 'normal' else 1)


train, test_df = train_test_split(df, test_size= 0.2 , random_state= 0 )

df['Target'] = data_attack



df.head()


df.value_counts()


df['Target'].head()


df['Target'].value_counts()


df.dtypes


attack_vs_features = pd.crosstab(df.attack_features, df.Target)
attack_vs_features


df.drop_duplicates(keep=False,inplace=True)


df.duplicated().any()


labels = ["Normal",'Attack']
sizes = [dict(df.Target.value_counts())[0], dict(df.Target.value_counts())[1]]
plt.figure(figsize = (10,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend(["Normal", "Attack"])
plt.title('The percentage of Normal and Attack Requests in dataset')
plt.show()


Targett_dict = dict(df.value_counts())
sns.countplot(df.all())


Target_dict = dict(df.Target.value_counts())
sns.countplot(df.Target) 




print(df["protocol_type"].value_counts())
print(df["service"].value_counts())
print(df["flag"].value_counts())


LEnc = LabelEncoder()
df['protocol_type'] = LEnc.fit_transform(df['protocol_type'])
df['service'] = LEnc.fit_transform(df['service'])
df['flag'] = LEnc.fit_transform(df['flag'])
df


print(df["protocol_type"].value_counts())


df['protocol_type']


df['flag']


df['service']



x = df.drop(['Target','attack_features'], axis=1)
y = df['Target'].copy()
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.25, random_state=42)

Normalize = StandardScaler()
x_train = Normalize.fit_transform(x_train)
x_test = Normalize.fit_transform(x_test)


y.count()


x_train


x_train.shape


y_train.shape


x_test.shape


y_test.shape


#LogisticRegression

logr = linear_model.LogisticRegression(max_iter=1500)
logr.fit(x_train,y_train)

logr.score(x_train,y_train)


predictions = logr.predict(x_test)
predictions




cm =confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 25},
            cmap=sns.color_palette(['#f8eded', '#b84747'], as_cmap=True), cbar=False,
            yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)

ax.set_title('Confusion Matrix', size=15, pad=20)
ax.set_xlabel('Predicted Values', size=18)
ax.set_ylabel('Actual Values', size=18)

additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
for text_elt, additional_text in zip(ax.texts, additional_texts):
    ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=13)


Accuracy2 = metrics.accuracy_score(y_test, predictions)
Accuracy2


Precision2 = metrics.precision_score(y_test, predictions)
Precision2


Sensitivity_recall2 = metrics.recall_score(y_test, predictions)
Sensitivity_recall2


Specificity = metrics.recall_score(y_test, predictions, pos_label=0)
Specificity


F1_score1 = metrics.f1_score(y_test, predictions)
F1_score1


#arbre de decision

dtree = DecisionTreeClassifier(max_depth=15)
dtree.fit(x_train,y_train)

predictions1 = dtree.predict(x_test)


cm =confusion_matrix(y_test, predictions1)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 25},
            cmap=sns.color_palette(['#f8eded', '#b84747'], as_cmap=True), cbar=False,
            yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)

ax.set_title('Confusion Matrix', size=15, pad=20)
ax.set_xlabel('Predicted Values', size=18)
ax.set_ylabel('Actual Values', size=18)

additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
for text_elt, additional_text in zip(ax.texts, additional_texts):
    ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=13)
    
dtree.score(x_train,y_train)


Accuracy = metrics.accuracy_score(y_test, predictions1)
Accuracy


Precision1 = metrics.precision_score(y_test, predictions1)
Precision1


Sensitivity_recall1 = metrics.recall_score(y_test, predictions1)
Sensitivity_recall1


Specificity = metrics.recall_score(y_test, predictions1, pos_label=0)
Specificity


F1_score = metrics.f1_score(y_test, predictions1)
F1_score 


# KNeighborsClassifier(KNN)


kcl = KNeighborsClassifier(n_neighbors=5)
kcl.fit(x_train,y_train)
predictions2 = kcl.predict(x_test)


cm =confusion_matrix(y_test, predictions2)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 25},
            cmap=sns.color_palette(['#f8eded', '#b84747'], as_cmap=True), cbar=False,
            yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)

ax.set_title('Confusion Matrix', size=15, pad=20)
ax.set_xlabel('Predicted Values', size=18)
ax.set_ylabel('Actual Values', size=18)

additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
for text_elt, additional_text in zip(ax.texts, additional_texts):
    ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=13)
kcl.score(x_train,y_train)    


Accuracy = metrics.accuracy_score(y_test, predictions2)
Accuracy


Precision = metrics.precision_score(y_test, predictions2)
Precision


Sensitivity_recall = metrics.recall_score(y_test, predictions2)
Sensitivity_recall


Specificity = metrics.recall_score(y_test, predictions2, pos_label=0)
Specificity


f1_score = metrics.f1_score(y_test, predictions2)
f1_score