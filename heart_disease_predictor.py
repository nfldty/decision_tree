import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/heart.csv')


# print(df.head())
# print(df.columns)
# print(df.dtypes)

q1, q2, q3 = df.age.quantile([0.25, 0.5, 0.75])
df['age_q1'] = [1 if x < q1 else 0 for x in df.age]
df['age_q2'] = [1 if x < q2 and x <= q1 else 0 for x in df.age]
df['age_q3'] = [1 if x < q3 and x <= q2 else 0 for x in df.age]
df['age_q4'] = [1 if x >= q3 else 0 for x in df.age]

df = pd.get_dummies(df, columns=['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True, dtype='int')

q1, q2, q3 = df.trestbps.quantile([0.25, 0.5, 0.75])
df['trestbps_q1'] = [1 if x < q1 else 0 for x in df.trestbps]
df['trestbps_q2'] = [1 if x < q2 and x <= q1 else 0 for x in df.trestbps]
df['trestbps_q3'] = [1 if x < q3 and x <= q2 else 0 for x in df.trestbps]
df['trestbps_q4'] = [1 if x >= q3 else 0 for x in df.trestbps]

q1, q2, q3 = df.thalach.quantile([0.25, 0.5, 0.75])
df['thalach_q1'] = [1 if x < q1 else 0 for x in df.thalach]
df['thalach_q2'] = [1 if x < q2 and x <= q1 else 0 for x in df.thalach]
df['thalach_q3'] = [1 if x < q3 and x <= q2 else 0 for x in df.thalach]
df['thalach_q4'] = [1 if x >= q3 else 0 for x in df.thalach]

q1, q2, q3 = df.oldpeak.quantile([0.25, 0.5, 0.75])
df['oldpeak_q1'] = [1 if x < q1 else 0 for x in df.oldpeak]

df['oldpeak_q2'] = [1 if x < q2 and x <= q1 else 0 for x in df.oldpeak]
df['oldpeak_q3'] = [1 if x < q3 and x <= q2 else 0 for x in df.oldpeak]
df['oldpeak_q4'] = [1 if x >= q3 else 0 for x in df.oldpeak]

df.drop(columns=['oldpeak', 'chol', 'thalach', 'trestbps', 'age'], inplace=True)
# print(df.shape)
# print(df.head())
# print(df.columns)
# print(df.dtypes)
#
# print(df.sex.value_counts())

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['target'], test_size=0.3 ,random_state=10)

model = DecisionTree()
model.fit(X_train, y_train)
print(X_train.head())
print(y_train.head())
print(model.accuracy(X_test, y_test))