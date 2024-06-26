# -*- coding: utf-8 -*-
"""Aktuelle Version von ADASYN & XGBoostMsgRank .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/101zulTuDW8sZY_RSEq5ZNhJBpoKYpEXb
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imblearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
from sklearn.metrics import accuracy_score, classification_report

path="/content/drive/MyDrive/MachineB_with_Job.csv"
df=pd.read_csv(path,sep=";")
df.head(20)
df.info()

df=df[['MsgRank','CheckIn']]
df.info()

#convert CheckIn to datetime |old version
##df['CheckIn']=pd.to_datetime(df['CheckIn'],format='%Y-%m-%d %H:%M:%S.%f')
#df.info()

#convert CheckIn and CheckOut to datetime
date_format = "%Y-%m-%d %H:%M:%S%z"
df['CheckIn'] = df['CheckIn'].str.replace('\.\d+', '', regex=True)
df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)

#Extracting features from CheckIn
df['CheckInHour'] = df['CheckIn'].dt.hour
df['CheckInMinute'] = df['CheckIn'].dt.minute
df['CheckInSecond'] = df['CheckIn'].dt.second
df['CheckInMicrosecond'] = df['CheckIn'].dt.microsecond
df['CheckInYear'] = df['CheckIn'].dt.year
df['CheckInMonth'] = df['CheckIn'].dt.month
df['CheckInDay'] = df['CheckIn'].dt.day

#drop MsgRank smaller 0
df = df.drop(df.index[df['MsgRank']<0])

#use label encoder
le = LabelEncoder()
df['MsgRank_encoded']= le.fit_transform(df['MsgRank'])

#assign features and target
X=df[['CheckInDay', 'CheckInMonth', 'CheckInYear', 'CheckInHour', 'CheckInMinute', 'CheckInSecond', 'CheckInMicrosecond']]
y=df['MsgRank_encoded']

"""*Sampling*

Adaptive Synthetic Sampling
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN

#count occurences of MsgRank
label_counts = df['MsgRank'].value_counts()
print("Label Occurrences:")
print(label_counts)

#train test split
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.3,shuffle=False,random_state=80)

# Apply ADASYN to oversample the minority classes
adasyn = ADASYN(random_state=42)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)

"""Create XGBoost Model-Classification"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

model=XGBClassifier()
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

#inverse label encoding
y_test_rev= le.inverse_transform(y_test)
y_pred_rev=le.inverse_transform(y_pred)

#print classification report
from sklearn.metrics import classification_report

print(classification_report(y_test_rev, y_pred_rev))

#visualize results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:10], label='Actual', marker='o')
plt.plot(y_pred[:10], label='Predicted', marker='o')

plt.title('Actual vs. Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.show()

#calculating how often the prediction was correct with a threshold from 100
threshold= 1.0
absolute_difference =np.abs(y_test.values -y_pred)
correct_predictions = np.sum(absolute_difference <= threshold)

accuracy = correct_predictions / len(y_test)

print(f'Correct Predictions: {correct_predictions}/{len(y_test)}')
print(f'Accuracy: {accuracy * 100:.2f}%')
