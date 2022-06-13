import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

df = pd.read_csv('C:/Users/JEON_SANGEON/codestates/project/toy/project_land/files/land.csv')

df_ = df.copy()
df_ = df_[df_['면적(㎡)']!=0].reset_index(drop=True)
df_m = df_.loc[df_['계약구분']=='매매']

feature = ['면적(㎡)']
target = ['거래금액(만원)']
X = df_m[feature]
y = df_m[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

salary_model = LinearRegression()
salary_model.fit(X_train, y_train)

y_pred = salary_model.predict(X_test)

# pickle.dump(salary_model, open('model/salary_model.pkl','wb'))

import joblib

joblib.dump(salary_model, "salary_model.pkl")