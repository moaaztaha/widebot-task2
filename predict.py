# imports
import pandas as pd
import numpy as np
from pickle import load

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# getting the data
valid_ds = pd.read_csv('validation.csv', delimiter=';', decimal=',')
one_row = valid_ds.sample(1)

# preprocessing
preprocessing_pipeline = load(open('preprocess_pipeline.pkl', 'rb'))
labelencoder = load(open('labelencoder.pkl', 'rb'))

y = one_row.classLabel

# drop variable18
X = one_row.drop('classLabel', axis=1)
X.drop('variable18', axis=1, inplace=True)

X_pro = preprocessing_pipeline.transform(X)
y_pro  = labelencoder.transform(y)
print('Actual Class:', y_pro)


# the model
knn_model = load(open('rf_all.pkl', 'rb'))
# making prediction
pred = knn_model.predict(X_pro)
print('Prediction:', pred)

