import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score
import pickle
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import confusion_matrix

X_train = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/x_train.xlsx')
y_train = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/y_classification_train.xlsx')
#y_train = (y_train['Profit Class'] > 1).astype(int)
X_test =pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/x_test.xlsx')
y_test = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/y_classification_test.xlsx')
#y_test = (y_test['Profit Class'] > 1).astype(int)

model = svm.SVC()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print(confusion_matrix(y_test, prediction))
