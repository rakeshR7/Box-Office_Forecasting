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

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix

X_train = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/x_train.xlsx')
y_train = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/y_classification_train.xlsx')
#y_train = (y_train['Profit Class'] > 1).astype(int)
X_test =pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/x_test.xlsx')
y_test = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/y_classification_test.xlsx')
#y_test = (y_test['Profit Class'] > 1).astype(int)
xg_model = xgb.XGBClassifier(booster='gblinear',
n_estimators=1000,
gamma=1,
max_depth=4,
min_child_weight=6,
objective='multi:softmax',
num_class=4,
scale_pos_weight=1,
random_state=27, learning_rate=0.2, max_delta_step=5,subsample=0.33, early_stopping_rounds=10)
# xg_model =xgb.XGBClassifier(booster='gblinear', colsample_bylevel=0.8, colsample_bytree=1,
#        gamma=1, learning_rate=0.2, max_delta_step=5,
#        max_depth=3, min_child_weight=10, missing=None,
#        n_estimators=300, nthread=-1, objective='binary:logitraw',
#        scale_pos_weight=1, seed=1, silent=True,
#        subsample=0.33, early_stopping_rounds=10)
xg_model.fit(X_train, y_train)
xg_prediction = xg_model.predict(X_test)
#xgb.plot_importance(xg_model,max_num_features=20)
accuracy = accuracy_score(y_test, xg_prediction)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(confusion_matrix(y_test, xg_prediction))

X_train['production_log'] = X_train['production'].apply(np.log).astype(np.float)
X_train['actors_log'] = X_train['actors'].apply(np.log).astype(np.float)
X_train['directors_log'] = X_train['director'].apply(np.log).astype(np.float)
X_train['budget_log'] = X_train['Budget After Inflation'].apply(np.log).astype(np.float)
X_train['movie_votes_log'] = X_train['movie_votes'].apply(np.log).astype(np.float)
X_train.drop(['production','actors','director','movie_votes','Budget After Inflation'], axis=1, inplace=True)
# # X_train=X_train.astype(float)
# # y_train=y_train.astype(float)
#
#
X_test['production_log'] = X_test['production'].apply(np.log).astype(np.float)
X_test['actors_log'] = X_test['actors'].apply(np.log).astype(np.float)
X_test['directors_log'] = X_test['director'].apply(np.log).astype(np.float)
X_test['budget_log'] = X_test['Budget After Inflation'].apply(np.log).astype(np.float)
X_test['movie_votes_log'] = X_test['movie_votes'].apply(np.log).astype(np.float)
X_test.drop(['production','actors','director','movie_votes','Budget After Inflation'], axis=1, inplace=True)
# X_test=X_test.astype(float)
# y_test=y_test.astype(float)


#print(prediction)
#plt.show()
#X_train = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/x_train.xlsx')
y_train = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/y_regression_train.xlsx')#
y_train = y_train['Box Office After Inflation'].apply(np.log).astype(np.float)
#y_train = (y_train_df['Profit Class'] > 1).astype(int)
#X_test = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/x_test.xlsx')
y_test = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/y_regression_test.xlsx')#['Box Office After Inflation']
y_test = y_test['Box Office After Inflation'].apply(np.log).astype(np.float)
#y_test = (y_test_df['Profit Class'] > 1).astype(int)
model =xgb.XGBRegressor(booster='gbtree',
                    objective= 'survival:cox',
                    eval_metric='rmse',
                    gamma = 1,
                    min_child_weight= 3,
                    max_depth= 5,
                    subsample= 0.33,
                    colsample_bytree= 0.7,
                    learning_rate=0.01,
                    n_estimators=1000,
                    nthread=4,
                    scale_pos_weight=1,
                    reg_alpha=100,
                    seed=1850,
                    early_stopping_rounds=10, max_delta_step=5)
model.fit(X_train, y_train)
print(model.score(X_train,y_train))

prediction = model.predict(X_test)
print(model.score(X_test,y_test))
print(mean_squared_error(y_test, prediction))



# svm_model = svm.SVC()
# svm_model.fit(X_train, y_train)
# svm_prediction = svm_model.predict(X_test)
#
# confusion_matrix(y_test, prediction)
