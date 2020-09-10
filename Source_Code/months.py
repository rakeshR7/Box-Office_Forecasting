import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
df = pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/IAC4-data_train.xlsx')

df1 = df[['release_date','Profit Class']]
df1['month'] = pd.DatetimeIndex(df1['release_date']).month
print(df1.head(20))
months = pd.get_dummies(df1['month'])
print(months.head(20))
months = months.join(df['Box Office After Inflation'])
months.rename(columns={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12:'Dec'}, inplace=True)
Jan = months.loc[months.Jan == 1]['Box Office After Inflation']
Feb = months.loc[months.Feb == 1]['Box Office After Inflation']
Mar = months.loc[months.Mar == 1]['Box Office After Inflation']
Apr = months.loc[months.Apr == 1]['Box Office After Inflation']
May = months.loc[months.May == 1]['Box Office After Inflation']
Jun = months.loc[months.Jun == 1]['Box Office After Inflation']
Jul = months.loc[months.Jul == 1]['Box Office After Inflation']
Aug = months.loc[months.Aug == 1]['Box Office After Inflation']
Sep = months.loc[months.Sep == 1]['Box Office After Inflation']
Oct = months.loc[months.Oct == 1]['Box Office After Inflation']
Nov = months.loc[months.Nov == 1]['Box Office After Inflation']
Dec = months.loc[months.Dec == 1]['Box Office After Inflation']

Month_df = pd.DataFrame(dict(Jan=Jan, Feb=Feb, Mar=Mar,Apr=Apr, May=May, Jun=Jun, Jul=Jul, Aug=Aug, Sep=Sep, Oct=Oct, Nov=Nov, Dec=Dec))
Month_mean_box_office = Month_df.mean()
Month_mean_box_office.plot(kind='barh', title = 'Mean_Box_Office_by_Month', logx=False)
plt.show()
