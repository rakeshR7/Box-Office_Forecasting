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

df1 = df[['keywords','Profit Class']]
keywords = df1['keywords'].str.get_dummies(sep = '|')
keywords = keywords.join(df['Box Office After Inflation'])
#keywords.rename(columns={'tent-pole':'tent_pole'}, inplace=True)
filename = '/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/Keywords.txt'
file = open(filename, "w")
#for column in keywords.columns:
#    file.write(column)
#    file.write('\n')
#file.close()
#print('tent_pole' in keywords.columns)
remake = keywords.loc[keywords.remake == 1]['Box Office After Inflation']
#tent_pole = keywords.loc[keywords.tent_pole == 1]['Box Office After Inflation']
sequel = keywords.loc[keywords.sequel == 1]['Box Office After Inflation']

keywords_df = pd.DataFrame(dict(Remake=remake, Sequel=sequel))#Remake=remake,  Sequel=sequel))#Tent_Pole=tent_pole,
keywords_box_office = keywords_df.mean()
print(keywords_box_office)
keywords_box_office.plot(kind='barh', title = 'Mean_Box_Office_by_keywords', logx=False)
plt.show()
