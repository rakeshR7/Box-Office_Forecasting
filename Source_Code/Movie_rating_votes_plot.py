#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

df=pd.read_excel('/home/oncopeltus/python/DataScienceWorkshop/IAC_4.0/IAC4-data_train.xlsx')
#df.rename(columns={'movie_rating':'movie_rating1'}, inplace=True)


# In[14]:


#get_ipython().run_line_magic('matplotlib', 'inline')
df.plot(kind='scatter',x='Profit Class',y='movie_rating_score')
plt.ylabel('Movie Rating')
plt.xlabel('Profit Class')
plt.show()


# In[ ]:
#get_ipython().run_line_magic('matplotlib', 'inline')
df.plot(kind='scatter',x='Profit Class',y='movie_votes')
plt.ylabel('Movie votes')
plt.xlabel('Profit Class')
plt.show()

df.plot(kind='scatter',y='Box Office After Inflation',x='movie_votes')
plt.xlabel('Movie votes')
plt.ylabel('box office')
plt.show()

df.plot(kind='scatter',y='Box Office After Inflation',x='movie_rating_score')
plt.ylabel('box office')
plt.xlabel('movie rating')
plt.show()

# In[ ]:
