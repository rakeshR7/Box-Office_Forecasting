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

df1 = df[['genre','Profit Class']]
#print(df1.info())
genre = df1['genre'].str.get_dummies(sep = ', ')
X = genre
#df1.join(gerne)
genre = genre.join(df1['Profit Class'])
#genre = genre.join(df['Box Office After Inflation'])
genre = genre.join(df['Box Office After Inflation'])
genre.rename(columns={'Sci-Fi':'Sci_Fi'}, inplace=True)

print(genre.info())
Action = genre.loc[genre.Action == 1]['Box Office After Inflation']
#print(Action.head())
Adventure = genre.loc[genre.Adventure == 1]['Box Office After Inflation']
Animation = genre.loc[genre.Animation == 1]['Box Office After Inflation']
Biography = genre.loc[genre.Biography == 1]['Box Office After Inflation']
Comedy = genre.loc[genre.Comedy == 1]['Box Office After Inflation']
Crime = genre.loc[genre.Crime == 1]['Box Office After Inflation']
Documentary = genre.loc[genre.Documentary == 1]['Box Office After Inflation']
Drama = genre.loc[genre.Drama == 1]['Box Office After Inflation']
Family = genre.loc[genre.Family == 1]['Box Office After Inflation']
Fantasy = genre.loc[genre.Fantasy == 1]['Box Office After Inflation']
History = genre.loc[genre.History == 1]['Box Office After Inflation']
Horror = genre.loc[genre.Horror == 1]['Box Office After Inflation']
Music = genre.loc[genre.Music == 1]['Box Office After Inflation']
Musical = genre.loc[genre.Musical == 1]['Box Office After Inflation']
News = genre.loc[genre.News == 1]['Box Office After Inflation']
Romance = genre.loc[genre.Romance == 1]['Box Office After Inflation']
Sci_Fi = genre.loc[genre.Sci_Fi == 1]['Box Office After Inflation']
Short = genre.loc[genre.Short == 1]['Box Office After Inflation']
Sport = genre.loc[genre.Sport == 1]['Box Office After Inflation']
Thriller = genre.loc[genre.Thriller == 1]['Box Office After Inflation']
War = genre.loc[genre.War == 1]['Box Office After Inflation']
Western = genre.loc[genre.Western == 1]['Box Office After Inflation']

Genre_df = pd.DataFrame(dict(Action=Action, Adventure=Adventure, Animation=Animation,Biography=Biography, Comedy=Comedy, Crime=Crime, Documentary=Documentary, Drama=Drama,
Family=Family, Fantasy=Fantasy, History=History, Horror=Horror, Music=Music, Musical=Musical, News=News, Romance=Romance, Sci_Fi=Sci_Fi, Short=Short, Sport=Sport, Thriller=Thriller, War=War, Western=Western))#.reset_index()
#Genre_df.drop('index', axis = 1, inplace=True)
print(Genre_df.mean())
Genre_mean_box_office = Genre_df.mean()
Genre_mean_box_office_log = np.log(Genre_mean_box_office)
Genre_std_box_office = Genre_df.std()
Genre_mean_box_office.plot(kind='barh', title = 'Mean_Box_Office_by_Genre', logx=False)
plt.show()

df2=df[['movie_rating','Profit Class']]
#print(df2.info())
df2.fillna('UNRATED')
df2.replace('NOT RATED', 'UNRATED',inplace=True)
df2.replace('NR','UNRATED',inplace=True)
df2.replace('TV-PG', 'PG',inplace=True)
df2.replace('NC-17', 'NC_17',inplace=True)
df2.replace('PG-13', 'PG_13',inplace=True)

movie_rating = df2['movie_rating'].str.get_dummies(sep = ', ')
#print(movie_rating)
X = X.join(movie_rating)
movie_rating = movie_rating.join(df1['Profit Class'])

movie_rating = movie_rating.join(df['Box Office After Inflation'])

G = movie_rating.loc[movie_rating.G == 1]['Box Office After Inflation']
PG = movie_rating.loc[movie_rating.PG == 1]['Box Office After Inflation']
PG_13 = movie_rating.loc[movie_rating.PG_13 == 1]['Box Office After Inflation']
R = movie_rating.loc[movie_rating.R == 1]['Box Office After Inflation']
NC_17 = movie_rating.loc[movie_rating.NC_17 == 1]['Box Office After Inflation']
UNRATED = movie_rating.loc[movie_rating.UNRATED == 1]['Box Office After Inflation']

movie_rating_df = pd.DataFrame(dict(G=G, PG=PG, PG_13=PG_13, R=R, NC_17=NC_17, UNRATED=UNRATED))#.reset_index()

#movie_rating_df.drop('index', axis = 1, inplace=True)
print(movie_rating_df.mean())
movie_rating_box_office = movie_rating_df.mean()
movie_rating_box_office.plot(kind='barh', title = 'Mean_Box_Office_by_Rating', logx=False)
movie_rating_box_office.plot(kind='pie', title = 'Mean_Box_Office_by_Rating')
plt.show()





y = df1['Profit Class']
#bestfeatures = SelectKBest(score_func=f_classif, k=10)
#fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(20,'Score'))  #print 10 best features



#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
# grid search
#model = XGBClassifier()
#max_depth = range(1, 21, 2)
#print(max_depth)
#param_grid = dict(max_depth=max_depth)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
#grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
#grid_result = grid_search.fit(X, y)
# summarize results
##print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))


#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.bar(range(len(model.feature_importances_)), model.feature_importances_)

#model = xgb.XGBClassifier(n_estimators=250, n_jobs=4)
#model.fit(X,y)
#xgb.plot_importance(model, max_num_features = 30)
#plt.show()
