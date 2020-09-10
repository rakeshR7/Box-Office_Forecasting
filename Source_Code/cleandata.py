# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:02:03 2019

@author: JLiu
"""


import pandas as pd
import math
import openpyxl
import xlsxwriter
import statistics
import datetime 
df = pd.read_excel('IAC4-data_whole.xlsx')
actors = set()
actors_box_office = dict()
for index,row in df.iterrows():
    y= row['actors']
    if(pd.notnull(y)):
        x= y.split(', ')
        actors.update(list(x))
        


for index,row in df.iterrows():
    y = row['actors']
    if(pd.notnull(y)):
        a= y.split(', ')
        for i in range(len(a)):
            if a[i] not in actors_box_office:
                actors_box_office[a[i]] = []
            if row['release_date'].year < 2016:
                actors_box_office[a[i]].append(row['Box Office After Inflation'])
#print(actors_box_office['Robert Downey Jr.'])
        
for index,row in df.iterrows():
    sum = 0
    y = row['actors']
    if(pd.notnull(y)):
        a = y.split(', ')
        for i in range(len(a)):
            if len(actors_box_office[a[i]]) != 0:                
                sum = sum+statistics.mean(actors_box_office[a[i]])
    df.loc[index,'actors']=sum

#print(df['actors'])
production = set()
for index,row in df.iterrows():
    y = row['production']
    if(pd.notnull(y)):
        a = y.split(', ')
        production.update(list(a))
        
production_box_office = dict()

for index,row in df.iterrows():
    y = row['production']
    if(pd.notnull(y)):
        a= y.split(', ')
        for i in range(len(a)):
            if a[i] not in production_box_office:
                production_box_office[a[i]] = []
            if row['release_date'].year < 2016:
                production_box_office[a[i]].append(row['Box Office After Inflation'])

for index,row in df.iterrows():
    sum = 0
    y = row['production']
    if(pd.notnull(y)):
        a = y.split(', ')
        for i in range(len(a)):
            if len(production_box_office[a[i]]) != 0:                
                sum = sum+statistics.mean(production_box_office[a[i]])
    df.loc[index,'production']=float(sum)
#print(df['production'])
            
director = set()
for index,row in df.iterrows():
    y = row['director']
    if(pd.notnull(y)):
        a = y.split(', ')
        director.update(list(a))
director_box_office = dict()        
for index,row in df.iterrows():
    y = row['director']
    if(pd.notnull(y)):
        a= y.split(', ')
        for i in range(len(a)):
            if a[i] not in director_box_office:
                director_box_office[a[i]] = []
            if row['release_date'].year < 2016:
                director_box_office[a[i]].append(row['Box Office After Inflation'])

for index,row in df.iterrows():
    sum = 0
    y = row['director']
    if(pd.notnull(y)):
        a = y.split(', ')
        for i in range(len(a)):
            if len(director_box_office[a[i]]) != 0:                
                sum = sum+statistics.mean(director_box_office[a[i]])
    sum = sum/len(a)
    df.loc[index,'director']=float(sum)
#print(df['director'])
for index,row in df.iterrows():
    runtime = 0
    y = row['runtime']   
    #print(y)
    a = str(y).split(' ')
    runtime = int(a[0])
    df.loc[index,'runtime']=runtime

df['movie_rating'].fillna('UNRATED', inplace=True)
df['movie_rating'].replace('NOT RATED', 'UNRATED',inplace=True)
df['movie_rating'].replace('NR','UNRATED',inplace=True)
df['movie_rating'].replace('TV-PG', 'PG', inplace=True)
#print(df['movie_rating'])
movie_rating = df['movie_rating'].str.get_dummies(sep = ', ')
df = df.join(movie_rating)

genre = df['genre'].str.get_dummies(sep = ', ')
df = df.join(genre)
#print(type(df.info()))
#df.set_index('title')
df.drop(['movie_id','title', 'plot', 'movie_rating', 'metacritic', 'dvd_release', 'genre', 'awards', 'keywords', 'Budget', 'Box Office Gross','Net Gross', 'Net Gross After Inflation','Gross Ratio'], axis=1,inplace=True)
df['release_season'] = ""
for index,row in df.iterrows():
    runtime = 0
    y = row['release_date']   
    #print(y)
    if y.month> 4 and y.month < 8:
        df.loc[index,'release_season']="Summer"
    elif y.month> 10:
        df.loc[index,'release_season']="Holiday"
    else:
        df.loc[index,'release_season']="Dump"

release_season = df['release_season'].str.get_dummies(sep = ', ')
df = df.join(release_season)


oldmovie = df['release_date']<datetime.datetime(2016,1,1)
newmovie = df['release_date']>=datetime.datetime(2016,1,1)
train_data = df[oldmovie]
test_data = df[newmovie]
train_data.drop(['release_date','release_season'], axis = 1, inplace = True)
test_data.drop(['release_date', 'release_season'], axis = 1, inplace = True)
#print(train_data.info())
train_data = train_data.apply(pd.to_numeric, errors='coerce')
#print(train_data.info())
test_data = test_data.apply(pd.to_numeric, errors='coerce')
x_train = train_data.drop(['Box Office After Inflation', 'Profit Class'], axis = 1)
x_test = test_data.drop(['Box Office After Inflation', 'Profit Class'], axis = 1)
y_regression_train = train_data[['Box Office After Inflation']]
y_regression_test = test_data[['Box Office After Inflation']]
y_classification_train = train_data[['Profit Class']]
y_classification_test = test_data[['Profit Class']]
#print(df.loc[100])
#df.to_excel('updated_test_train.xlsx')
x_train.to_excel('x_train.xlsx')
x_test.to_excel('x_test.xlsx')
y_regression_train.to_excel('y_regression_train.xlsx')
y_regression_test.to_excel('y_regression_test.xlsx')
y_classification_train.to_excel('y_classification_train.xlsx')
y_classification_test.to_excel('y_classification_test.xlsx')