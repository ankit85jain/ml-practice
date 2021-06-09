# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:06:01 2019

@author: Ankit
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import Recommenders as Recommenders
    
song_df=pd.read_csv('train_orig.csv',encoding='latin')
rec_df=pd.read_csv('recommendations_orig.csv',encoding='latin',names=['user_id'])
#print(song_df.columns)
#Merge song title and artist_name columns to make a merged column\n",
#song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
song_grouped = song_df.groupby(['song_id']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song_id'], ascending = [0,1])
users1 = song_df['user_id'].unique()
users=rec_df['user_id'].unique()
songs = song_df['song_id'].unique()
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
#print(train_data.head(5))
pm = Recommenders.popularity_recommender_py()
pm.create(song_df, 'user_id', 'song_id')
user_id = users[5]
df2=pd.DataFrame(columns=['song_id1','song_id2','song_id3','song_id4','song_id5','song_id6','song_id7','song_id8','song_id9','song_id10'],dtype=str,index=users)
#df1=pd.DataFrame([rec4],columns=['song_id1','song_id2','song_id3','song_id4','song_id5','song_id6','song_id7','song_id8','song_id9','song_id10'],dtype=str,index=[user_id])
i=0
#Recommend songs for the user using personalized model
for user_id1 in users:
    rec2=pm.recommend(user_id1).drop(['score','Rank'],1).set_index(['user_id'])['song_id'].tolist()
    df2.loc[user_id1]=rec2
    '''i=i+1
    if i==3:
        break'''
df2.to_csv('recomendation1.csv')

#sol2
is_model = Recommenders.item_similarity_recommender_py()
is_model.create(song_df, 'user_id', 'song_id')
df=pd.DataFrame(columns=['user_id','song_id1','song_id2','song_id3','song_id4','song_id5','song_id6','song_id7','song_id8','song_id9','song_id10'])
rec2=is_model.recommend(user_id).drop(['score','rank'],1)
rec3=rec2.set_index(['user_id'])
rec4=rec2['song'].tolist()

df1=pd.DataFrame(columns=['song_id1','song_id2','song_id3','song_id4','song_id5','song_id6','song_id7','song_id8','song_id9','song_id10'],dtype=str,index=users)
#df1=pd.DataFrame([rec4],columns=['song_id1','song_id2','song_id3','song_id4','song_id5','song_id6','song_id7','song_id8','song_id9','song_id10'],dtype=str,index=[user_id])
i=0
#Recommend songs for the user using personalized model
for user_id1 in users:
    rec2=is_model.recommend(user_id1).drop(['score','rank'],1).set_index(['user_id'])['song'].tolist()
    df1.loc[user_id1]=rec2
    '''i=i+1
    if i==3:
        break'''
df1.to_csv('recomendation2.csv')