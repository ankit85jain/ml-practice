# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:47:55 2019

@author: Ankit
"""

import pandas as pd
import re
import sklearn.metrics as metrices
dataset=pd.read_csv('C:/Users/Ankit/Desktop/bigdata/python/twitter-gender-csestudy/gender-classifier-DFE-791531.csv',encoding='latin')
print(dataset.columns)
org_dataset=pd.read_csv('C:/Users/Ankit/Desktop/bigdata/python/twitter-gender-csestudy/gender-classifier-DFE-791531.csv',encoding='latin')
dataset=dataset[dataset['gender'].notnull()]
dataset=dataset[dataset['text'].notnull()]
cat_columns=['tweet_location','user_timezone','gender']
def normalize_text(s):
    # just in case
    s = str(s)
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s

dataset['text_norm'] = [normalize_text(s) for s in dataset['text']]
dataset['description_norm'] = [normalize_text(s) for s in dataset['description']]

dataset = dataset[dataset['gender:confidence']==1]

import sklearn.preprocessing as preprocessing
for col in cat_columns:
    dataset[col]=preprocessing.LabelEncoder().fit_transform(dataset[col].astype('str'))
'''
'_last_judgment_at', 'created','tweet_created',  'tweet_coord',
'''
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text_norm'])



'''X = dataset[['_unit_id', '_golden', '_unit_state', '_trusted_judgments',
        'gender:confidence', 'profile_yn',
       'profile_yn:confidence', 'description_norm', 'fav_number',
       'gender_gold', 'link_color', 'name', 'profile_yn_gold',
       'retweet_count', 'sidebar_color', 'text_norm', 'tweet_count',
        'tweet_id',  'user_timezone']]'''
y = dataset['gender']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#-------svm
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = metrices.confusion_matrix(y_test, y_pred)
acc=metrices.accuracy_score(y_test, y_pred)



#----randomforest

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = metrices.confusion_matrix(y_test, y_pred)
acc=metrices.accuracy_score(y_test, y_pred)
#0.8509554140127389



dataset2=pd.read_csv('C:/Users/Ankit/Desktop/bigdata/python/casestudy5/test.csv')
cat_columns=['workclass','education', 
       'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'native-country']

import sklearn.preprocessing as preprocessing
for col in cat_columns:
    dataset2[col]=preprocessing.LabelEncoder().fit_transform(dataset2[col].astype('str'))

num_col=['age', 'fnlwgt',  'educational-num',  
       'capital-gain', 'capital-loss', 'hours-per-week']
for col in num_col:
    dataset2[col].fillna(dataset2[col].mean(),inplace=True)
X2 = dataset2.iloc[:, :].values
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred2 = classifier.predict(X2)

df=pd.DataFrame(y_pred2 ,columns=['outcome'])
df.to_csv('C:\\Users\\Ankit\\Desktop\\bigdata\\python\\casestudy5\\casestudy5result2.csv')

'''
#---logistic
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = metrices.confusion_matrix(y_test, y_pred)
acc=metrices.accuracy_score(y_test, y_pred)
#your model accuracy is 77.8642936596218
'''