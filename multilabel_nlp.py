# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:16:27 2019

@author: dhruv
"""
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
train=pd.read_csv("train_data.csv")
labels=pd.read_csv("train_label.csv")
test=pd.read_csv("test_data.csv")
mix=pd.concat([train,test],axis=0)
nlp = spacy.load('en_core_web_sm')
nlp =en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])

from sklearn.feature_extraction.text import CountVectorizer
text_to_tokenize=list(mix["text"])
def words_of_length(words, length):
    result=[word.lower_ for word in words if len(word)>=length]
    return(set(result))
def dummy_fun(doc):
    return doc
def text_dataframe_tolist(dataframe):
    text_to_tokenize=list(dataframe)
    tokenized =[words_of_length(nlp(word),3) for word in text_to_tokenize]
    tokenized_list= [list(y) for  y in tokenized]
    new=[]
    for i in tokenized_list:
        lis=[]
        for j in i:
            if j.isalpha():
                lis.append(j)
            new.append(lis)
    return new

traindata=text_dataframe_tolist(mix["text"])
vectorizer=CountVectorizer(tokenizer=dummy_fun, stop_words='english', lowercase=False)
vectorizer.fit(traindata)
v_parsed=vectorizer.transform(train["text"])
vectorizer.get_feature_names()
features=vectorizer.get_feature_names()


id= labels["id"]
id=list(set(id))
dic={}
for i in id:
    dic[i]=[]
for i in range(len(labels)):
    dic[labels["id"][i]].append(labels["label"][i])

vectorizer2=CountVectorizer(tokenizer=dummy_fun, stop_words='english', lowercase=False)
v2=vectorizer.fit_transform(dic.values())

a=pd.Series(list(dic.keys()),name="id")
b=pd.Series(list(v2.toarray()),name="labelvec")
y_=pd.concat([a,b],axis=1)
final=pd.merge(train,y_,on="id")
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
target=final["labelvec"]

cop=np.asarray(target)
temp=[]
for i in range(len(cop)):
    temp.append(list(cop[i]))
target=np.array(temp)

fit1=clf.fit(v_parsed,target)


testparsed=text_dataframe_tolist(train["text"])
test_parsed=vectorizer.transform(testparsed)
results=clf.predict(test_parsed)



