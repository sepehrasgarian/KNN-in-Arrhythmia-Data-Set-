  # -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:25:09 2018

@author: sepehr
"""
import requests
import pandas as pd
import io
import numpy
import numpy as np  
import matplotlib.pyplot as plt  
names = ['smple']
link = "http://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
csv_text = requests.get(link).text
# if you don't care about column names omit names=names and do headers=None instead
dataset = pd.read_csv(io.StringIO(csv_text),header=None)
dataset= dataset.replace("?", 55)
# fill missing values with mean column values
dataset.dropna(inplace=True)
X = dataset.iloc[:, :-1].values
#s = X.Series()                
#print(pd.to_numeric(s, errors='coerce'))                
print(X)
y = dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=0)  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier  
'''class sklearn.neighbors.NearestNeighbors(n_neighbors=5, radius=1.0, 
    algorithm='auto', leaf_size=30, 
    metric='minkowski', p=2, metric_params=None, n_jobs=1, **kwargs)'''
classifier = KNeighborsClassifier(n_neighbors=9,metric="manhattan")  
classifier.fit(X_train, y_train)  
X_global=X_test#define global
y_global=y_test
y_pred = classifier.predict(X_global) 
#Evaluating the Algorithm
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_global, y_pred))  
print(classification_report(y_global, y_pred))  
from sklearn.model_selection import cross_val_score
acc=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print(acc.mean()) 
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print( "errorrr:",1-metrics.accuracy_score( y_global,y_pred))
from sklearn.metrics import f1_score
f1_score(y_global, y_pred, average='micro')
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_global)
    error.append(np.mean(pred_i != y_global))
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')   
acc2=[] 
for i in range(1, 40):  
    print("this is i",i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    acc=cross_val_score(estimator=knn,X=X_test,y=y_test,cv=10)
    print("cfget    ",i,acc.mean())
    acc2.append(acc.mean())
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), acc2, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('kfold')  
plt.xlabel('K Value')  
plt.ylabel('acc.mean')
x=np.argmax(acc2)
m = max(acc2)
print("bestparamdeth",[i for i, j in enumerate(acc2) if j == m])
knn1 = KNeighborsClassifier(n_neighbors=i)
knn1.fit(X_train, y_train)

    

