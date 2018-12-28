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
link = "http://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
csv_text = requests.get(link).text
# if you don't care about column names omit names=names and do headers=None instead
dataset = pd.read_csv(io.StringI (csv_text),header=None)
dataset.iloc[:,13]
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
classifier = KNeighborsClassifier(n_neighbors=7,metric="cosine")  
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
print("this is cv=10 fold crossvalidation mean ",acc.mean()) 
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print( "errorrr:",1-metrics.accuracy_score( y_global,y_pred))
from sklearn.metrics import f1_score
print("f1score",f1_score(y_global, y_pred, average='micro'))

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
#ghesmate roc e  code bardshte shud e
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle
n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_global))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='green', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Naive Bayes - IRIS DATASET')
plt.legend(loc="lower right")
plt.show()