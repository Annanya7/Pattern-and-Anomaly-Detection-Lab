# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:43:27 2021

@author: hp
"""

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

#%%
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=1)
data=pd.DataFrame({'X' : X.flatten(), 'y' : y})
#%%
X=data.iloc[:, :-1].values
y=data.iloc[:, 1].values
#%%
X_train, X_test, Y_train, Y_test=train_test_split(X, y, test_size=0.2, random_state=1)

#%%
tuned_params  =[
        {"fit_intercept" : ["True"] , 
         "normalize" :["True"]},
         {
           "fit_intercept": ["False"],
           "normalize" :["True"]
                 },
           {
           "fit_intercept": ["False"],
           "normalize" :["False"]
                 }
 ]

score = 'r2'
#%%
model = GridSearchCV(
        LinearRegression() , tuned_params,cv = 5 , scoring=  score)
model.fit(X_train , Y_train)

#%%
model.best_params_
#%%
from sklearn.datasets import make_classification
X , Y = make_classification(n_samples= 100 , n_features= 5 , n_classes= 2 )
#%%
model_params  ={ 
    'svm' : {
        'model' : svm.SVC(gamma = 'auto'),
        'params' : {
            'C' : [1 , 10  , 20] , 
            'kernel' : ['rbf' , 'linear']
        }
    },
    
    'random_forest' : {
        
        'model' : RandomForestClassifier(),
        'params' : {
            'n_estimators' : [1 ,5 , 10]
        }
        
    },


'ridge_classifier' :{
            
            'model' : RidgeClassifier(normalize = True , solver = 'saga'   ), 
            'params' : {}
            },
 
   }

#%%
scores = []
for model_name  , mp in model_params.items():
    
    clf = GridSearchCV(mp['model'] , mp['params'] , cv = 5 , return_train_score=False)
    clf.fit(X , Y)
    scores.append({
        'model' : model_name ,
        'best_score' :clf.best_score_ ,
        'best_params' :clf.best_params_
    })
#%%
df = pd.DataFrame(scores , columns=['model' , 'best_score' , 'best_params'])
print(df)     
#%%