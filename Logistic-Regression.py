#See the whole example in action
"""
import numpy as np
from sklearn import linear_model

x=np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr=linear_model.LogisticRegression()
logr.fit(x,y)

predicted=logr.predict(np.array([3.46]).reshape(-1,1))
print(predicted)


#Coefficient

import numpy as np
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn import linear_model

#Reshaped for Logistic function.
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr=linear_model.LogisticRegression()
logr.fit(X,y)

log_odds=logr.coef_
odds=np.exp(log_odds)
print(odds)
"""

#Function Explained

import numpy as np
from sklearn import linear_model

X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


logr=linear_model.LogisticRegression()
logr.fit(X,y)

def logit2prob(logr,X):
    log_odds=logr.coef_ *X + logr.intercept_
    odds=np.exp(log_odds)
    probability=odds/(1+odds)
    return (probability)

print(logit2prob(logr, X))