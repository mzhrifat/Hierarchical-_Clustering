"""
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()

x=iris['data']
y=iris['target']

logit=LogisticRegression(max_iter=1000)

print(logit.fit(x,y))

print(logit.score(x,y))
"""


#grid search  implementing

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

iris=datasets.load_iris()

x=iris['data']
y=iris['target']

logit=LogisticRegression(max_iter=1000)

C=[0.25,0.5,0.75,1,1.25,1.5,1.75,2]

scores=[]

for choice in C:
  logit.set_params(C=choice)
  logit.fit(x,y)
  scores.append(logit.score(x,y))

print(scores)