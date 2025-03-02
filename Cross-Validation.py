#chatgpt example
"""
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# ডেটাসেট লোড করা
data = load_iris()
X, y = data.data, data.target

# মডেল ডিফাইন করা
model = LogisticRegression(max_iter=200)

# k-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

# রেজাল্ট প্রিন্ট করা
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


#Run k-fold for cv

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,cross_val_score

X,y= datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=42)

k_folds = KFold(n_splits = 5)

scores=cross_val_score(clf,X,y ,cv=k_folds)

print("Croos Validation Scores:",scores)
print("Average CV Score:",scores.mean())
print("Number of CV SCores used in Average:",len(scores))


#k-Fold Cross Validation

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Iris ডেটাসেট লোড করা
x,y = datasets.load_iris(return_X_y=True)

#Decision Tree model make

clf=DecisionTreeClassifier(random_state=42)

#Startified k-fold cross validation setup
sk_folds = StratifiedKFold(n_splits=5)

# মডেল মূল্যায়ন করা
scores = cross_val_score(clf, x, y, cv=sk_folds)

# ফলাফল প্রিন্ট করা
print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))



#Leave-One-Out (LOO) cross validation
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut,cross_val_score

X,y=datasets.load_iris(return_X_y=True)

clf=DecisionTreeClassifier(random_state=42)

loo=LeaveOneOut()

scores=cross_val_score(clf,X,y,cv=loo)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

"""
#Leave-p-out

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeavePOut,cross_val_score

X,y=datasets.load_iris(return_X_y=True)

clf=DecisionTreeClassifier(random_state=42)

lpo=LeavePOut(p=2)

scores=cross_val_score(clf,X,y,cv=lpo)


print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))
