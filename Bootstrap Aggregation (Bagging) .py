#wine 82% accurate
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split  # Fixed typo here
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the wine dataset
data = datasets.load_wine(as_frame=True)

# Define features (X) and target (y)
X = data.data  # Use uppercase X for consistency
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Initialize and train the Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(X_train, y_train)

# Make predictions
y_pred = dtree.predict(X_test)

# Calculate accuracy
print("Train data accuracy:", accuracy_score(y_true=y_train, y_pred=dtree.predict(X_train)))
print("Test data accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))


#creating as begging classifier

import matplotlib.pyplot as plt

#Generate the plot of Scores against number of estimators
plt.figure(figsize=(9,6))
plt.plot(estimator_range,scores)

#Adjust labels and font (to make visable)
plt.xlabel("n_estimators",fontsize=18)
plt.ylabel("score",fontsize=18)
plt.tick_params(labelsize=16)

#visualize plt
plt.show()
"""


"""
# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = load_wine()
X = data.data  # Features
y = data.target  # Target labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Define the range of n_estimators to test
estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]

# Lists to store models and their corresponding scores
models = []
scores = []

# Loop through the estimator range
for n_estimators in estimator_range:
    # Create a Bagging Classifier
    clf = BaggingClassifier(
        n_estimators=n_estimators,
        random_state=22
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy and store the model and score
    score = accuracy_score(y_test, y_pred)
    models.append(clf)
    scores.append(score)

# Plot the results
plt.figure(figsize=(9, 6))  # Set the figure size
plt.plot(estimator_range, scores, marker='o', linestyle='-', color='b')  # Plot the scores

# Add labels and title
plt.xlabel("Number of Estimators (n_estimators)", fontsize=18)
plt.ylabel("Accuracy Score", fontsize=18)
plt.title("Bagging Classifier Performance", fontsize=20)
plt.tick_params(labelsize=16)  # Increase tick label font size

# Show the plot
plt.show()

"""

#Create a model with out-of-bag metric.

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

data=datasets.load_wine(as_frame=True)

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=22)

oob_model=BaggingClassifier(n_estimators=12,oob_score=True,random_state=22)

oob_model.fit(X_train,y_train)

print(oob_model.oob_score)

