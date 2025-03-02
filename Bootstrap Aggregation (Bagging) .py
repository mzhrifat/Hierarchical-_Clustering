#wine 82% accurate

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

#creatin as begging classifier

