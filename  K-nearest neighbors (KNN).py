#Start by visualizing some data points:
"""
import matplotlib.pyplot as plt

x=[4,5,10,4,3,11,14,8,10,12]
y=[21,19,24,17,16,25,24,22,21,21]

classes=[0,0,1,0,0,1,1,0,1,1]

plt.scatter(x,y,classes)
plt.show()


#
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]
classes = [0, 0, 1, 1, 1]  # Example class labels

# Combine x and y into a 2D array
data = list(zip(x, y))

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data, classes)

# New point to predict
new_x = 8
new_y = 21
new_point = [(new_x, new_y)]

# Predict the class of the new point
prediction = knn.predict(new_point)

# Plot the data points and the new point
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
"""

#KNN full code

#libary import kora
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#DATA MAKE
x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

#Data Visulaize
plt.scatter(x,y,classes)
plt.show()

#knn model fit(k=1)
data=list(zip(x,y))
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(dataclasses)

#new data predict
new_x=8
new_y=21
new_point=[(new_x,new_y)]
prediction=knn.predict(new_point)

#prediction visulize

plt.scatter(x+[new_x],y+[new_y],c=classes+[prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()