# লাইব্রেরি ইম্পোর্ট করা
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ডেটা প্রস্তুত করা
x = [4, 5, 10, 4, 3, 11, 14, 8, 10, 12]  # ইনপুট ফিচার 1
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]  # ইনপুট ফিচার 2
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]  # টার্গেট ক্লাস (0 বা 1)

# ডেটা ভিজুয়ালাইজ করা
plt.scatter(x, y, c=classes, cmap='viridis', label='Data Points')
plt.title("Initial Data Points")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.show()

# ডেটাকে (x, y) জোড়ায় রূপান্তর করা
data = list(zip(x, y))

# KNN মডেল ফিট করা (K=1)
knn = KNeighborsClassifier(n_neighbors=1)  # K=1 সেট করা
knn.fit(data, classes)  # মডেল ট্রেনিং

# নতুন ডেটা পয়েন্ট প্রেডিক্ট করা
new_x = 8
new_y = 21
new_point = [(new_x, new_y)]  # নতুন পয়েন্ট
prediction = knn.predict(new_point)  # ক্লাস প্রেডিকশন
print(f"Prediction with K=1: {prediction[0]}")

# প্রেডিকশন ভিজুয়ালাইজ করা (K=1)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]], cmap='viridis', label='Data Points')
plt.scatter(new_x, new_y, c='red', marker='x', s=100, label='New Point')
plt.text(new_x - 1.7, new_y - 0.7, s=f"Class: {prediction[0]}", fontsize=12, color='red')
plt.title("KNN Prediction with K=1")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.show()

# K=5 ব্যবহার করে আবার মডেল ফিট করা
knn = KNeighborsClassifier(n_neighbors=5)  # K=5 সেট করা
knn.fit(data, classes)  # মডেল ট্রেনিং

# নতুন ডেটা পয়েন্ট প্রেডিক্ট করা (K=5)
prediction = knn.predict(new_point)  # ক্লাস প্রেডিকশন
print(f"Prediction with K=5: {prediction[0]}")

# প্রেডিকশন ভিজুয়ালাইজ করা (K=5)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]], cmap='viridis', label='Data Points')
plt.scatter(new_x, new_y, c='red', marker='x', s=100, label='New Point')
plt.text(new_x - 1.7, new_y - 0.7, s=f"Class: {prediction[0]}", fontsize=12, color='red')
plt.title("KNN Prediction with K=5")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.show()