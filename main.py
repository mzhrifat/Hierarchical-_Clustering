"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage


x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]

data=list(zip(x,y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

#plt.scatter(x,y)
plt.show()

"""

#a 2-dimensional plot:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

# affinity সরিয়ে ফেলা হয়েছে
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')

labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x, y, c=labels, cmap='viridis')  # সুন্দর কালার ম্যাপ যোগ করা হয়েছে
plt.show()





