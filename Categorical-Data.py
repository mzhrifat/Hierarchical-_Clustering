#Categorical Data
"""
import pandas as pd

cars=pd.read_csv('C:\\Machine-Leraning\\Hierarchical_Clustering\\data1.csv')

print (cars.to_string())


#One Hot Encoding
import pandas as pd

cars=pd.read_csv('C:\\Machine-Leraning\\Hierarchical_Clustering\\data1.csv')
print(ohe_cars.to_string())
"""

#predict co2

import pandas as pd
import numpy as np
from sklearn import linear_model

# CSV ফাইল থেকে ডেটা লোড করা
cars = pd.read_csv('C:\\Machine-Leraning\\Hierarchical_Clustering\\data1.csv')

# ক্যাটাগরিকাল ডেটাকে ডামি ভেরিয়েবলে রূপান্তর
ohe_cars = pd.get_dummies(cars[['Car']])

# ইনডিপেনডেন্ট (X) এবং ডিপেনডেন্ট (y) ভেরিয়েবল তৈরি
X = pd.concat([cars[['volume', 'weight']], ohe_cars], axis=1)
y = cars['CO2']

# মডেল তৈরি করা
regr = linear_model.LinearRegression()
regr.fit(X, y)

# নতুন ডেটার জন্য পূর্বাভাস
test_data = np.array([[2300, 1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])
predictedCO2 = regr.predict(test_data)

print(predictedCO2)
