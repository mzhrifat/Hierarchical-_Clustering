
"""
import pandas as pd

cars=pd.read_csv('C:\\Machine-Leraning\\Hierarchical_Clustering\\data1.csv')

print (cars.to_string())
"""


import pandas as pd

cars = pd.read_csv('C:\\Machine-Leraning\\Hierarchical_Clustering\\data1.csv')
ohe_cars = pd.get_dummies(cars[['Car']])

print(ohe_cars.to_string())