from pandas import read_csv
from numpy import set_printoptions # allows you to set the number of decimal places, the precision, the threshold for printing arrays, and other formatting options.
from sklearn.preprocessing import MinMaxScaler # transforming the data so that the minimum value becomes 0 and the maximum value becomes 1

filename = 'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards
array = dataframe.values

# separate array into input and output components
X = array[:,0:8] # extracts the first 8 columns (features) from the NumPy array array and assigns them to the variable X
Y = array[:,8] # extracts the last column (target variable) from the NumPy array array and assigns it to the variable y
scaler = MinMaxScaler(feature_range=(0, 1)) # transform the data to have a minimum value of 0 and a maximum value of 1
rescaledX = scaler.fit_transform(X) # calculates the scaling parameters based on X

# summarize transformed data
set_printoptions(precision=3) # when the transformed data is printed, it will be rounded to 3 decimal places
print(rescaledX[0:5,:]) # print first 5 row of transformed data

# Output
'''
[[0.353 0.744 0.59  0.354 0.    0.501 0.234 0.483] 
 [0.059 0.427 0.541 0.293 0.    0.396 0.117 0.167] 
 [0.471 0.92  0.525 0.    0.    0.347 0.254 0.183] 
 [0.059 0.447 0.541 0.232 0.111 0.419 0.038 0.   ] 
 [0.    0.688 0.328 0.354 0.199 0.642 0.944 0.2  ]]
 '''