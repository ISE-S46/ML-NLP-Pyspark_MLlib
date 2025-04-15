from pandas import read_csv
from numpy import set_printoptions # allows you to set the number of decimal places, the precision, the threshold for printing arrays, and other formatting options.
from sklearn.preprocessing import Normalizer 

filename = '../diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards
array = dataframe.values

# separate array into input and output components
X = array[:,0:8] # extracts the first 8 columns (features) from the NumPy array array and assigns them to the variable X
Y = array[:,8] # extracts the last column (target variable) from the NumPy array array and assigns it to the variable y
scaler = Normalizer().fit(X) 
normalizedX = scaler.transform(X) 

# summarize transformed data
set_printoptions(precision=3) # when the transformed data is printed, it will be rounded to 3 decimal places
print(normalizedX[0:5,:]) # print first 5 row of transformed data

# Output
'''
[[0.034 0.828 0.403 0.196 0.    0.188 0.004 0.28 ]
 [0.008 0.716 0.556 0.244 0.    0.224 0.003 0.261]
 [0.04  0.924 0.323 0.    0.    0.118 0.003 0.162]
 [0.007 0.588 0.436 0.152 0.622 0.186 0.001 0.139]
 [0.    0.596 0.174 0.152 0.731 0.188 0.01  0.144]]
'''