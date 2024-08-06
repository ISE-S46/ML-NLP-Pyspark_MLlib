from pandas import read_csv
from numpy import set_printoptions # allows you to set the number of decimal places, the precision, the threshold for printing arrays, and other formatting options.
from sklearn.preprocessing import StandardScaler # standardizes input(features) by removing the mean and scaling to unit variance 

filename = 'D:\Important_Data\Code\diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards
array = dataframe.values

# separate array into input and output components
X = array[:,0:8] # extracts the first 8 columns (features) from the NumPy array array and assigns them to the variable X
Y = array[:,8] # extracts the last column (target variable) from the NumPy array array and assigns it to the variable y
scaler = StandardScaler().fit(X) # transfrom all value of X to have mean of 0 and a standard deviation of 1 ***not standard normal distribution
rescaledX = scaler.transform(X) # applies the standardization to the input data X

# summarize transformed data
set_printoptions(precision=3) # when the transformed data is printed, it will be rounded to 3 decimal places
print(rescaledX[0:5,:]) # print first 5 row of transformed data

# Output
'''
[[ 0.64   0.848  0.15   0.907 -0.693  0.204  0.468  1.426] 
 [-0.845 -1.123 -0.161  0.531 -0.693 -0.684 -0.365 -0.191] 
 [ 1.234  1.944 -0.264 -1.288 -0.693 -1.103  0.604 -0.106] 
 [-0.845 -0.998 -0.161  0.155  0.123 -0.494 -0.921 -1.042] 
 [-1.142  0.504 -1.505  0.907  0.766  1.41   5.485 -0.02 ]]
'''