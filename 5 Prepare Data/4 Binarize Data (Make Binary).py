from pandas import read_csv
from numpy import set_printoptions 
from sklearn.preprocessing import Binarizer 

filename = 'D:\Important_Data\Code\diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) 
array = dataframe.values

# separate array into input and output components
X = array[:,0:8] 
Y = array[:,8] 
binarizer = Binarizer(threshold=0.0).fit(X) # all values equal or less than 0 are marked 0 and all of those above 0 are marked 1
binaryX = binarizer.transform(X)

# summarize transformed data
set_printoptions(precision=3) 
print(binaryX[0:5,:]) 

# Output
'''
[[1. 1. 1. 1. 0. 1. 1. 1.]
 [1. 1. 1. 1. 0. 1. 1. 1.]
 [1. 1. 1. 0. 0. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [0. 1. 1. 1. 1. 1. 1. 1.]]
'''