from pandas import read_csv

filename = '../diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names, skiprows=1)
shape = data.shape
print(shape) # display row and column of dataset

# Output
'''
(768, 9)
'''