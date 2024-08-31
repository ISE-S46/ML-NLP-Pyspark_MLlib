from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

filename = 'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards

scatter_matrix(data)
pyplot.show()