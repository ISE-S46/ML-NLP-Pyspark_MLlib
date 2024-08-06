from pandas import read_csv
from matplotlib import pyplot

filename = 'D:\Important_Data\Code\diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards

data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False) # same parameter as density plot
pyplot.show()