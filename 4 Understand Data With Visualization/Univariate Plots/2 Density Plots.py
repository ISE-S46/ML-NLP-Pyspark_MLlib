from pandas import read_csv
from matplotlib import pyplot

filename = 'D:\Important_Data\Code\diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False) 
# specify as density plot, subplots should be created for each column in the DataFrame, layout of the subplots as a 3x3 grid, each subplot should have its own x-axis
pyplot.show() # show density plot