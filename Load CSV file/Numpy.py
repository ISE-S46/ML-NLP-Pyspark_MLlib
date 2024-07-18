import numpy as np

# if import file from internet directly use this
'''from urllib import request
filename = 'https://goo.gl/vhm1eU'
raw_data = request.urlopen(filename)'''

# code to load CSV (comma-separated values) file 

filename = 'D:\Important_Data\Code\pima-indians-diabetes.csv' #file directort
raw_data = open(filename, 'rb') #open file, set read as binary mode
data = np.loadtxt(raw_data, delimiter=",") # delimeter tell NumPy to split the values in each line based on commas
print(data.shape) 
