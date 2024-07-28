from pandas import read_csv

filename = 'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names, skiprows=1)
class_counts = data.groupby('class').size()
print(class_counts)

'''
This code calculates the number of occurrences of each unique value in the 'class' column.

    data.groupby('class'): Groups the data by the 'class' column.
    .size(): Counts the number of rows in each group.
    print(class_counts): Prints the resulting counts to the console.
    
explaination from gemini
'''
#Output
'''
class
0    500
1    268

class have 500 rows of value 0 and 268 rows of value 1
'''