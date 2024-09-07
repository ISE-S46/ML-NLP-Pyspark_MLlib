from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

filename = 'D:\Important_Data\Code\diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) 
array = dataframe.values
X = array[:,0:8] 
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
model = DecisionTreeClassifier(random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Output
'''
0.6940874914559126
'''