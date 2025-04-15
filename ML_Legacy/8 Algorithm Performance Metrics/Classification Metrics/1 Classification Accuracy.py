from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = '../diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) 
array = dataframe.values
X = array[:,0:8] 
Y = array[:,8]
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(max_iter=1000)
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring) # cv is Cross Validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

# Output
'''
Accuracy: 0.772 (0.050)
'''