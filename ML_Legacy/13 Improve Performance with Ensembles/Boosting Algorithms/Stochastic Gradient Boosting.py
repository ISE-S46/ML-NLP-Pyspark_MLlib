# Stochastic Gradient Boosting Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

filename = '../diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) 
array = dataframe.values
X = array[:,0:8] 
Y = array[:,8]
seed=7
num_trees = 100
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Output (mean estimate of classification accuracy)
# 0.7578947368421053