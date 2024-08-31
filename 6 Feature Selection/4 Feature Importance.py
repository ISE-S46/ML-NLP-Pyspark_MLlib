# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
filename = 'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) 
array = dataframe.values
X = array[:,0:8] 
Y = array[:,8] 
# feature extraction
model = ExtraTreesClassifier(random_state=7)
model.fit(X, Y)
print(model.feature_importances_)

# Output
'''
[0.10750328 0.22984525 0.09910173 0.0790034  0.07802185 0.14452426
 0.11741104 0.14458919]
'''