# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
filename = 'D:\Important_Data\Code\diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names, skiprows=1) # setting skiprows=1, telling Pandas to start reading the data from the second row onwards
array = dataframe.values
X = array[:,0:8] # extracts the first 8 columns (features) from the NumPy array array and assigns them to the variable X
Y = array[:,8] # extracts the last column (target variable) from the NumPy array array and assigns it to the variable y
# feature extraction
test = SelectKBest(score_func=chi2, k=4) # uses the chi-squared (chi2) statistical test for non-negative features to select 4 of the best features (k)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

# Output
'''
[ 111.52  1411.887   17.605   53.108 2175.565  127.669    5.393  181.304]
[[148.    0.   33.6  50. ] 
 [ 85.    0.   26.6  31. ] 
 [183.    0.   23.3  32. ] 
 [ 89.   94.   28.1  21. ] 
 [137.  168.   43.1  33. ]]
'''