# Practicum 1 - Decision Tree Algorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = {
    'Temperature': ['Mild', 'Hot', 'Hot', 'Hot', 'Mild', 'Mild', 'Mild', 'Cool', 'Cool', 'Cool', 'Hot'],
    'Humidity': ['High', 'High', 'High', 'Low', 'High', 'High', 'Low', 'Low', 'High', 'Low', 'High'],
    'Outlook': ['Overcast', 'Sunny', 'Sunny', 'Overcast', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Sunny', 'Sunny', 'Overcast'],
    'PlayTennis': ['Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Low': 1})
df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rainy': 2})
df['PlayTennis'] = df['PlayTennis'].map({'No': 0, 'Yes': 1})

X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.title("Decision Tree Visualization for Weather Prediction")
plt.show()

print("\nPrediction Results and Explanation:")
for i in range(len(X_test)):
    test_data = X_test.iloc[i]
    prediction = "Play Tennis" if y_pred[i] == 1 else "Do not play Tennis"
    actual = "Play Tennis" if y_test.iloc[i] == 1 else "Do not play Tennis"
    print(f"\nTest Data {i + 1}:")
    print(f"  Temperature: {['Hot', 'Mild', 'Cool'][test_data['Temperature']]}, "
          f"Humidity: {['High', 'Low'][test_data['Humidity']]}, "
          f"Outlook: {['Sunny', 'Overcast', 'Rainy'][test_data['Outlook']]}")
    print(f"  Predicted: {prediction} | Actual: {actual}")
    print("Explanation: The decision tree uses the combination of features to make the prediction.\n")