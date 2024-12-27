# Practicum 1 - Case 1. Naïve Bayes Classifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Dataset ==> features data [humidity, temperature, leaf_color], target/label: some diseases
X = np.array([
    [80, 30, 1],
    [81, 31, 1],
    [60, 35, 2],
    [90, 25, 1],
    [50, 40, 3],
    [52, 41, 3],
    [70, 32, 2],
    [85, 28, 1],
    [60, 38, 3],
    [55, 33, 2],
    [54, 31, 2],
])

y = np.array(['Powdery Mildew', 'Powdery Mildew', 'Rust', 'Powdery Mildew',
              'Blight', 'Rust', 'Rust', 'Powdery Mildew', 'Blight', 'Rust', 'Rust'])

model = GaussianNB()
model.fit(X, y)

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nEnter data to predict plant disease:")
humidity = float(input("Humidity (0-100): "))
temperature = float(input("Temperature (Celsius): "))
leaf_color = int(input("Leaf color (1 = green, 2 = yellowish, 3 = brown): "))

input_data = np.array([[humidity, temperature, leaf_color]])
prediction = model.predict(input_data)

print(f"Predicted Disease: {prediction[0]}")