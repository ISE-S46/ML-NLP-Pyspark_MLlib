# Practicum 3 - Case 3. Naïve Bayes Classifier
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os

# Filepath to dataset
filepath = '/content/drive/MyDrive/Dataset/SMSSpamCollection.txt'

print(f"File exists: {os.path.exists(filepath)}")

if os.path.exists(filepath):
    # Load dataset
    sms_data = pd.read_csv(filepath, sep='\t', header=None, names=['Label', 'Message'])

    print(sms_data.head())

    # Map labels to numerical values
    sms_data['Label'] = sms_data['Label'].map({'ham': 0, 'spam': 1})

    # Split data into features and target variable
    X = sms_data['Message']
    y = sms_data['Label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Vectorize text data
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Train Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)

    # Make predictions
    y_pred = model.predict(X_test_vect)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Model Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("The file does not exist at the given path. Please check the path.")