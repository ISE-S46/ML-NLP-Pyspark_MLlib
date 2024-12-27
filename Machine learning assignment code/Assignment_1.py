# Practicum 1. Rule-Based Classifier with Accuracy

def predict_class(data):
    if data["age"] > 30 and data["income"] > 50000:
        return "buy"
    elif data["marital_status"] == "single" and data["income"] < 30000:
        return "don't buy"
    elif data["age"] < 25 and data["income"] > 40000:
        return "buy"
    else:
        return "unknown"

new_data = [
    {"age": 32, "income": 61000, "marital_status": "married", "true_class": "buy"},
    {"age": 22, "income": 25000, "marital_status": "single", "true_class": "don't buy"},
    {"age": 28, "income": 45000, "marital_status": "married", "true_class": "buy"},
    {"age": 40, "income": 35000, "marital_status": "single", "true_class": "don't buy"},
    {"age": 19, "income": 50000, "marital_status": "single", "true_class": "buy"},
    {"age": 45, "income": 55000, "marital_status": "married", "true_class": "buy"},
    {"age": 23, "income": 42000, "marital_status": "single", "true_class": "buy"},
    {"age": 35, "income": 29000, "marital_status": "single", "true_class": "don't buy"},
    {"age": 30, "income": 70000, "marital_status": "married", "true_class": "buy"},
    {"age": 27, "income": 38000, "marital_status": "single", "true_class": "don't buy"},
    {"age": 50, "income": 75000, "marital_status": "married", "true_class": "buy"}
]

correct_predictions = 0
total_predictions = len(new_data)

for idx, data in enumerate(new_data):
    prediction_result = predict_class(data)
    if prediction_result == data["true_class"]:
        correct_predictions += 1
    print(f"Data {idx + 1}: Prediction = {prediction_result}, True class = {data['true_class']}")

accuracy = correct_predictions / total_predictions
print(f"\nAccuracy: {accuracy * 100:.2f}%")