def predict_class(data):
    if data["credit_score"] >= 700 and data["annual_income"] > 50000:
        return "approved"
    elif data["credit_score"] >= 650 and data["debt_to_income_ratio"] < 35:
        return "approved"
    elif data["credit_score"] < 600 or data["debt_to_income_ratio"] > 45:
        return "rejected"
    else:
        return "pending"

# Sample Loan Application Data
loan_data = [
    {"credit_score": 720, "annual_income": 80000, "debt_to_income_ratio": 30, "true_class": "approved"},
    {"credit_score": 680, "annual_income": 40000, "debt_to_income_ratio": 25, "true_class": "approved"},
    {"credit_score": 550, "annual_income": 30000, "debt_to_income_ratio": 50, "true_class": "rejected"},
    {"credit_score": 710, "annual_income": 60000, "debt_to_income_ratio": 40, "true_class": "approved"},
    {"credit_score": 630, "annual_income": 45000, "debt_to_income_ratio": 42, "true_class": "rejected"},
    {"credit_score": 640, "annual_income": 55000, "debt_to_income_ratio": 33, "true_class": "pending"},
    {"credit_score": 760, "annual_income": 120000, "debt_to_income_ratio": 20, "true_class": "approved"},
    {"credit_score": 590, "annual_income": 20000, "debt_to_income_ratio": 48, "true_class": "rejected"},
    {"credit_score": 680, "annual_income": 75000, "debt_to_income_ratio": 37, "true_class": "approved"},
    {"credit_score": 710, "annual_income": 85000, "debt_to_income_ratio": 28, "true_class": "approved"},
    {"credit_score": 580, "annual_income": 40000, "debt_to_income_ratio": 46, "true_class": "rejected"},
    {"credit_score": 640, "annual_income": 65000, "debt_to_income_ratio": 32, "true_class": "pending"}
]

# Evaluate the classifier
correct_predictions = 0
total_predictions = len(loan_data)

for idx, data in enumerate(loan_data):
    prediction_result = predict_class(data)
    if prediction_result == data["true_class"]:
        correct_predictions += 1
    print(f"Application {idx + 1}: Prediction = {prediction_result}, True class = {data['true_class']}")

accuracy = correct_predictions / total_predictions
print(f"\nAccuracy: {accuracy * 100:.2f}%")