import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("insurance_data.csv")

print("Insurance Dataset Loaded Successfully!")

# Select Features
X = data[[
    "age",
    "policy_years",
    "claim_amount",
    "previous_claims"
]]

y = data["claim_approved"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Check Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("Model Accuracy:", accuracy * 100, "%")

# Customer Input
print("\n--- Customer Information ---")

age = int(input("Enter Customer Age: "))
policy_years = int(input("Enter Policy Years: "))
claim_amount = float(input("Enter Claim Amount: "))
previous_claims = int(input("Enter Previous Claims: "))

# Predict Claim
customer = pd.DataFrame([[
    age,
    policy_years,
    claim_amount,
    previous_claims
]], columns=[
    "age",
    "policy_years",
    "claim_amount",
    "previous_claims"
])

prediction = model.predict(customer)

print("\n--- Insurance Claim Result ---")

if prediction[0] == 1:
    print("Insurance Claim Approved")
else:
    print("Insurance Claim Not Approved")

print("\nPrediction Completed Successfully!")