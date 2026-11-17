import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page Settings
st.set_page_config(page_title="Insurance Claim Prediction")

st.title("Insurance Claim Prediction System")

st.write("Predict whether an insurance claim will be approved.")

# Load Dataset
data = pd.read_csv("insurance_data.csv")

# Prepare Data
X = data.drop("claim_approved", axis=1)
y = data["claim_approved"]

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.success("Machine Learning Model Loaded Successfully!")

# Customer Information
st.header("Enter Customer Information")

age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

policy_years = st.number_input(
    "Policy Years", min_value=1, max_value=50, value=5
)

claim_amount = st.number_input(
    "Claim Amount", min_value=1000, value=70000
)

previous_claims = st.number_input(
    "Previous Claims", min_value=0, max_value=20, value=1
)

# Prediction Button
if st.button("Predict Insurance Claim"):

    customer = pd.DataFrame(
        [[age, policy_years, claim_amount, previous_claims]],
        columns=X.columns
    )

    prediction = model.predict(customer)

    st.header("Prediction Result")

    if prediction[0] == 1:
        st.success("Insurance Claim Approved!")
    else:
        st.error("Insurance Claim Not Approved!")