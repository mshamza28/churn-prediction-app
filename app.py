import streamlit as st
import pandas as pd
from joblib import load

# Load the trained Decision Tree model
model_path = "decision_tree_churn_model.joblib"
best_model = load(model_path)

# Expected feature columns (matching training dataset)
expected_columns = [
    "SeniorCitizen", "MonthlyCharges", "TotalCharges",
    "gender_Female", "gender_Male", "Partner_No", "Partner_Yes",
    "Dependents_No", "Dependents_Yes",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "tenure_range_1-12", "tenure_range_13-24", "tenure_range_25-36",
    "tenure_range_37-48", "tenure_range_49-60", "tenure_range_61-72",
    
    # 🔴 Previously Missing Features (Now Added)
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "DeviceProtection_No", "DeviceProtection_No internet service", "DeviceProtection_Yes",
    
    # 🔴 Newly Missing Features (Now Added)
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "MultipleLines_No", "MultipleLines_No phone service", "MultipleLines_Yes"
]

def predict_churn(input_data):
    """ Predict churn based on input features. """
    X_new = pd.DataFrame([input_data])

    # Ensure all expected columns exist
    X_new = X_new.reindex(columns=expected_columns, fill_value=0)

    # Debugging: Print shape to check
    print(f"Expected columns: {len(expected_columns)}, Input data columns: {X_new.shape[1]}")

    # Make prediction
    prediction = best_model.predict(X_new)[0]
    return "Churn" if prediction == 1 else "No Churn"

# Streamlit App UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User Input Fields
senior_citizen = st.radio("Is the customer a senior citizen?", [0, 1])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

gender = st.radio("Gender", ["Female", "Male"])
partner = st.radio("Has a partner?", ["Yes", "No"])
dependents = st.radio("Has dependents?", ["Yes", "No"])

payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)", "Credit card (automatic)",
    "Electronic check", "Mailed check"
])

tenure_range = st.selectbox("Tenure Range", [
    "1-12", "13-24", "25-36", "37-48", "49-60", "61-72"
])

# 🔴 **New Inputs for More Missing Features**
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
device_protection = st.radio("Has Device Protection?", ["Yes", "No", "No internet service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])

# Convert User Input to Model Features
input_data = {
    "SeniorCitizen": senior_citizen,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender_Female": gender == "Female",
    "gender_Male": gender == "Male",
    "Partner_No": partner == "No",
    "Partner_Yes": partner == "Yes",
    "Dependents_No": dependents == "No",
    "Dependents_Yes": dependents == "Yes",
    "PaymentMethod_Bank transfer (automatic)": payment_method == "Bank transfer (automatic)",
    "PaymentMethod_Credit card (automatic)": payment_method == "Credit card (automatic)",
    "PaymentMethod_Electronic check": payment_method == "Electronic check",
    "PaymentMethod_Mailed check": payment_method == "Mailed check",
    "tenure_range_1-12": tenure_range == "1-12",
    "tenure_range_13-24": tenure_range == "13-24",
    "tenure_range_25-36": tenure_range == "25-36",
    "tenure_range_37-48": tenure_range == "37-48",
    "tenure_range_49-60": tenure_range == "49-60",
    "tenure_range_61-72": tenure_range == "61-72",
    
    # 🔴 **Previously Added Features**
    "Contract_Month-to-month": contract_type == "Month-to-month",
    "Contract_One year": contract_type == "One year",
    "Contract_Two year": contract_type == "Two year",
    "DeviceProtection_No": device_protection == "No",
    "DeviceProtection_No internet service": device_protection == "No internet service",
    "DeviceProtection_Yes": device_protection == "Yes",

    # 🔴 **Newly Added Features**
    "InternetService_DSL": internet_service == "DSL",
    "InternetService_Fiber optic": internet_service == "Fiber optic",
    "InternetService_No": internet_service == "No",
    "MultipleLines_No": multiple_lines == "No",
    "MultipleLines_No phone service": multiple_lines == "No phone service",
    "MultipleLines_Yes": multiple_lines == "Yes"
}

# Predict Button
if st.button("Predict Churn"):
    prediction = predict_churn(input_data)
    st.subheader(f"Prediction: {prediction}")
