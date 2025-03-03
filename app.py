import streamlit as st
import pandas as pd
from joblib import load

# Load the trained Decision Tree model
model_path = "decision_tree_churn_model.joblib"
best_model = load(model_path)

# Define expected features for input
expected_columns = [
    "SeniorCitizen", "MonthlyCharges", "TotalCharges",
    "gender_Female", "gender_Male", "Partner_No", "Partner_Yes",
    "Dependents_No", "Dependents_Yes",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "tenure_range_1-12", "tenure_range_13-24", "tenure_range_25-36",
    "tenure_range_37-48", "tenure_range_49-60", "tenure_range_61-72"
]

def predict_churn(input_data):
    """ Predict churn based on input features. """
    X_new = pd.DataFrame([input_data])
    X_new = X_new.reindex(columns=expected_columns, fill_value=0)
    prediction = best_model.predict(X_new)[0]
    return "Churn" if prediction == 1 else "No Churn"

# Streamlit App UI
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User Input Fields
senior_citizen = st.selectbox("Is the customer a senior citizen?", [0, 1])
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, step=1.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, step=1.0)

gender = st.radio("Gender", ["Female", "Male"])
partner = st.radio("Has a partner?", ["Yes", "No"])
dependents = st.radio("Has dependents?", ["Yes", "No"])

payment_method = st.selectbox("Payment Method", [
    "Bank transfer (automatic)", "Credit card (automatic)",
    "Electronic check", "Mailed check"])

tenure_range = st.selectbox("Tenure Range", [
    "1-12", "13-24", "25-36", "37-48", "49-60", "61-72"])

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
    "tenure_range_61-72": tenure_range == "61-72"
}

# Predict Button
if st.button("Predict Churn"):
    prediction = predict_churn(input_data)
    st.subheader(f"Prediction: {prediction}")
