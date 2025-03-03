import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Load the trained Decision Tree model
model_path = "decision_tree_churn_model.joblib"
best_model = load(model_path)

# Extract the correct feature order from the model
expected_columns = best_model.feature_names_in_.tolist()

def predict_churn(input_data):
    """ Predict churn based on input features. """
    X_new = pd.DataFrame([input_data])

    # Reorder columns to match model training
    X_new = X_new[expected_columns]

    # Predict churn and probabilities
    prediction = best_model.predict(X_new)[0]
    probability = best_model.predict_proba(X_new)[0][1]  # Probability of churn

    return "Churn" if prediction == 1 else "No Churn", probability

# 🎨 Streamlit UI Customization
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction")
st.markdown("Use this tool to predict whether a customer will **churn** or **stay**.")

# 📌 Move input fields to the sidebar
st.sidebar.header("📋 Enter Customer Details")
senior_citizen = st.sidebar.radio("Is the customer a senior citizen?", ["Yes", "No"])
monthly_charges = st.sidebar.number_input("💰 Monthly Charges ($)", min_value=0.0, step=1.0)
total_charges = st.sidebar.number_input("💳 Total Charges ($)", min_value=0.0, step=1.0)

gender = st.sidebar.radio("🧑 Gender", ["Female", "Male"])
partner = st.sidebar.radio("💑 Has a partner?", ["Yes", "No"])
dependents = st.sidebar.radio("👶 Has dependents?", ["Yes", "No"])

payment_method = st.sidebar.selectbox("💳 Payment Method", [
    "Bank transfer (automatic)", "Credit card (automatic)",
    "Electronic check", "Mailed check"
])

tenure_range = st.sidebar.selectbox("📅 Tenure Range", [
    "1-12", "13-24", "25-36", "37-48", "49-60", "61-72"
])

contract_type = st.sidebar.selectbox("📝 Contract Type", ["Month-to-month", "One year", "Two year"])
device_protection = st.sidebar.radio("🔒 Device Protection", ["Yes", "No", "No internet service"])
internet_service = st.sidebar.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
multiple_lines = st.sidebar.selectbox("📞 Multiple Lines", ["No", "No phone service", "Yes"])
paperless_billing = st.sidebar.radio("📄 Uses Paperless Billing?", ["Yes", "No"])
phone_service = st.sidebar.radio("📱 Has Phone Service?", ["Yes", "No"])
online_backup = st.sidebar.selectbox("💾 Online Backup", ["No", "No internet service", "Yes"])
online_security = st.sidebar.selectbox("🔐 Online Security", ["No", "No internet service", "Yes"])
tech_support = st.sidebar.selectbox("🛠 Tech Support", ["No", "No internet service", "Yes"])
streaming_tv = st.sidebar.selectbox("📺 Streaming TV", ["No", "No internet service", "Yes"])
streaming_movies = st.sidebar.selectbox("🎥 Streaming Movies", ["No", "No internet service", "Yes"])

# Convert User Input to Model Features
input_data = {
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
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
    "Contract_Month-to-month": contract_type == "Month-to-month",
    "Contract_One year": contract_type == "One year",
    "Contract_Two year": contract_type == "Two year",
    "DeviceProtection_No": device_protection == "No",
    "DeviceProtection_No internet service": device_protection == "No internet service",
    "DeviceProtection_Yes": device_protection == "Yes",
    "InternetService_DSL": internet_service == "DSL",
    "InternetService_Fiber optic": internet_service == "Fiber optic",
    "InternetService_No": internet_service == "No",
    "MultipleLines_No": multiple_lines == "No",
    "MultipleLines_No phone service": multiple_lines == "No phone service",
    "MultipleLines_Yes": multiple_lines == "Yes",
    "OnlineBackup_No": online_backup == "No",
    "OnlineBackup_No internet service": online_backup == "No internet service",
    "OnlineBackup_Yes": online_backup == "Yes",
    "OnlineSecurity_No": online_security == "No",
    "OnlineSecurity_No internet service": online_security == "No internet service",
    "OnlineSecurity_Yes": online_security == "Yes",
    "TechSupport_No": tech_support == "No",
    "TechSupport_No internet service": tech_support == "No internet service",
    "TechSupport_Yes": tech_support == "Yes",
    "StreamingTV_No": streaming_tv == "No",
    "StreamingTV_No internet service": streaming_tv == "No internet service",
    "StreamingTV_Yes": streaming_tv == "Yes",
    "StreamingMovies_No": streaming_movies == "No",
    "StreamingMovies_No internet service": streaming_movies == "No internet service",
    "StreamingMovies_Yes": streaming_movies == "Yes",
    "PaperlessBilling_No": paperless_billing == "No",
    "PaperlessBilling_Yes": paperless_billing == "Yes",
    "PhoneService_No": phone_service == "No",
    "PhoneService_Yes": phone_service == "Yes"
}

# Predict Button
if st.sidebar.button("🔮 Predict Churn"):
    prediction, probability = predict_churn(input_data)

    st.markdown(f"## **Prediction: {'🛑 The Customer will Churn' if prediction == 'Churn' else '✅ The Customer will stay'}**")
    if prediction == 'Churn':
        st.progress(probability)  # Show probability as progress bar
        st.write(f"📊 Probability of churn: **{probability:.2%}**")
