import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="centered"
)

# Load model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Loan Approval Prediction System")

st.write("---")

# Sidebar for inputs (professional UI style)
st.sidebar.header("Applicant Information")

st.write("Enter applicant details")

Credit_History = st.selectbox("Credit History", [0, 1])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)

#if st.button("Predict"):
if st.button("Check Eligibility"):
    input_data = pd.DataFrame({
        'Credit_History': [Credit_History],
        'ApplicantIncome': [ApplicantIncome],
        'LoanAmount': [LoanAmount],
        'CoapplicantIncome': [CoapplicantIncome],
        'Loan_Amount_Term': [Loan_Amount_Term]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")