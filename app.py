import streamlit as st
from prediction import predict

st.title('Welcome to Customer Churn Prediction')
st.markdown('Churn generally means people leaving or stopping a service, product, or organization. This model helps you predict customer Churn.')
st.markdown('Input the following features to get your prediction.')

st.header('Churn Features')
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    PhoneService = st.selectbox("Phone Service", ["Yes","No"])

with col2:
    DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])

with col3:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    prediction = predict(input_data)
    st.success(f"Prediction: {prediction}")