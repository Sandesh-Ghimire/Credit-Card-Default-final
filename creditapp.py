import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# LOAD MODEL

model = joblib.load("model.pkl")

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.title("Credit Default Risk Prediction Web App")
st.write("Enter customer details to predict default risk.")


# USER INPUT


LIMIT_BAL = st.number_input("Credit Limit", min_value=1000, max_value=1000000, value=50000)
SEX = st.selectbox("Gender: Male=1, Female=2", [1, 2])  # 1=Male, 2=Female
EDUCATION = st.selectbox("Education Level: Graduate School=1, University=2, High School=3, Others=4", [1,2,3,4])
MARRIAGE = st.selectbox("Marital Status: Single=1, Married=2, Divorced=3", [1,2,3])
AGE = st.slider("Age in Years", 18, 80, 30)
st.subheader("Repayment Status (PAY_0 to PAY_6)")
st.write("""
| Value | Meaning |
| --- | --- |
| -2 | No consumption |
| -1 | Paid duly |
| 0 | Paid on time |
| 1 | 1 month delay |
| 2 | 2 months delay |
| 3 | 3 months delay |
| 4+ | 4 months + delay |
""")

PAY_0 = st.slider("Repayment Status lastest month", -2, 8, 0)
PAY_2 = st.slider("Repayment Status 1 month before", -2, 8, 0)
PAY_3 = st.slider("Repayment Status 2 months before", -2, 8, 0)
PAY_4 = st.slider("Repayment Status 3 months before", -2, 8, 0)
PAY_5 = st.slider("Repayment Status 4 months before", -2, 8, 0)
PAY_6 = st.slider("Repayment Status 5 months before", -2, 8, 0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bill Amounts (BILL_AMT1 to BILL_AMT6)")
    BILL_AMT1 = st.number_input("Bill Amount latest month", value=0)
    BILL_AMT2 = st.number_input("Bill Amount 1 month before", value=0)
    BILL_AMT3 = st.number_input("Bill Amount 2 months before", value=0)
    BILL_AMT4 = st.number_input("Bill Amount 3 months before", value=0)
    BILL_AMT5 = st.number_input("Bill Amount 4 months before", value=0)
    BILL_AMT6 = st.number_input("Bill Amount 5 months before", value=0)

with col2:
    st.subheader("Payments (PAY_AMT1 to PAY_AMT6)")
    PAY_AMT1 = st.number_input("Payment latest month", value=0)
    PAY_AMT2 = st.number_input("Payment 1 month before", value=0)
    PAY_AMT3 = st.number_input("Payment 2 months before", value=0)
    PAY_AMT4 = st.number_input("Payment 3 months before", value=0)
    PAY_AMT5 = st.number_input("Payment 4 months before", value=0)
    PAY_AMT6 = st.number_input("Payment 5 months before", value=0)


# FEATURE ENGINEERING (same as training)

TOTAL_BILL = BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6
TOTAL_PAY = PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6
PAY_RATIO = TOTAL_PAY / (TOTAL_BILL + 1)
AVG_DELAY = np.mean([PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6])


# CREATE INPUT DATAFRAME


input_data = pd.DataFrame({
    'LIMIT_BAL': [LIMIT_BAL],
    'SEX': [SEX],
    'EDUCATION': [EDUCATION],
    'MARRIAGE': [MARRIAGE],
    'AGE': [AGE],
    'PAY_0': [PAY_0],
    'PAY_2': [PAY_2],
    'PAY_3': [PAY_3],
    'PAY_4': [PAY_4],
    'PAY_5': [PAY_5],
    'PAY_6': [PAY_6],
    'BILL_AMT1': [BILL_AMT1],
    'BILL_AMT2': [BILL_AMT2],
    'BILL_AMT3': [BILL_AMT3],
    'BILL_AMT4': [BILL_AMT4],
    'BILL_AMT5': [BILL_AMT5],
    'BILL_AMT6': [BILL_AMT6],
    'PAY_AMT1': [PAY_AMT1],
    'PAY_AMT2': [PAY_AMT2],
    'PAY_AMT3': [PAY_AMT3],
    'PAY_AMT4': [PAY_AMT4],
    'PAY_AMT5': [PAY_AMT5],
    'PAY_AMT6': [PAY_AMT6],
    'TOTAL_BILL': [TOTAL_BILL],
    'TOTAL_PAY': [TOTAL_PAY],
    'PAY_RATIO': [PAY_RATIO],
    'AVG_DELAY': [AVG_DELAY]
})

# PREDICTION

if st.button("Predict Risk"):
    prob = model.predict_proba(input_data)[0][1]
    pred = int(prob > 0.5)

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f" High Risk of Default (Probability: {prob:.2f})")
    else:
        st.success(f" Low Risk (Probability: {prob:.2f})")

    # Risk level classification
    if prob < 0.25:
        risk = "Low Risk"
    elif prob < 0.5:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.info(f"Risk Category: {risk}")

    # SHAP Explanation
    st.subheader("Prediction Explanation (SHAP)")
    st.write("This plot shows how each feature contributed to the model's prediction.")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], show=False)
    st.pyplot(fig)
