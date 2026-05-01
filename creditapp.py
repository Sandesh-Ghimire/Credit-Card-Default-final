import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# LOAD MODEL

model = joblib.load("model.pkl")

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-left: 12rem;
        padding-right: 12rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Credit Default Risk Prediction Web App")
st.write("Enter customer details to predict default risk.")


# USER INPUT


LIMIT_BAL = st.number_input("Credit Limit", min_value=1000, max_value=1000000, value=50000)

SEX = st.selectbox(
    "Gender",
    options=[1, 2],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

EDUCATION = st.selectbox(
    "Education Level",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Others"
    }[x]
)

MARRIAGE = st.selectbox(
    "Marital Status",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Married",
        2: "Single",
        3: "Others"
    }[x]
)

AGE = st.slider("Age in Years", 18, 80, 30)

rep_col1, rep_col2 = st.columns(2)

with rep_col1:
    st.subheader("Repayment Status Guide")
    st.write("""
| Value | Meaning |
|---|---|
| -2 | No consumption |
| -1 | Paid duly |
| 0 | Paid on time |
| 1 | 1 month delay |
| 2 | 2 months delay |
| 3 | 3 months delay |
| 4+ | 4 months or more delay |
""")

with rep_col2:
    PAY_0 = st.slider("Repayment Status latest month", -2, 8, 0)
    PAY_2 = st.slider("Repayment Status 1 month before", -2, 8, 0)
    PAY_3 = st.slider("Repayment Status 2 months before", -2, 8, 0)
    PAY_4 = st.slider("Repayment Status 3 months before", -2, 8, 0)
    PAY_5 = st.slider("Repayment Status 4 months before", -2, 8, 0)
    PAY_6 = st.slider("Repayment Status 5 months before", -2, 8, 0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Bill Amounts")
    BILL_AMT1 = st.number_input("Bill Amount latest month", value=0)
    BILL_AMT2 = st.number_input("Bill Amount 1 month before", value=0)
    BILL_AMT3 = st.number_input("Bill Amount 2 months before", value=0)
    BILL_AMT4 = st.number_input("Bill Amount 3 months before", value=0)
    BILL_AMT5 = st.number_input("Bill Amount 4 months before", value=0)
    BILL_AMT6 = st.number_input("Bill Amount 5 months before", value=0)

with col2:
    st.subheader("Payment Amounts")
    PAY_AMT1 = st.number_input("Payment latest month", value=0)
    PAY_AMT2 = st.number_input("Payment 1 month before", value=0)
    PAY_AMT3 = st.number_input("Payment 2 months before", value=0)
    PAY_AMT4 = st.number_input("Payment 3 months before", value=0)
    PAY_AMT5 = st.number_input("Payment 4 months before", value=0)
    PAY_AMT6 = st.number_input("Payment 5 months before", value=0)


# FEATURE ENGINEERING


TOTAL_BILL = BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6
TOTAL_PAY = PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6
PAY_RATIO = TOTAL_PAY / (TOTAL_BILL + 1)
AVG_DELAY = np.mean([PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6])

input_data = pd.DataFrame({
    "LIMIT_BAL": [LIMIT_BAL],
    "SEX": [SEX],
    "EDUCATION": [EDUCATION],
    "MARRIAGE": [MARRIAGE],
    "AGE": [AGE],
    "PAY_0": [PAY_0],
    "PAY_2": [PAY_2],
    "PAY_3": [PAY_3],
    "PAY_4": [PAY_4],
    "PAY_5": [PAY_5],
    "PAY_6": [PAY_6],
    "BILL_AMT1": [BILL_AMT1],
    "BILL_AMT2": [BILL_AMT2],
    "BILL_AMT3": [BILL_AMT3],
    "BILL_AMT4": [BILL_AMT4],
    "BILL_AMT5": [BILL_AMT5],
    "BILL_AMT6": [BILL_AMT6],
    "PAY_AMT1": [PAY_AMT1],
    "PAY_AMT2": [PAY_AMT2],
    "PAY_AMT3": [PAY_AMT3],
    "PAY_AMT4": [PAY_AMT4],
    "PAY_AMT5": [PAY_AMT5],
    "PAY_AMT6": [PAY_AMT6],
    "TOTAL_BILL": [TOTAL_BILL],
    "TOTAL_PAY": [TOTAL_PAY],
    "PAY_RATIO": [PAY_RATIO],
    "AVG_DELAY": [AVG_DELAY]
})


# PREDICTION


if st.button("Predict Risk"):
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prob < 0.30:
        risk = "Low Risk"
        st.success(f"Low Risk of Default (Probability: {prob:.2f})")
    elif prob < 0.50:
        risk = "Medium Risk"
        st.warning(f"Medium Risk of Default (Probability: {prob:.2f})")
    else:
        risk = "High Risk"
        st.error(f"High Risk of Default (Probability: {prob:.2f})")

    st.info(f"Risk Category: {risk}")

    tab1, tab2, tab3 = st.tabs(
        ["SHAP Explanation", "Model Performance", "Insights"]
    )

    # SHAP
   
    with tab1:
        st.subheader("SHAP Explanation")
        st.write("SHAP shows how each feature contributed to this prediction.")

        try:
            explainer = shap.TreeExplainer(model)
            explanation = explainer(input_data)

            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation[0], show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.warning("SHAP explanation could not be generated.")
            st.write(e)

   
    # ================================
    # MODEL PERFORMANCE
    # ================================
    with tab2:
        st.subheader("Model Performance Metrics")
        st.write("Final XGBoost model performance on the test set:")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Accuracy", "76.23%")
        with c2:
            st.metric("Precision", "47%")
        with c3:
            st.metric("Recall", "63%")
        with c4:
            st.metric("F1-Score", "54%")

        st.info("ROC-AUC Score: 0.779")
        st.caption("These metrics are computed from the held-out test dataset, not from the current individual input.")
        st.write("### Confusion Matrix")

        cm_table = pd.DataFrame(
            {
                "Predicted No Default": [3735, 488],
                "Predicted Default": [938, 839]
            },
            index=["Actual No Default", "Actual Default"]
        )

        st.table(cm_table)

    # INSIGHTS
  
    with tab3:
        st.subheader("Key Insights")

        st.write("""
### Main Factors Influencing Default Risk

1. **Repayment Status (PAY_0)**  
   Recent repayment delay is the strongest indicator of default risk.

2. **Average Delay (AVG_DELAY)**  
   Consistent delays across several months increase the likelihood of default.

3. **Payment Ratio (PAY_RATIO)**  
   A higher payment-to-bill ratio usually reduces default risk.

4. **Total Payment (TOTAL_PAY)**  
   Higher total payments indicate stronger repayment behaviour.

5. **Credit Limit (LIMIT_BAL)**  
   Credit limit provides additional context about the customer's financial profile.

### Decision Support Use

This system does not replace human judgement. It provides probability, risk category, and explanation to support credit analysts in making informed decisions.
""")