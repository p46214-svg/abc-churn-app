"""
ABC Ltd. - Customer Churn Predictor (simple version)

Uses the two models trained in ABC_Ltd_Predictive_Analytics.ipynb:
    models/logistic_churn_model.pkl        -> churn probability
    models/linear_monthly_charge_model.pkl -> expected monthly charge
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

MODEL_DIR = Path(__file__).parent / "models"

LOGISTIC_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
LINEAR_FEATURES = LOGISTIC_FEATURES[:-2]  # linear model has no MonthlyCharges / TotalCharges


@st.cache_resource
def load_models():
    logistic = joblib.load(MODEL_DIR / "logistic_churn_model.pkl")
    linear = joblib.load(MODEL_DIR / "linear_monthly_charge_model.pkl")
    return logistic, linear


def classify_risk(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


try:
    logistic_model, linear_model = load_models()
except FileNotFoundError:
    st.error("Model files not found. Keep both .pkl files inside a folder called `models`.")
    st.stop()

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
st.title("Customer Churn Predictor")
st.write("Fill in customer details to predict if they will churn (leave).")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

phone = st.selectbox("Phone Service", ["Yes", "No"])
if phone == "Yes":
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
else:
    multiple_lines = "No phone service"

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
add_on_names = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}
add_ons = {}
for col, label in add_on_names.items():
    if internet == "No":
        add_ons[col] = "No internet service"
    else:
        add_ons[col] = st.selectbox(label, ["No", "Yes"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
total = st.number_input(
    "Total Charges",
    min_value=0.0, max_value=10000.0,
    value=float(round(tenure * monthly, 2)),
    step=10.0,
    help="Defaults to tenure × monthly charges. Change it if you know the actual amount.",
)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("Predict", type="primary"):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        **add_ons,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])[LOGISTIC_FEATURES]

    probability = logistic_model.predict_proba(customer)[0][1]
    will_churn = logistic_model.predict(customer)[0] == 1
    risk = classify_risk(probability)
    expected_charge = linear_model.predict(customer[LINEAR_FEATURES])[0]

    st.subheader("Result")
    if will_churn:
        st.error(f"This customer is likely to CHURN ({probability:.1%} probability).")
    else:
        st.success(f"This customer is likely to STAY ({probability:.1%} churn probability).")

    st.write(f"**Risk category:** {risk}  (Low < 30%, Medium 30–60%, High ≥ 60%)")
    st.write(f"**Expected monthly charge for this plan:** {expected_charge:.2f}")

# ------------------------------------------------------------------
# Model accuracy (test-set results from the notebook, 80/20 split)
# ------------------------------------------------------------------
st.divider()
st.subheader("Model Accuracy")
st.caption("Measured on the 20% test set (1,407 customers) held back during training.")

left, right = st.columns(2)
with left:
    st.write("**Churn model** (Logistic Regression)")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Value": ["80.53%", "65.15%", "57.49%", "0.6108", "0.8361"],
        }),
        hide_index=True,
        width="stretch",
    )
with right:
    st.write("**Monthly charge model** (Linear Regression)")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["R²", "MAE", "RMSE"],
            "Value": ["0.9988", "0.78", "1.04"],
        }),
        hide_index=True,
        width="stretch",
    )
