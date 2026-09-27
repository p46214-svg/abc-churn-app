"""
ABC Ltd. - Customer Churn & Monthly Charge Predictor
Streamlit app built on the two models trained in ABC_Ltd_Predictive_Analytics.ipynb

    models/logistic_churn_model.pkl        -> Logistic Regression (churn)
    models/linear_monthly_charge_model.pkl -> Linear Regression (monthly charges)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ABC Ltd. Customer Analytics",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# Columns expected by each model (same order as training)
LOGISTIC_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
LINEAR_FEATURES = [c for c in LOGISTIC_FEATURES if c not in ("MonthlyCharges", "TotalCharges")]

# Allowed values (taken from the training data)
OPTIONS = {
    "gender": ["Female", "Male"],
    "YesNo": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "AddOn": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
ADD_ON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Test-set results from the notebook
LOGISTIC_METRICS = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Value": [0.8053, 0.6515, 0.5749, 0.6108, 0.8361],
})
LINEAR_METRICS = pd.DataFrame({
    "Metric": ["R²", "MAE", "RMSE"],
    "Value": [0.9988, 0.7835, 1.0390],
})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    logistic = joblib.load(MODEL_DIR / "logistic_churn_model.pkl")
    linear = joblib.load(MODEL_DIR / "linear_monthly_charge_model.pkl")
    return logistic, linear


def classify_risk(probability: float) -> str:
    """Same thresholds as the notebook."""
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    return "High"


RISK_COLOURS = {"Low": "green", "Medium": "orange", "High": "red"}


def predict(df: pd.DataFrame, logistic, linear) -> pd.DataFrame:
    """Run both models on a dataframe that has all LOGISTIC_FEATURES."""
    X_log = df[LOGISTIC_FEATURES]
    X_lin = df[LINEAR_FEATURES]

    prob = logistic.predict_proba(X_log)[:, 1]
    churn = logistic.predict(X_log)
    charge = linear.predict(X_lin)

    return pd.DataFrame({
        "Churn Probability (%)": np.round(prob * 100, 2),
        "Predicted Churn": np.where(churn == 1, "Yes", "No"),
        "Risk Category": [classify_risk(p) for p in prob],
        "Predicted Monthly Charges": np.round(charge, 2),
    }, index=df.index)


def pretty_feature(name: str) -> str:
    """'categorical__Contract_Two year' -> 'Contract = Two year'."""
    if name.startswith("numerical__"):
        return name.replace("numerical__", "")
    name = name.replace("categorical__", "")
    for col in LOGISTIC_FEATURES:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def churn_drivers(customer: pd.DataFrame, logistic) -> pd.DataFrame:
    """
    Contribution of each input to the churn log-odds for one customer
    (coefficient x transformed value). Numbers are measured against an
    'average' customer in the baseline category of each variable.
    """
    pre = logistic.named_steps["preprocessor"]
    model = logistic.named_steps["model"]
    x = pre.transform(customer[LOGISTIC_FEATURES])
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    contrib = x * model.coef_[0]

    out = pd.DataFrame({
        "Factor": [pretty_feature(f) for f in pre.get_feature_names_out()],
        "Effect on churn log-odds": contrib,
    })
    out = out[out["Effect on churn log-odds"].abs() > 0.01]
    return out.sort_values("Effect on churn log-odds", key=np.abs, ascending=False)


def clean_uploaded(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the same cleaning as the notebook to an uploaded file."""
    notes = []
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"])
    dropped = before - len(df)
    if dropped:
        notes.append(f"{dropped} row(s) removed because a numeric field was blank or invalid.")

    for col in LOGISTIC_FEATURES:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df, notes


# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------
try:
    logistic_model, linear_model = load_models()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure the `models` folder with "
        "`logistic_churn_model.pkl` and `linear_monthly_charge_model.pkl` "
        "is in the same place as `app.py`."
    )
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("📊 ABC Ltd. – Customer Churn & Monthly Charge Predictor")
st.caption(
    "Logistic Regression estimates how likely a customer is to leave. "
    "Linear Regression estimates the monthly charge expected for the customer's service plan."
)

tab_single, tab_batch, tab_about = st.tabs(
    ["🔍 Single customer", "📁 Batch prediction (CSV)", "ℹ️ About the models"]
)

# ------------------------------------------------------------------
# Tab 1: single customer
# ------------------------------------------------------------------
with tab_single:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer profile")
        gender = st.selectbox("Gender", OPTIONS["gender"])
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Has partner", OPTIONS["YesNo"], horizontal=True)
        dependents = st.radio("Has dependents", OPTIONS["YesNo"], horizontal=True)
        tenure = st.slider("Tenure (months with ABC Ltd.)", 0, 72, 12)

    with col2:
        st.subheader("Services")
        phone = st.radio("Phone service", OPTIONS["YesNo"], index=1, horizontal=True)
        if phone == "No":
            multiple_lines = "No phone service"
            st.selectbox("Multiple lines", ["No phone service"], disabled=True)
        else:
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])

        internet = st.selectbox("Internet service", OPTIONS["InternetService"], index=1)

        add_ons = {}
        labels = {
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "TechSupport": "Tech support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
        }
        if internet == "No":
            st.info("No internet service – add-on services set to 'No internet service'.")
            for s in ADD_ON_SERVICES:
                add_ons[s] = "No internet service"
        else:
            a, b = st.columns(2)
            for i, s in enumerate(ADD_ON_SERVICES):
                with (a if i % 2 == 0 else b):
                    add_ons[s] = "Yes" if st.checkbox(labels[s]) else "No"

    with col3:
        st.subheader("Account & billing")
        contract = st.selectbox("Contract", OPTIONS["Contract"])
        paperless = st.radio("Paperless billing", OPTIONS["YesNo"], index=1, horizontal=True)
        payment = st.selectbox("Payment method", OPTIONS["PaymentMethod"])
        monthly = st.number_input(
            "Current monthly charges", min_value=0.0, max_value=200.0,
            value=70.0, step=0.5,
        )
        auto_total = st.checkbox("Calculate total charges as tenure × monthly charges", value=True)
        if auto_total:
            total = round(tenure * monthly, 2)
            st.number_input("Total charges", value=total, disabled=True)
        else:
            total = st.number_input(
                "Total charges", min_value=0.0, max_value=10000.0,
                value=round(tenure * monthly, 2), step=10.0,
            )

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

    st.divider()

    if st.button("Predict", type="primary", width="stretch"):
        result = predict(customer, logistic_model, linear_model).iloc[0]
        prob = result["Churn Probability (%)"]
        risk = result["Risk Category"]
        pred_charge = result["Predicted Monthly Charges"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn probability", f"{prob:.1f}%")
        m2.markdown(
            f"**Risk category**<br><span style='font-size:2rem;color:{RISK_COLOURS[risk]}'>"
            f"{risk}</span>",
            unsafe_allow_html=True,
        )
        m3.metric("Predicted to churn?", result["Predicted Churn"])
        m4.metric(
            "Expected monthly charge",
            f"{pred_charge:.2f}",
            delta=f"current bill {monthly - pred_charge:+.2f} vs expected",
            delta_color="off",
        )
        st.progress(min(int(round(prob)), 100))

        st.caption(
            "Risk bands: Low < 30%, Medium 30–60%, High ≥ 60%. "
            "'Predicted to churn' uses the model's default 50% cut-off. "
            "Expected monthly charge is what the Linear Regression model predicts for this "
            "service plan; the difference shows how far the current bill is from that."
        )

        with st.expander("What is driving this churn estimate?", expanded=True):
            drivers = churn_drivers(customer, logistic_model)
            up = drivers[drivers["Effect on churn log-odds"] > 0].head(5)
            down = drivers[drivers["Effect on churn log-odds"] < 0].head(5)
            c_up, c_down = st.columns(2)
            with c_up:
                st.markdown("**Pushing churn risk UP**")
                if up.empty:
                    st.write("None")
                else:
                    st.dataframe(up.round(3), hide_index=True, width="stretch")
            with c_down:
                st.markdown("**Pulling churn risk DOWN**")
                if down.empty:
                    st.write("None")
                else:
                    st.dataframe(down.round(3), hide_index=True, width="stretch")
            st.caption(
                "Each value is the model coefficient × this customer's (scaled) input. "
                "Numeric variables are compared with the average customer; categories are "
                "compared with the base category (e.g. Contract is compared with Month-to-month)."
            )
    else:
        st.info("Fill in the customer details above and click **Predict**.")

# ------------------------------------------------------------------
# Tab 2: batch prediction
# ------------------------------------------------------------------
with tab_batch:
    st.subheader("Score many customers at once")
    st.write(
        "Upload a CSV in the same format as the ABC Ltd. dataset. "
        "`customerID` and `Churn` columns are optional and are not used for prediction."
    )

    template = pd.DataFrame([{
        "customerID": "0001-ABCD", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }])
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="customer_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        raw.columns = [c.strip() for c in raw.columns]
        missing = [c for c in LOGISTIC_FEATURES if c not in raw.columns]
        if missing:
            st.error("These required columns are missing: " + ", ".join(missing))
        else:
            data, notes = clean_uploaded(raw)
            for n in notes:
                st.warning(n)

            if data.empty:
                st.error("No valid rows left to score.")
            else:
                preds = predict(data, logistic_model, linear_model)
                id_cols = [c for c in ["customerID"] if c in data.columns]
                output = pd.concat([data[id_cols + LOGISTIC_FEATURES], preds], axis=1)
                output = output.sort_values("Churn Probability (%)", ascending=False)

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Customers scored", f"{len(output):,}")
                k2.metric("High risk", f"{(output['Risk Category'] == 'High').sum():,}")
                k3.metric("Medium risk", f"{(output['Risk Category'] == 'Medium').sum():,}")
                k4.metric("Low risk", f"{(output['Risk Category'] == 'Low').sum():,}")
                k5.metric(
                    "Average churn probability",
                    f"{output['Churn Probability (%)'].mean():.1f}%",
                )

                show_cols = id_cols + [
                    "tenure", "Contract", "InternetService", "MonthlyCharges",
                    "Churn Probability (%)", "Risk Category", "Predicted Churn",
                    "Predicted Monthly Charges",
                ]
                st.dataframe(output[show_cols], hide_index=True, width="stretch")

                st.download_button(
                    "Download all predictions (CSV)",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="abc_ltd_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )

# ------------------------------------------------------------------
# Tab 3: about
# ------------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Logistic Regression – churn")
        st.write(
            "**Target:** Churn (Yes = 1, No = 0)  \n"
            "**Inputs:** all 19 customer variables, including MonthlyCharges and TotalCharges  \n"
            "**Training / test split:** 80 / 20, stratified, 7,032 customers"
        )
        st.dataframe(LOGISTIC_METRICS, hide_index=True, width="stretch")
    with a2:
        st.subheader("Linear Regression – monthly charges")
        st.write(
            "**Target:** MonthlyCharges  \n"
            "**Inputs:** 17 variables (TotalCharges and Churn excluded)  \n"
            "**Training / test split:** 80 / 20, 7,032 customers"
        )
        st.dataframe(LINEAR_METRICS, hide_index=True, width="stretch")

    st.subheader("How to read the results")
    st.markdown(
        "- **Churn probability** – the model's estimate that this customer will leave.\n"
        "- **Risk category** – Low (< 30%), Medium (30–60%), High (≥ 60%).\n"
        "- **Predicted to churn** – Yes if probability is 50% or more.\n"
        "- **Expected monthly charge** – what a customer with this service mix is typically billed. "
        "The model explains almost all variation in charges (R² = 0.999) because the bill is "
        "largely set by the services taken."
    )
