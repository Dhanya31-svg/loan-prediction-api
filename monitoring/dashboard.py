import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Loan Model Monitoring",
    layout="wide"
)

st.title("Loan Prediction Model Monitoring Dashboard")

# =========================
# MODEL PERFORMANCE
# =========================

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "80%")
col2.metric("Predictions", "5")
col3.metric("Approval Rate", "80%")
col4.metric("Rejection Rate", "20%")


# =========================
# DATA DRIFT
# =========================

st.header("Data Drift")

drift_data = pd.DataFrame({
    "Feature": [
        "age",
        "income",
        "loan_amount",
        "credit_score"
    ],
    "PSI": [
        6.9176,
        13.0718,
        3.8499,
        8.7741
    ],
    "Status": [
        "HIGH DRIFT",
        "HIGH DRIFT",
        "HIGH DRIFT",
        "HIGH DRIFT"
    ]
})

st.dataframe(
    drift_data,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    drift_data.set_index("Feature")["PSI"]
)

st.error("HIGH DATA DRIFT DETECTED")


# =========================
# CONCEPT DRIFT
# =========================

st.header("Concept Drift")

concept_data = pd.DataFrame({
    "Metric": [
        "Historical Accuracy",
        "Production Accuracy",
        "Accuracy Drop"
    ],
    "Value": [
        1.00,
        0.40,
        0.60
    ]
})

st.dataframe(
    concept_data,
    use_container_width=True,
    hide_index=True
)

st.error("HIGH CONCEPT DRIFT DETECTED")


# =========================
# SHAP
# =========================

st.header("Model Interpretability")

shap_data = pd.DataFrame({
    "Feature": [
        "age",
        "income",
        "loan_amount",
        "credit_score"
    ],
    "SHAP Value": [
        3.919541e-09,
        -93.30324,
        44.43356,
        -0.000651
    ]
})

st.dataframe(
    shap_data,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    shap_data.set_index("Feature")["SHAP Value"]
)

st.success("SHAP analysis available")


# =========================
# FOOTER
# =========================

st.divider()

st.caption("Loan Prediction ML Monitoring System")