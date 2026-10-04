import streamlit as st

from data_drift import get_data_drift_report
from concept_drift import get_concept_drift_report
from shap_explainability import get_shap_explanation
from ml_pipeline import run_pipeline


st.set_page_config(
    page_title="Loan Model Monitoring",
    layout="wide"
)

st.title("Loan Prediction Model Monitoring Dashboard")


# =========================
# MODEL PERFORMANCE
# =========================

st.header("Model Performance")

result, accuracy = run_pipeline()

predictions = len(result)

approval_rate = (
    result["prediction"].mean()
)

rejection_rate = 1 - approval_rate

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.0%}"
)

col2.metric(
    "Predictions",
    predictions
)

col3.metric(
    "Approval Rate",
    f"{approval_rate:.0%}"
)

col4.metric(
    "Rejection Rate",
    f"{rejection_rate:.0%}"
)


# =========================
# DATA DRIFT
# =========================

st.header("Data Drift")

drift_data, drift_found = (
    get_data_drift_report()
)

st.dataframe(
    drift_data,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    drift_data.set_index("Feature")["PSI"]
)

if drift_found:
    st.error("HIGH DATA DRIFT DETECTED")
else:
    st.success("NO SIGNIFICANT DATA DRIFT")


# =========================
# CONCEPT DRIFT
# =========================

st.header("Concept Drift")

concept_data, concept_status = (
    get_concept_drift_report()
)

st.dataframe(
    concept_data,
    use_container_width=True,
    hide_index=True
)

if concept_status == "HIGH CONCEPT DRIFT":
    st.error(concept_status)

elif concept_status == "MODERATE CONCEPT DRIFT":
    st.warning(concept_status)

else:
    st.success(concept_status)


# =========================
# SHAP
# =========================

st.header("Model Interpretability")

prediction, probability, shap_data = (
    get_shap_explanation()
)

st.metric(
    "Approval Probability",
    f"{probability:.2%}"
)

st.dataframe(
    shap_data,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    shap_data.set_index("Feature")["SHAP Value"]
)


# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Loan Prediction ML Monitoring System"
)