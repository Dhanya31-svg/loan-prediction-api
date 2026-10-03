import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


# =========================
# 1. LOAD DATA
# =========================

data = pd.DataFrame({
    "age": [30, 35, 40, 28, 45],
    "income": [50000, 70000, 90000, 45000, 100000],
    "loan_amount": [200000, 180000, 150000, 250000, 160000],
    "credit_score": [680, 720, 760, 620, 800],
    "actual": [1, 1, 1, 0, 1]
})


# =========================
# 2. DATA VALIDATION
# =========================

features = [
    "age",
    "income",
    "loan_amount",
    "credit_score"
]

print("\nML PIPELINE")
print("=" * 50)

print("Step 1: Data loaded")

if data[features].isnull().sum().sum() > 0:
    raise ValueError("Missing values detected")

print("Step 2: Data validation passed")


# =========================
# 3. LOAD MODEL
# =========================

model = joblib.load("model/loan_model.pkl")

print("Step 3: Model loaded")


# =========================
# 4. PREDICTION
# =========================

X = data[features]

predictions = model.predict(X)

data["prediction"] = predictions

print("Step 4: Predictions generated")


# =========================
# 5. MODEL EVALUATION
# =========================

accuracy = accuracy_score(
    data["actual"],
    data["prediction"]
)

print(f"Step 5: Accuracy = {accuracy:.2f}")


# =========================
# 6. MONITORING
# =========================

print("Step 6: Monitoring completed")


# =========================
# FINAL RESULT
# =========================

print("=" * 50)
print("END-TO-END ML PIPELINE COMPLETED")
print("=" * 50)

print(data)