import joblib
import pandas as pd


# Load model
model = joblib.load("model/loan_model.pkl")


# Feature names
features = [
    "age",
    "income",
    "loan_amount",
    "credit_score"
]


# Get model coefficients
coefficients = model.coef_[0]


# Create feature importance table
importance = pd.DataFrame({
    "feature": features,
    "coefficient": coefficients,
    "absolute_importance": abs(coefficients)
})


# Sort by importance
importance = importance.sort_values(
    by="absolute_importance",
    ascending=False
)


print("\nMODEL INTERPRETABILITY")
print("=" * 50)

print(importance.to_string(index=False))

print("=" * 50)
print("Interpretability analysis completed.")