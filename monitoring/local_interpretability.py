import joblib
import pandas as pd


# Load model
model = joblib.load("model/loan_model.pkl")


# One customer
customer = pd.DataFrame({
    "age": [30],
    "income": [50000],
    "loan_amount": [200000],
    "credit_score": [700]
})


# Prediction
prediction = model.predict(customer)[0]

probability = model.predict_proba(customer)[0]

# Model coefficients
coefficients = model.coef_[0]

# Contribution of each feature
contributions = customer.iloc[0].values * coefficients

explanation = pd.DataFrame({
    "feature": customer.columns,
    "value": customer.iloc[0].values,
    "coefficient": coefficients,
    "contribution": contributions
})

explanation["absolute_contribution"] = abs(
    explanation["contribution"]
)

explanation = explanation.sort_values(
    by="absolute_contribution",
    ascending=False
)


print("\nLOCAL MODEL INTERPRETABILITY")
print("=" * 60)

print(f"Prediction: {prediction}")

if prediction == 1:
    print("Decision: Loan Approved")
else:
    print("Decision: Loan Rejected")

print(f"Probability of Rejection: {probability[0]:.4f}")
print(f"Probability of Approval: {probability[1]:.4f}")

print("\nFeature Contributions")
print("-" * 60)

print(
    explanation[
        ["feature", "value", "coefficient", "contribution"]
    ].to_string(index=False)
)

print("=" * 60)
print("Local interpretability completed.")