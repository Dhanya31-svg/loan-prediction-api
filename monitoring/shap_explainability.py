import joblib
import pandas as pd
import shap

model = joblib.load("model/loan_model.pkl")

X = pd.DataFrame({
    "age": [22, 25, 30, 35, 40, 45, 28, 32, 50, 27],
    "income": [25000, 35000, 50000, 60000, 80000,
               90000, 45000, 55000, 100000, 40000],
    "loan_amount": [300000, 250000, 200000, 180000, 150000,
                    200000, 300000, 220000, 180000, 280000],
    "credit_score": [580, 620, 680, 720, 760,
                     780, 640, 700, 800, 600]
})

customer = pd.DataFrame({
    "age": [30],
    "income": [50000],
    "loan_amount": [200000],
    "credit_score": [700]
})


def get_shap_explanation():

    explainer = shap.LinearExplainer(model, X)

    shap_values = explainer(customer)

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    explanation = pd.DataFrame({
        "Feature": customer.columns,
        "Value": customer.iloc[0].values,
        "SHAP Value": shap_values.values[0]
    })

    return prediction, probability, explanation


if __name__ == "__main__":

    prediction, probability, explanation = get_shap_explanation()

    print("\nSHAP MODEL INTERPRETABILITY")
    print("=" * 60)

    print(f"Prediction: {prediction}")
    print(f"Approval Probability: {probability:.4f}")

    print("\nSHAP Contributions")
    print("-" * 60)

    print(explanation.to_string(index=False))

    print("=" * 60)
    print("SHAP explanation completed.")