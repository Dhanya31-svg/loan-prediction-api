import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


features = [
    "age",
    "income",
    "loan_amount",
    "credit_score"
]


data = pd.DataFrame({
    "age": [30, 35, 40, 28, 45],
    "income": [50000, 70000, 90000, 45000, 100000],
    "loan_amount": [200000, 180000, 150000, 250000, 160000],
    "credit_score": [680, 720, 760, 620, 800],
    "actual": [1, 1, 1, 0, 1]
})


def run_pipeline():

    if data[features].isnull().sum().sum() > 0:
        raise ValueError("Missing values detected")

    model = joblib.load("model/loan_model.pkl")

    X = data[features]

    predictions = model.predict(X)

    result = data.copy()
    result["prediction"] = predictions

    accuracy = accuracy_score(
        result["actual"],
        result["prediction"]
    )

    return result, accuracy


if __name__ == "__main__":

    result, accuracy = run_pipeline()

    print("\nML PIPELINE")
    print("=" * 50)
    print(f"Accuracy = {accuracy:.2f}")
    print("=" * 50)
    print(result)