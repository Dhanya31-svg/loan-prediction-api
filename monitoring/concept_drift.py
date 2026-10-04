import pandas as pd
from sklearn.metrics import accuracy_score


historical_data = pd.DataFrame({
    "actual": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    "prediction": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
})


production_data = pd.DataFrame({
    "actual": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    "prediction": [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
})


def get_concept_drift_report():

    historical_accuracy = accuracy_score(
        historical_data["actual"],
        historical_data["prediction"]
    )

    production_accuracy = accuracy_score(
        production_data["actual"],
        production_data["prediction"]
    )

    accuracy_drop = (
        historical_accuracy - production_accuracy
    )

    if accuracy_drop >= 0.20:
        status = "HIGH CONCEPT DRIFT"

    elif accuracy_drop >= 0.10:
        status = "MODERATE CONCEPT DRIFT"

    else:
        status = "NO SIGNIFICANT CONCEPT DRIFT"

    report = pd.DataFrame({
        "Metric": [
            "Historical Accuracy",
            "Production Accuracy",
            "Accuracy Drop"
        ],
        "Value": [
            historical_accuracy,
            production_accuracy,
            accuracy_drop
        ]
    })

    return report, status


if __name__ == "__main__":

    report, status = get_concept_drift_report()

    print("\nCONCEPT DRIFT REPORT")
    print("=" * 50)
    print(report.to_string(index=False))
    print("=" * 50)
    print(f"STATUS = {status}")