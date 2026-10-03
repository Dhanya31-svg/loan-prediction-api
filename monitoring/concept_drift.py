import pandas as pd
from sklearn.metrics import accuracy_score


# Historical model predictions
historical_data = pd.DataFrame({
    "actual":    [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    "prediction": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
})


# Recent production predictions
production_data = pd.DataFrame({
    "actual":    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    "prediction": [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
})


historical_accuracy = accuracy_score(
    historical_data["actual"],
    historical_data["prediction"]
)

production_accuracy = accuracy_score(
    production_data["actual"],
    production_data["prediction"]
)

accuracy_drop = historical_accuracy - production_accuracy


print("\nCONCEPT DRIFT REPORT")
print("=" * 50)

print(f"Historical Accuracy : {historical_accuracy:.2f}")
print(f"Production Accuracy : {production_accuracy:.2f}")
print(f"Accuracy Drop       : {accuracy_drop:.2f}")

print("=" * 50)

if accuracy_drop >= 0.20:
    print("STATUS = HIGH CONCEPT DRIFT")
    print("ALERT: Model performance has significantly decreased!")

elif accuracy_drop >= 0.10:
    print("STATUS = MODERATE CONCEPT DRIFT")

else:
    print("STATUS = NO SIGNIFICANT CONCEPT DRIFT")