import pandas as pd
import numpy as np


FEATURES = [
    "age",
    "income",
    "loan_amount",
    "credit_score"
]


# Training/reference data
training_data = pd.DataFrame({
    "age": [22, 25, 30, 35, 40, 45, 28, 32, 50, 27],
    "income": [25000, 35000, 50000, 60000, 80000,
               90000, 45000, 55000, 100000, 40000],
    "loan_amount": [300000, 250000, 200000, 180000, 150000,
                    200000, 300000, 220000, 180000, 280000],
    "credit_score": [580, 620, 680, 720, 760,
                     780, 640, 700, 800, 600]
})


# Simulated production data
production_data = pd.DataFrame({
    "age": [30, 35, 40, 42, 45, 38, 36, 41, 44, 39],
    "income": [90000, 120000, 150000, 110000, 180000,
               130000, 125000, 160000, 175000, 140000],
    "loan_amount": [150000, 180000, 200000, 170000, 160000,
                    190000, 210000, 180000, 150000, 200000],
    "credit_score": [780, 800, 820, 790, 850,
                     810, 830, 800, 840, 820]
})


def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )

    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return psi


print("\nDATA DRIFT REPORT")
print("=" * 50)

drift_found = False

for feature in FEATURES:

    psi = calculate_psi(
        training_data[feature],
        production_data[feature]
    )

    print(f"{feature:15} PSI = {psi:.4f}")

    if psi >= 0.25:
        print("                 STATUS = HIGH DRIFT")
        drift_found = True

    elif psi >= 0.10:
        print("                 STATUS = MODERATE DRIFT")

    else:
        print("                 STATUS = NO SIGNIFICANT DRIFT")


print("=" * 50)

if drift_found:
    print("ALERT: Significant data drift detected!")
else:
    print("No significant data drift detected.")