import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Sample loan data
data = pd.DataFrame({
    "age": [22, 25, 30, 35, 40, 45, 28, 32, 50, 27],
    "income": [25000, 35000, 50000, 60000, 80000,
               90000, 45000, 55000, 100000, 40000],
    "loan_amount": [300000, 250000, 200000, 180000, 150000,
                    200000, 300000, 220000, 180000, 280000],
    "credit_score": [580, 620, 680, 720, 760,
                     780, 640, 700, 800, 600],
    "approved": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
})

# Input columns
X = data[["age", "income", "loan_amount", "credit_score"]]

# Target column
y = data["approved"]

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X, y)

# Create model folder
import os
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/loan_model.pkl")

print("Model trained successfully!")
print("Model saved successfully!")