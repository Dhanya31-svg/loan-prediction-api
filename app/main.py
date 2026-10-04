from fastapi import FastAPI
import joblib
import logging
import sys


# -----------------------------------------
# Configure application logging
# -----------------------------------------

logger = logging.getLogger("loan_api")
logger.setLevel(logging.INFO)

# Send logs directly to Docker stdout
handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

handler.setFormatter(formatter)

logger.addHandler(handler)

# Prevent duplicate logging
logger.propagate = False


# -----------------------------------------
# Create FastAPI application
# -----------------------------------------

app = FastAPI()


# -----------------------------------------
# Load trained ML model
# -----------------------------------------

model = joblib.load("model/loan_model.pkl")


# -----------------------------------------
# Home endpoint
# -----------------------------------------

@app.get("/")
def home():

    logger.info("Home endpoint accessed")

    return {
        "message": "Loan Prediction API is running"
    }


# -----------------------------------------
# Prediction endpoint
# -----------------------------------------

@app.post("/predict")
def predict(
    age: int,
    income: float,
    loan_amount: float,
    credit_score: int
):

    logger.info("Prediction request received")

    data = [[
        age,
        income,
        loan_amount,
        credit_score
    ]]

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    logger.info(
        f"Prediction completed: {result}"
    )

    return {
        "prediction": int(prediction),
        "message": result
    }