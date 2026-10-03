from fastapi import FastAPI, Request
import joblib
import logging
import time


# =========================================================
# Logging
# =========================================================

logger = logging.getLogger("uvicorn.error")


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Loan Prediction API",
    description="ML API for predicting loan approval",
    version="1.0.0"
)


# =========================================================
# Request Logging Middleware
# =========================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"REQUEST: {request.method} {request.url.path} | "
        f"STATUS: {response.status_code} | "
        f"TIME: {process_time:.4f}s"
    )

    return response


# =========================================================
# Load ML Model
# =========================================================

logger.info("Loading ML model...")

model = joblib.load("model/loan_model.pkl")

logger.info("ML model loaded successfully")


# =========================================================
# Home Endpoint
# =========================================================

@app.get("/")
def home():

    logger.info("HOME endpoint accessed")

    return {
        "message": "Loan Prediction API is running"
    }


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post("/predict")
def predict(
    age: int,
    income: float,
    loan_amount: float,
    credit_score: int
):

    logger.info(
        f"PREDICTION REQUEST | "
        f"age={age}, "
        f"income={income}, "
        f"loan_amount={loan_amount}, "
        f"credit_score={credit_score}"
    )

    # Prepare input
    data = [[
        age,
        income,
        loan_amount,
        credit_score
    ]]

    # Prediction
    prediction = model.predict(data)[0]

    # Result
    if prediction == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    logger.info(
        f"PREDICTION RESULT | "
        f"prediction={int(prediction)} | "
        f"result={result}"
    )

    return {
        "prediction": int(prediction),
        "message": result
    }