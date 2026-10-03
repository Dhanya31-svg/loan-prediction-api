from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Loan Prediction API is running"
    }


def test_prediction():
    response = client.post(
        "/predict",
        params={
            "age": 30,
            "income": 50000,
            "loan_amount": 200000,
            "credit_score": 700
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "message" in result