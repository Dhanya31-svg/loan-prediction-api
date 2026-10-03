from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
output = BASE_DIR / "loan_features.parquet"

data = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],

    "age": [30, 35, 40, 28, 45],

    "income": [50000, 70000, 90000, 45000, 100000],

    "loan_amount": [200000, 180000, 150000, 250000, 160000],

    "credit_score": [680, 720, 760, 620, 800],

    "event_timestamp": pd.to_datetime([
        "2026-09-25",
        "2026-09-25",
        "2026-09-25",
        "2026-09-25",
        "2026-09-25"
    ])
})

data.to_parquet(output, index=False)

print(f"Feature data created: {output}")