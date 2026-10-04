from feast import FeatureStore

store = FeatureStore(repo_path="feature_store")

features = store.get_online_features(
    features=[
        "loan_features:age",
        "loan_features:income",
        "loan_features:loan_amount",
        "loan_features:credit_score",
    ],
    entity_rows=[
        {"customer_id": 1}
    ],
).to_dict()

print("\nFEATURE STORE RESULT")
print("=" * 50)

for key, value in features.items():
    print(f"{key}: {value}")

print("=" * 50)
