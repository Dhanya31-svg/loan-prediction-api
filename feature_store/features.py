from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource


loan_source = FileSource(
    name="loan_features_source",
    path="data/loan_features.parquet",
    timestamp_field="event_timestamp",
)

customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Unique loan customer identifier",
)

loan_features = FeatureView(
    name="loan_features",
    entities=[customer],
    ttl=None,
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="loan_amount", dtype=Float32),
        Field(name="credit_score", dtype=Int64),
    ],
    source=loan_source,
)