
import joblib
import pandas as pd

import xailens as xai
from examples import config
from examples.helpers import HEART_COLUMN_MAPPINGS

raw = pd.read_parquet('./cache/heart/baseline/raw_instances.parquet')
X_test = pd.read_parquet('./cache/heart/baseline/X_test.parquet')
X_train = pd.read_parquet('./cache/heart/baseline/X_train.parquet')
y_train = pd.read_parquet('./cache/heart/baseline/y_train.parquet')
y_pred = pd.read_parquet('./cache/heart/baseline/y_pred.parquet')
y_test = pd.read_parquet('./cache/heart/baseline/y_test.parquet')
model = joblib.load('./cache/heart/baseline/model.joblib')

for col, mapping in HEART_COLUMN_MAPPINGS.items():
    raw[col] = raw[col].map(mapping)

data_schema = xai.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="y_pred")

model_data = xai.ModelData(
    data_schema=data_schema,
    raw_instances=raw,
    x_train=X_train,
    x_test=X_test,
    y_train=y_train,
    y_pred=y_pred,
    y_test=y_test)

ctx = xai.ModelContext(
    "logistic_regression",
    ["coefficients"],
    "cleveland-heart-data",
    model_data,
    model=model,
    model_dir_name="heart_lr",
    model_display_name="Logistic Regression",)

xai.run(ctx)