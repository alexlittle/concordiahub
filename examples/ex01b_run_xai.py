
import joblib
import pandas as pd

import xailens as xai
from examples import config

X = pd.read_parquet('./cache/heart/baseline/X.parquet')
y = pd.read_parquet('./cache/heart/baseline/y.parquet')
y_pred = pd.read_parquet('./cache/heart/baseline/y_pred.parquet')
model = joblib.load('./cache/heart/baseline/model.joblib')

data_schema = xai.DataSchema(
    sample_id_column=config.SAMPLE_UUID_COL,
    target_column="target",
    prediction_column="y_pred")

run_data = xai.RunData(
    data_schema=data_schema,
    x=X,
    y=y,
    y_pred=y_pred)

ctx = xai.RunContext(
    "logistic_regression",
    ["coefficients"],
    "cleveland-heart-data",
    run_data,
    model=model,
    run_name="heart_lr",
    model_id="Example 1")

xai.run(ctx)