import joblib
import pandas as pd

import xailens as xai
from examples import config

X_train = pd.read_parquet('./cache/heart/xgboost/X_train.parquet')
X_test = pd.read_parquet('./cache/heart/xgboost/X_test.parquet')
y_train = pd.read_parquet('./cache/heart/xgboost/y_train.parquet')
y_test = pd.read_parquet('./cache/heart/xgboost/y_test.parquet')
y_pred = pd.read_parquet('./cache/heart/xgboost/y_pred.parquet')
model = joblib.load('./cache/heart/xgboost/model.joblib')

data_schema = xai.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="y_pred")

model_data = xai.ModelData(
    data_schema=data_schema,
    x_train=X_train,
    x_test=X_test,
    y_train=y_train,
    y_test=y_test,
    y_pred=y_pred)

ctx = xai.ModelContext(
    "xgboost",
    ["tree_shap"],
    "cleveland-heart-data",
    model_data,
    model=model,
    run_name="heart_xgboost",
    model_id="Example 2")

xai.run(ctx)