import joblib
import pandas as pd

import xailens as xai
from examples import config
from helpers import ICD10_CHAPTERS


X_train = pd.read_parquet('./cache/medsynth/xgboost/X_train.parquet')
X_test = pd.read_parquet('./cache/medsynth/xgboost/X_test.parquet')
y_train = pd.read_parquet('./cache/medsynth/xgboost/y_train.parquet')
y_test = pd.read_parquet('./cache/medsynth/xgboost/y_test.parquet')
y_pred = pd.read_parquet('./cache/medsynth/xgboost/y_pred.parquet')
model = joblib.load('./cache/medsynth/xgboost/model.joblib')


y_train["y_train"] = y_train["y_train"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)
y_test["y_test"] = y_test["y_test"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)
y_pred["y_pred"] = y_pred["y_pred"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)


data_schema = xai.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="y_test",
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
    "medsynth",
    model_data,
    model=model,
    model_dir_name="medsynth_xgboost",
    model_display_name="XGBoost")

xai.run(ctx)