import joblib
import pandas as pd

import xai_compare as xai


X_train = pd.read_parquet('./cache/heart/xgboost/X_train.parquet')
X_test = pd.read_parquet('./cache/heart/xgboost/X_test.parquet')
y_train = pd.read_parquet('./cache/heart/xgboost/y_train.parquet')
y_test = pd.read_parquet('./cache/heart/xgboost/y_test.parquet')
y_pred = pd.read_parquet('./cache/heart/xgboost/y_pred.parquet')
model = joblib.load('./cache/heart/xgboost/model.joblib')


run_data = xai.RunData(
                   x_train=X_train,
                   x_test=X_test,
                   y_train=y_train,
                   y_test=y_test,
                   y_pred=y_pred)

ctx = xai.RunContext("xgboost",
               ["shap_tree"],
               "cleveland-heart-data",
               run_data,
               model=model)
xai.run(ctx)