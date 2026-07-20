"""
Script to register the XGBoost medsynth model with the XAI Lens Framework, from
pre-generated explanations
"""


import joblib
import pandas as pd

import xailens
from examples import config
from helpers import ICD10_CHAPTERS
from examples.helpers import  create_confusion_matrix, save_class_report
from sklearn.metrics import classification_report

raw = pd.read_parquet('./cache/medsynth/xgboost/raw_instances.parquet')
X_test = pd.read_parquet('./cache/medsynth/xgboost/X_test.parquet')
y_test = pd.read_parquet('./cache/medsynth/xgboost/y_test.parquet')
y_pred = pd.read_parquet('./cache/medsynth/xgboost/y_pred.parquet')
model = joblib.load('./cache/medsynth/xgboost/model.joblib')


y_test["y_test"] = y_test["y_test"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)
y_pred["y_pred"] = y_pred["y_pred"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)

report = classification_report(y_test["y_test"],
                               y_pred['y_pred'],
                               zero_division=True)
print(report)

save_class_report(report, "medsynth", "xgboost")
class_names = sorted(set(y_test["y_test"]) | set(y_pred["y_pred"]))
create_confusion_matrix(y_test["y_test"],
                        y_pred['y_pred'],
                        "medsynth",
                        "xgboost",
                        figsize=(10, 10),
                        class_names=class_names)

data_schema = xailens.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="y_test",
    prediction_column="y_pred")

model_data = xailens.ModelData(
    data_schema=data_schema,
    raw_instances=raw,
    x_test=X_test,
    y_test=y_test,
    y_pred=y_pred)

ctx = xailens.ModelContext(
    "xgboost",
    ["tree_shap"],
    "medsynth",
    model_data,
    model=model,
    model_dir_name="medsynth_xgboost",
    model_display_name="XGBoost")

xailens.run(ctx)