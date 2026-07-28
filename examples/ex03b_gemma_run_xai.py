"""
File to register the LLM Gemma model with the XAI Lens Framework, from
pre-generated explanations
"""

import pandas as pd
import xailens

from examples import config
from examples.helpers import safe_dirname, create_confusion_matrix, save_class_report
from sklearn.metrics import classification_report

current_model = "gemma3:27b-it-qat"
dir_name = f"llm_{safe_dirname(current_model)}"

# load the cached files previously saved from running ex03_llm_heart.py
raw = pd.read_parquet(f"./cache/heart/{dir_name}/raw_instances_filtered.parquet")
X = pd.read_parquet(f"./cache/heart/{dir_name}/X_filtered.parquet")
y = pd.read_parquet(f"./cache/heart/{dir_name}/y_filtered.parquet")
y_preds = pd.read_parquet(f"./cache/heart/{dir_name}/y_preds_filtered.parquet")
y_exp = pd.read_parquet(f"./cache/heart/{dir_name}/y_exp_filtered.parquet")

# print/save the classification & confusion matrix
report = classification_report(y["target"], y_preds["prediction"])
print(report)
save_class_report(report, "heart", "llm_gemma")
create_confusion_matrix(y["target"],
                        y_preds["prediction"],
                        "heart",
                        "llm_gemma")



schema = xailens.ResponseSchema(
    contributions_field="feature_contributions",
    contribution_key=None,
)

data_schema = xailens.DataSchema(
    instance_id_column=config.INSTANCE_UUID_COL,
    target_column="target",
    prediction_column="prediction")

model_data = xailens.ModelData(
    data_schema=data_schema,
    raw_instances=raw,
    x_test=X,
    y_test=y,
    y_pred=y_preds,
    y_exp=y_exp)

ctx = xailens.ModelContext(
    "llm",
    ["llm_reasoning"],
    "cleveland-heart-data",
    model_data,
    model_subtype=current_model,
    model_dir_name=f"heart_llm_{safe_dirname(current_model)}",
    model_display_name="Gemma 3",
    response_schema=schema)

xailens.run(ctx)