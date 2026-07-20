"""
File to register the LLM Qwen model with the XAI Lens Framework, from
pre-generated explanations
"""
import pandas as pd
import xailens

from examples import config
from examples.helpers import safe_dirname, create_confusion_matrix, save_class_report
from sklearn.metrics import classification_report

current_model = "qwen2.5:14b-instruct-q8_0"
dir_name = f"llm_{safe_dirname(current_model)}"

raw = pd.read_parquet(f"./cache/heart/{dir_name}/raw_instances.parquet")
X = pd.read_parquet(f"./cache/heart/{dir_name}/X.parquet")
y = pd.read_parquet(f"./cache/heart/{dir_name}/y.parquet")
y_preds = pd.read_parquet(f"./cache/heart/{dir_name}/y_preds.parquet")
y_exp = pd.read_parquet(f"./cache/heart/{dir_name}/y_exp.parquet")

report = classification_report(y["target"], y_preds["prediction"])
print(report)
save_class_report(report, "heart", "llm_qwen")
create_confusion_matrix(y["target"],
                        y_preds["prediction"],
                        "heart",
                        "llm_qwen")


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
    model_display_name="Qwen 2.5",
    response_schema=schema)

xailens.run(ctx)