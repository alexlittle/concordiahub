"""
Script to register the LLM Qwen medsynth model with the XAI Lens Framework, from
pre-generated explanations
"""


import pandas as pd
import xailens

from examples import config
from examples.helpers import safe_dirname
from helpers import ICD10_CHAPTERS, create_confusion_matrix, save_class_report
from sklearn.metrics import classification_report

current_model = "qwen2.5:14b-instruct-q8_0"
dir_name = f"llm_{safe_dirname(current_model)}"
raw = pd.read_parquet(f'./cache/medsynth/{dir_name}/raw_instances.parquet')
X = pd.read_parquet(f"./cache/medsynth/{dir_name}/X.parquet")
y = pd.read_parquet(f"./cache/medsynth/{dir_name}/y.parquet")
y_pred = pd.read_parquet(f"./cache/medsynth/{dir_name}/y_preds.parquet")
y_exp = pd.read_parquet(f"./cache/medsynth/{dir_name}/y_exp.parquet")



y["target"] = y["target"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)
y_pred["prediction"] = y_pred["prediction"].map(
    lambda code: f"{code}: {ICD10_CHAPTERS[code]}" if code in ICD10_CHAPTERS else code
)


report = classification_report(y["target"],
                               y_pred['prediction'],
                               zero_division=True)
print(report)

save_class_report(report, "medsynth", "llm_qwen")
class_names = sorted(set(y["target"]) | set(y_pred["prediction"]))
create_confusion_matrix(y["target"],
                        y_pred['prediction'],
                        "medsynth",
                        "llm_qwen",
                        figsize=(10, 10),
                        class_names=class_names)

schema = xailens.ResponseSchema(
    contributions_field="key_factors",
    contribution_key="phrase",
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
    y_pred=y_pred,
    y_exp=y_exp)

ctx = xailens.ModelContext(
    "llm",
    ["llm_reasoning"],
    "medsynth",
    model_data,
    model_subtype=current_model,
    model_dir_name=f"medsynth_llm_{safe_dirname(current_model)}",
    model_display_name="Qwen 2.5",
    response_schema=schema)

xailens.run(ctx)