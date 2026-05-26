import pandas as pd
import xai_compare as xai

from examples import config


X = pd.read_parquet('./cache/medsynth/llm/X.parquet')
y = pd.read_parquet('./cache/medsynth/llm/y.parquet')
y_preds = pd.read_parquet('./cache/medsynth/llm/y_preds.parquet')
y_exp = pd.read_parquet('./cache/medsynth/llm/y_exp.parquet')

data_schema = xai.DataSchema(
    sample_id_column=config.SAMPLE_UUID_COL,
    target_column="target")

run_data = xai.RunData(
    data_schema=data_schema,
    x=X,
    y=y,
    y_pred=y_preds,
    y_exp=y_exp)

ctx = xai.RunContext(
    "llm",
    ["cor"],
    "medsynth",
    run_data,
    model_subtype=config.MEDSYNTH_OLLAMA_MODEL,
    run_name="llm_medsynth",
    model_id="LLM v1")

xai.run(ctx)