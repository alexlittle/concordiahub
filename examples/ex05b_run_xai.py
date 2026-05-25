import pandas as pd
import xai_compare as xai

from examples import config


X = pd.read_parquet('./cache/medsynth/llm/X.parquet')
y = pd.read_parquet('./cache/medsynth/llm/y.parquet')
y_preds = pd.read_parquet('./cache/medsynth/llm/y_preds.parquet')
y_exp = pd.read_parquet('./cache/medsynth/llm/y_exp.parquet')

ctx = xai.RunContext("llm",
               ["cor"],
               "medsynth",
               xai.RunData(x=X,
                   y=y,
                   y_pred=y_preds,
                   y_exp=y_exp,),
                model_subtype=config.MEDSYNTH_OLLAMA_MODEL)
xai.run(ctx)