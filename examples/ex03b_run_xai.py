
import pandas as pd
import xai_compare as xai

from examples import config


X = pd.read_parquet('./cache/heart/llm/X.parquet')
y = pd.read_parquet('./cache/heart/llm/y.parquet')
y_preds = pd.read_parquet('./cache/heart/llm/y_preds.parquet')
y_exp = pd.read_parquet('./cache/heart/llm/y_exp.parquet')

ctx = xai.RunContext("llm",
               ["cor"],
               "cleveland-heart-data",
               xai.RunData(x=X,
                   y=y,
                   y_pred=y_preds,
                   y_exp=y_exp,),
                model_subtype=config.HEART_OLLAMA_MODEL,
                run_name="llm_heart")
xai.run(ctx)