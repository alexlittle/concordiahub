
import pandas as pd
import xai_compare as xai

from examples import config

df = pd.read_csv(config.HEART_INPUT_FILE)

X = df.drop(columns=[config.SAMPLE_UUID_COL, "target"])
X.index = df[config.SAMPLE_UUID_COL]

y = df["target"]
y.index = df[config.SAMPLE_UUID_COL]

df_preds = pd.read_csv("results/heart/llm_heart_preds.csv")

y_preds = df_preds["prediction"]
y_preds.index = df_preds[config.SAMPLE_UUID_COL]

y_exp = df_preds["cor_explanation"]
y_exp.index = df_preds[config.SAMPLE_UUID_COL]

ctx = xai.RunContext("llm",
               ["cor"],
               "cleveland-heart-data",
               xai.RunData(x=X,
                   y=y,
                   y_pred=y_preds,
                   y_exp=y_exp,),
                model_subtype=config.HEART_OLLAMA_MODEL)
xai.run(ctx)