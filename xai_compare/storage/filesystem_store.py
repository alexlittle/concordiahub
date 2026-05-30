'''
Implementation of filesystem storage
'''
import os

import joblib
import json
import pandas as pd

from xai_compare.storage.store_base import DataStore

class FilesystemStore(DataStore):

    def __init__(self, dirname):
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    def save_json(self, name, data):
        with open(os.path.join(self.dirname, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=4)

    def save_dataframe(self, name, df):
        path = os.path.join(self.dirname, f"{name}.parquet")
        df.to_parquet(path)

    def save_explanation(self, name, df):
        path = os.path.join(self.dirname, f"{name}.parquet")
        df.columns = df.columns.astype(str)
        df.reset_index().to_parquet(path)

    def save_model(self, name, model):
        joblib.dump(model, os.path.join(self.dirname, f"{name}.joblib"))

    def load_dataframe(self, name:str) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.dirname, f"{name}.parquet"))