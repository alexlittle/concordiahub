"""
Implementation of filesystem storage
"""
import os

import joblib
import json
import pandas as pd

from xailens.storage.store_base import DataStore

class FilesystemStore(DataStore):

    def __init__(self, dirname):
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    def save_json(self, name, data):
        with open(os.path.join(self.dirname, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=4)

    def save_dataframe(self, name, df:pd.DataFrame):
        path = os.path.join(self.dirname, f"{name}.csv")
        df.to_csv(path, index=False)

    def save_series(self, name:str, series: pd.Series):
        path = os.path.join(self.dirname, f"{name}.csv")
        series.to_csv(path)

    def save_explanation(self, name, data, data_format="df"):
        if data_format == "df":
            path = os.path.join(self.dirname, f"{name}.csv")
            data.columns = data.columns.astype(str)
            data.to_csv(path, index=False)
        else:
            with open(os.path.join(self.dirname, f"{name}.json"), "w") as f:
                json.dump(data, f, indent=4)

    def save_model(self, name, model):
        joblib.dump(model, os.path.join(self.dirname, f"{name}.joblib"))

    def load_dataframe(self, name:str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dirname, f"{name}.csv"))

    def load_json(self, name:str) -> list:
        with open(os.path.join(self.dirname, f"{name}.json"), 'r') as f:
            return json.load(f)

    def load_series(self, name:str) -> pd.Series:
        return pd.read_csv(os.path.join(self.dirname, f"{name}.csv"), index_col=0).squeeze()