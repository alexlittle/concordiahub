"""
Implementation of filesystem storage
"""

import os

import joblib
import json
import pandas as pd
import streamlit as st

from typing import Any
from xailens.storage.store_base import DataStore


@st.cache_resource
def _cached_load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV file into dataframe. Caching in memory for faster response on next function call.

    Args:
        path (str): path to the CSV file

    Returns:
        pd.DataFrame: dataframe
    """
    return pd.read_csv(path)


@st.cache_resource
def _cached_load_json(path: str) -> list[dict[str, Any]]:
    """
    Loads a JSON file. Caching in memory for faster response on next function call.

    Args:
        path (str): path to the JSON file

    Returns:
        list[dict[str, Any]]: parsed JSON content
    """
    with open(path, 'r') as f:
        return json.load(f)

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
        return _cached_load_csv(os.path.join(self.dirname, f"{name}.csv"))

    def load_json(self, name:str) -> list[dict[str, Any]]:
        return _cached_load_json(os.path.join(self.dirname, f"{name}.json"))

    def load_series(self, name:str) -> pd.Series:
        return pd.read_csv(os.path.join(self.dirname, f"{name}.csv"), index_col=0).squeeze()