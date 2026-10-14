"""
Implementation of filesystem storage
"""

import os

import joblib
import json
import pandas as pd
import streamlit as st

from typing import Any, Optional

from concordiahub.config import ScikitLearnModel
from concordiahub.storage.store_base import DataStore


@st.cache_resource
def _cached_load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV file into dataframe. Caching in memory for faster response
    on next function call.

    Args:
        path (str): path to the CSV file

    Returns:
        pd.DataFrame: dataframe
    """
    return pd.read_csv(path)


@st.cache_resource
def _cached_load_json(path: str) -> list[dict[str, Any]]:
    """
    Loads a JSON file. Caching in memory for faster response on next function
    call.

    Args:
        path (str): path to the JSON file

    Returns:
        list[dict[str, Any]]: parsed JSON content
    """
    with open(path, 'r') as f:
        return json.load(f)

class FilesystemStore(DataStore):
    """
    Implementation of filesystem storage
    """

    def __init__(self, dirname):
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    def save_json(self, filename: str, data: str):
        """
        Saves data to JSON file

        Args:
            filename (str): name of file to save
            data (str): the JSON data as string

        """
        with open(os.path.join(self.dirname, f"{filename}.json"), "w") as f:
            json.dump(data, f, indent=4)

    def save_dataframe(self, filename: str, df: pd.DataFrame):
        """
        Saves dataframe to csv file

        Args:
           filename (str): name of file to save
           df (pd.DataFrame): the dataframe to save

        """
        path = os.path.join(self.dirname, f"{filename}.csv")
        df.to_csv(path, index=False)

    def save_series(self, filename:str, series: pd.Series):
        """
        Saves series to csv file

        Args:
           filename (str): name of file to save
           series (pd.Series): the series to save

        """
        path = os.path.join(self.dirname, f"{filename}.csv")
        series.to_csv(path)

    def save_explanation(
            self,
            filename:str,
            data: pd.DataFrame | str, data_format: Optional[str]="df"):
        """
        Saves explanation to csv or JSON file

        Args:
            filename (str): name of file to save
            data (pd.DataFrame | str): either a dataframe or JSON in string
                format
            data_format (Optional[str]): format to save, either "df" or "json",
                defaults to "df"

        """
        if data_format == "df":
            path = os.path.join(self.dirname, f"{filename}.csv")
            data.columns = data.columns.astype(str)
            data.to_csv(path, index=False)
        else:
            with open(os.path.join(self.dirname, f"{filename}.json"), "w") as f:
                json.dump(data, f, indent=4)

    def save_model(self, filename:str, model:ScikitLearnModel):
        """
        Saves model to joblib file

        Args:
            filename (str): name of the file to save
            model (ScikitLearnModel): model object to save

        """
        joblib.dump(model, os.path.join(self.dirname, f"{filename}.joblib"))

    def load_dataframe(self, filename:str) -> pd.DataFrame:
        """
        Loads dataframe from csv file. Will load from cache if already in memory

        Args:
            filename (str): name of file to load

        Returns:
            pd.DataFrame: dataframe to return
        """
        return _cached_load_csv(os.path.join(self.dirname, f"{filename}.csv"))

    def load_json(self, filename:str) -> list[dict[str, Any]]:
        """
        Loads JSON from file. Will load from cache if already in memory

        Args:
            filename (str): name of file to load

        Returns:
            list[dict[str, Any]]: JSON to return
        """
        return _cached_load_json(os.path.join(self.dirname, f"{filename}.json"))

    def load_series(self, filename:str) -> pd.Series:
        """
        Loads a series from csv file.

        Args:
            filename (str): name of file to load

        Returns:
            pd.Series: the series to return
        """
        return pd.read_csv(
            os.path.join(
                self.dirname, f"{filename}.csv"), index_col=0).squeeze()