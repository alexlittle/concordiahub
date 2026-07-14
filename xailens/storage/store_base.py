"""
Abstract class for storing data and model
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Any

from xailens.config import TreeModel


class DataStore(ABC):
    """
    Abstract DataStore class for storing data and model
    """

    @abstractmethod
    def save_json(self, filename: str , data:str) -> None:
        """
        Saves data to datastore

        Args:
            filename (str): filename to save to
            data (str): the JSON data as string

        """
        pass

    @abstractmethod
    def save_dataframe(self, filename:str, data: pd.DataFrame) -> None:
        """
        Saves dataframe to datastore

        Args:
           filename (str): name of file to save
           data (pd.DataFrame): the dataframe to save

        """
        pass

    @abstractmethod
    def save_series(self, filename:str , data: pd.Series) -> None:
        """
        Saves series to datastore

        Args:
           filename (str): name of file to save
           data (pd.Series): the series to save

        """
        pass

    @abstractmethod
    def save_model(self, filename:str, model: TreeModel) -> None:
        """
        Saves model to joblib

        Args:
            filename (str): name of the file to save
            model (TreeModel): model object to save

        """
        pass

    @abstractmethod
    def save_explanation(self, filename:str, data: pd.DataFrame | str, data_format: Optional[str] = "df")-> None:
        """
        Saves csv of JSON explanation to datastore

        Args:
            filename (str): name of file to save
            data (pd.DataFrame | str): either a dataframe or JSON in string format
            data_format (Optional[str]): format to save, either "df" or "json", defaults to "df"

        """
        pass

    @abstractmethod
    def load_dataframe(self, filename:str) -> pd.DataFrame:
        """
        Loads dataframe from datastore

        Args:
            filename (str): name of file to load

        Returns:
            pd.DataFrame: dataframe to return
        """
        pass

    @abstractmethod
    def load_json(self, filename: str) -> list[dict[str, Any]]:
        """
        Loads JSON from datastore
        Args:
            filename (str): name of file to load

        Returns:
            list[dict[str, Any]]: JSON to return
        """
        pass

    @abstractmethod
    def load_series(self, filename: str) -> pd.Series:
        """
        Loads a series from datastore

        Args:
            filename (str): name of file to load

        Returns:
            pd.Series: the series to return
        """
        pass