"""
ArtifactStore class - for handling data storage from the ModelContext
"""


import os
import pandas as pd
import numpy as np
import uuid

from typing import Optional, Any
from datetime import datetime
from dataclasses import asdict, fields, is_dataclass

from xailens.entities.context import ModelContext
from xailens import config
from xailens.storage.store_base import DataStore
from xailens.storage.filesystem_store import FilesystemStore


class ArtifactStore:
    """
    ArtifactStore - handles the data storage

     Args:
        ctx (ModelContext): The current model configuration.
        store (Optional[DataStore]): Storage backend. Defaults to FilesystemStore.
    """

    def __init__(self, ctx: ModelContext, store: Optional[DataStore] = None) -> None:
        self.ctx = ctx
        self.datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirname = self._resolve_dirname(ctx)
        self.store = store or FilesystemStore(dirname)


    def _resolve_dirname(self, ctx: ModelContext)-> str:
        """
        Private method to resolve the directory name

        Args:
            ctx (ModelContext): ModelContext object:

        Returns:
            str: directory name
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if ctx.model_dir_name:
            return os.path.join(base_dir, config.MODEL_DATA_DIR_NAME, ctx.model_dir_name)
        return os.path.join(base_dir, config.MODEL_DATA_DIR_NAME, self.datetime)


    def write_metadata(self) -> None:
        """
        Writes the artifact metadata directly from the context dataclass
        """
        excluded = {'data', 'model'}
        obj = {}
        for f in fields(self.ctx):
            if f.name in excluded:
                continue
            value = getattr(self.ctx, f.name)
            obj[f.name] = asdict(value) if is_dataclass(value) else value

        obj['datetime'] = self.datetime
        obj['model_uuid'] = str(uuid.uuid4())
        obj['data_schema'] = asdict(self.ctx.data.data_schema)
        self.store.save_json(config.META_INFO_FILENAME, obj)

    def write_model(self) -> None:
        """
        Writes the model to the data store directly from the context dataclass
        """
        if self.ctx.model:
            self.store.save_model(config.MODEL_FILENAME ,self.ctx.model)

    def write_metrics(self, metrics_series: pd.Series) -> None:
        """
        Writes the metrics to the data store directly from the context dataclass

        Args:
            metrics_series (pd.Series): pandas series containing metrics:
        """
        self.store.save_series("metrics", metrics_series)

    def write_data(self, name: str,
                   data: Optional[pd.Series | np.ndarray | list | pd.DataFrame] = None) -> bool:
        """
        Saves the given data to the data store

        Args:
            name (string): name of the data to save
            data (Optional[pd.Series | np.ndarray | list | pd.DataFrame]): data
                to save. Series, ndarray, and list inputs are converted to a
                DataFrame before saving.

        Returns:
            bool: True if the data was saved, False otherwise
        """

        if data is None:
            return False
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        self.store.save_dataframe(name, data)
        return True

    def write_explanation(self, name: str, data: pd.DataFrame | list[dict[str, Any]], data_format: str ="df") -> None:
        """
        Writes the given explanation to the data store

        Args:
            name (str): name of explanation file to write
            data (pd.DataFrame | list[dict[str, Any]]): the data to write, may be dataframe or list (processed JSON)
            data_format (str): either "df" or "json", must match what is sent in data
        """
        self.store.save_explanation(name, data, data_format)

    def load_dataframe(self, name:str) -> pd.DataFrame:
        """
        Loads a dataframe from the data store

        Args:
            name (str): Name of the dataframe to load

        Returns:
            pd.DataFrame: dataframe
        """
        return self.store.load_dataframe(name)

