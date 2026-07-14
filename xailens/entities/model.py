"""
Model class
"""

import pandas as pd
import dataclasses

from typing import Any,Optional

from xailens import config
from xailens.storage.filesystem_store import FilesystemStore
from xailens.schema import DataSchema

class Model:
    """
    Model class
    """

    def __init__(self, path: str, metadata: dict = None):
        self.path = path
        self.store = FilesystemStore(path)
        self.metadata = metadata or {}

    def get(self, key: str, default: Optional[str] = None) -> str :
        """
        Gets a value by its key, will also look in component dicts,
        (eg) dataschema.target_column

        Args:
            key (str): the key to look for
            default (Optional[str]): a default value to return if key is not found

        Returns:
            str:

        """
        parts = key.split(".")
        value = self.metadata

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_data_schema(self) -> DataSchema:
        """
        Returns the DataSchema dataclass for the model

        Returns:
            DataSchema:  dataclass instance
        """
        kwargs = {}
        for field in dataclasses.fields(DataSchema):
            value = self.get(f"data_schema.{field.name}", field.default)
            kwargs[field.name] = value
        return DataSchema(**kwargs)

    def _get_series(self) -> tuple[pd.Series, pd.Series]:
        """
        Private helper function to get the tuple from self.load_targets()

        Returns:
            tuple[pd.Series, pd.Series]: true target and predictions
        """
        y_true, y_pred = self.load_targets()
        return (y_true[self.get(config.DATA_SCHEMA_TARGET_COLUMN)],
                y_pred[self.get(config.DATA_SCHEMA_PREDICTION_COLUMN)])

    def load_targets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the target columns from the data

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: target and prediction columns
        """

        target_cols = [self.get(config.DATA_SCHEMA_ID_COLUMN), self.get(config.DATA_SCHEMA_TARGET_COLUMN)]
        pred_cols = [self.get(config.DATA_SCHEMA_ID_COLUMN), self.get(config.DATA_SCHEMA_PREDICTION_COLUMN)]
        y_true = self.store.load_dataframe("y_test")[target_cols]
        y_pred = self.store.load_dataframe("y_pred")[pred_cols]
        merged = y_true.merge(y_pred, on=self.get(config.DATA_SCHEMA_ID_COLUMN), how='inner')
        return merged[target_cols], merged[pred_cols]

    def get_metrics(self) -> pd.Series:
        """
        Get the metrics for the model

        Returns:
            pd.Series:
        """
        return self.store.load_series("metrics")

    def get_instance_data(self, instance_uuid: str) -> pd.Series:
        """
        Get the raw instance data

        Args:
            instance_uuid (str):

        Returns:
            pd.Series:
        """
        uuid_col = self.get(config.DATA_SCHEMA_ID_COLUMN)
        raw = self.store.load_dataframe("raw_instances")
        row = raw[raw[uuid_col] == instance_uuid].squeeze()

        return row.drop(labels=[uuid_col])

    def get_instance_prediction(self, instance_uuid: str) -> str:
        """
        Get the models prediction for the given instance

        Args:
            instance_uuid (str):

        Returns:
            str:
        """
        uuid_col = self.get(config.DATA_SCHEMA_ID_COLUMN)
        pred_col = self.get(config.DATA_SCHEMA_PREDICTION_COLUMN)
        y_pred = self.store.load_dataframe("y_pred")
        label = y_pred[y_pred[uuid_col] == instance_uuid][pred_col].squeeze()
        return label


