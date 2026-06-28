import pandas as pd
import dataclasses

from sklearn.metrics import confusion_matrix, classification_report
from xailens.storage.filesystem_store import FilesystemStore
import xailens.config as config
from xailens.schema import DataSchema

class ModelEval:

    def __init__(self, path: str = None, store=None, metadata: dict = None):
        self.path = path
        if store:
            self.store = store
        elif path:
            self.store = FilesystemStore(path)
        else:
            self.store = None
        self.metadata = metadata or {}

    @property
    def datetime(self):
        return self.metadata.get("datetime")

    def get(self, key, default=None):
        parts = key.split(".")
        value = self.metadata

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_data_schema(self) -> DataSchema:
        kwargs = {}
        for field in dataclasses.fields(DataSchema):
            value = self.get(f"data_schema.{field.name}", field.default)
            kwargs[field.name] = value
        return DataSchema(**kwargs)

    def _get_series(self):
        y_true, y_pred = self.load_targets()
        return (y_true[self.get(config.DATA_SCHEMA_TARGET_COLUMN)],
                y_pred[self.get(config.DATA_SCHEMA_PREDICTION_COLUMN)])

    def load_targets(self):
        target_cols = [self.get(config.DATA_SCHEMA_ID_COLUMN), self.get(config.DATA_SCHEMA_TARGET_COLUMN)]
        pred_cols = [self.get(config.DATA_SCHEMA_ID_COLUMN), self.get(config.DATA_SCHEMA_PREDICTION_COLUMN)]
        y_true = self.store.load_dataframe("y_test")[target_cols]
        y_pred = self.store.load_dataframe("y_pred")[pred_cols]
        merged = y_true.merge(y_pred, on=self.get(config.DATA_SCHEMA_ID_COLUMN), how='inner')
        return merged[target_cols], merged[pred_cols]

    def get_metrics(self):
        return self.store.load_series("metrics")

    def get_instance_data(self, instance_uuid) -> tuple[pd.Series, str]:
        uuid_col = self.get(config.DATA_SCHEMA_ID_COLUMN)
        target_col = self.get(config.DATA_SCHEMA_TARGET_COLUMN)

        raw = self.store.load_dataframe("raw_instances")
        row = raw[raw[uuid_col] == instance_uuid].squeeze()

        y_test = self.store.load_dataframe("y_test")
        label = y_test[y_test[uuid_col] == instance_uuid][target_col].squeeze()

        return row.drop(labels=[uuid_col]), label

    def confusion_matrix(self):
        y_true, y_pred = self._get_series()
        return confusion_matrix(y_true, y_pred)

    def classification_report(self):
        y_true, y_pred = self._get_series()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return pd.DataFrame(report).transpose()

