import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xailens.storage.filesystem_store import FilesystemStore
from xailens import constants

class Run():

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

    def _get_series(self):
        y_true, y_pred = self.load_targets()
        return (y_true[self.get(constants.DATA_SCHEMA_TARGET_COLUMN)],
                y_pred[self.get(constants.DATA_SCHEMA_PREDICTION_COLUMN)])

    def load_targets(self):
        target_cols = [self.get(constants.DATA_SCHEMA_ID_COLUMN), self.get(constants.DATA_SCHEMA_TARGET_COLUMN)]
        pred_cols = [self.get(constants.DATA_SCHEMA_ID_COLUMN), self.get(constants.DATA_SCHEMA_PREDICTION_COLUMN)]
        try:
            y_true = self.store.load_dataframe("y_test")[target_cols]
        except FileNotFoundError:
            y_true = self.store.load_dataframe("y")[target_cols]
        y_pred = self.store.load_dataframe("y_pred")[pred_cols]
        merged = y_true.merge(y_pred, on=self.get(constants.DATA_SCHEMA_ID_COLUMN), how='inner')
        return (merged[target_cols], merged[pred_cols])

    def accuracy(self):
        y_true, y_pred = self._get_series()
        return accuracy_score(y_true, y_pred)

    def confusion_matrix(self):
        y_true, y_pred = self._get_series()
        return confusion_matrix(y_true, y_pred)

    def classification_report(self):
        y_true, y_pred = self._get_series()
        report = classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()

