import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xai_compare.storage.filesystem_store import FilesystemStore
from xai_compare import constants

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


    def load_targets(self):
        try:
            y_true = self.store.load_dataframe("y_test")
        except FileNotFoundError:
            y_true = self.store.load_dataframe("y")

        y_pred = self.store.load_dataframe("y_pred")
        return y_true, y_pred

    def accuracy(self):
        y_true, y_pred = self.load_targets()
        return accuracy_score(y_true, y_pred)

    def confusion_matrix(self):
        y_true, y_pred = self.load_targets()
        return confusion_matrix(y_true, y_pred)

    def classification_report(self):
        y_true, y_pred = self.load_targets()
        report = classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()

    def load_local_explanation(self, xai_method: str, sample_id: str) -> pd.Series:
        """Retrieves the local explanation for a given sample_uuid
        Args:
            xai_method: xai method (str)
            sample_id: sample id (str - UUID)
        returns:
            explanation: pd.Series
        """
        df = self.store.load_dataframe(f"explanations_{xai_method}")
        row = df[df[self.get(constants.DATA_SCHEMA_ID_COLUMN)] == sample_id]
        if row.empty:
            raise ValueError(f"Sample '{sample_id}' not found in explanations for {xai_method}")
        return row.iloc[0]

