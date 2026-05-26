import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xai_compare.storage.filesystem_store import FilesystemStore


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
        return self.metadata.get(key, default)


    def _load_targets(self):
        try:
            y_true = self.store.load_dataframe("y_test")
        except FileNotFoundError:
            y_true = self.store.load_dataframe("y")

        y_pred = self.store.load_dataframe("y_pred")
        return y_true, y_pred

    def accuracy(self):
        y_true, y_pred = self._load_targets()
        return accuracy_score(y_true, y_pred)

    def confusion_matrix(self):
        y_true, y_pred = self._load_targets()
        return confusion_matrix(y_true, y_pred)

    def classification_report(self):
        y_true, y_pred = self._load_targets()
        report = classification_report(y_true, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()

