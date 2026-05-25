from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xai_compare.storage.filesystem_store import FilesystemStore

class Run:

    def __init__(self, path: str = None, store=None):
        self.path = path
        self.store = store or FilesystemStore(path)

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
        return classification_report(y_true, y_pred)
