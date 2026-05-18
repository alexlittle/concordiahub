import os
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from xai_compare import config
from xai_compare.storage.filesystem_store import FilesystemStore

from xai_compare.storage.store_base import DataStore

#-----------------------
# Helper functions
#------------------------
def extract_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("dataset_id")


class RunExplorer():

    def __init__(self, store: DataStore = None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.runs_dir = os.path.join(BASE_DIR, config.RUNS_DIR_NAME)
        self.store = store or FilesystemStore(self.runs_dir)

    def get_datasets(self):
        """Gets names of all datasets stored in the runs directory

        Returns:
            set: set of distinct dataset names
        """
        datasets = set()
        root = Path(self.runs_dir)
        for meta_file in root.glob(f"*/{config.META_INFO_FILENAME}"):
            datasets.add(extract_dataset(meta_file))
        return sorted(datasets)

    def get_accuracy_score(self) -> float:
        try:
            y_test = self.store.load_dataframe("y_test")
        except FileNotFoundError:
            y_test = self.store.load_dataframe("y")
        y_pred = self.store.load_dataframe("y_pred")
        return accuracy_score(y_test, y_pred)

    def get_confusion_matrix(self) -> confusion_matrix:
        try:
            y_test = self.store.load_dataframe("y_test")
        except FileNotFoundError:
            y_test = self.store.load_dataframe("y")
        y_pred = self.store.load_dataframe("y_pred")
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(self) -> classification_report:
        try:
            y_test = self.store.load_dataframe("y_test")
        except FileNotFoundError:
            y_test = self.store.load_dataframe("y")
        y_pred = self.store.load_dataframe("y_pred")
        return classification_report(y_test, y_pred)