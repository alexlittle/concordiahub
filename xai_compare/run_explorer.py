# ./run_explorer.py
import os
import json
from pathlib import Path

from xai_compare import config
from .storage.filesystem_store import FilesystemStore

from .storage.store_base import DataStore

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