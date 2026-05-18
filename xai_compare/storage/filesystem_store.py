'''
Implementation of filesystem storage
'''
import os

import joblib
import json
import numpy as np

from xai_compare.storage.store_base import DataStore

class FilesystemStore(DataStore):

    def __init__(self, dirname):
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    def save_json(self, filename, data):
        with open(os.path.join(self.dirname, filename), "w") as f:
            json.dump(data, f, indent=4)

    def save_dataframe(self, filename, data):
        data.to_parquet(os.path.join(self.dirname, filename))

    def save_model(self, filename, model):
        joblib.dump(model, os.path.join(self.dirname, filename))