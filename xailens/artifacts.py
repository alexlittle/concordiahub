import os
import pandas as pd
import numpy as np
import uuid

from datetime import datetime
from dataclasses import asdict, fields, is_dataclass

from xailens.entities.context import ModelContext
from xailens import config
from xailens.storage.store_base import DataStore
from xailens.storage.filesystem_store import FilesystemStore


class ArtifactStore:
    """ArtifactStore - handles the data storage

     Args:
        ctx (ModelContext): The current run configuration.
        store (DataStore): Storage backend. Defaults to FilesystemStore.
    """

    def __init__(self, ctx: ModelContext, store: DataStore = None):
        self.ctx = ctx
        self.datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirname = self._resolve_dirname(ctx)
        self.store = store or FilesystemStore(dirname)


    def _resolve_dirname(self, ctx):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if ctx.model_name:
            return os.path.join(BASE_DIR, config.MODEL_DATA_DIR_NAME, ctx.model_name)
        return os.path.join(BASE_DIR, config.MODEL_DATA_DIR_NAME, self.datetime)


    def write_metadata(self):
        """Writes the artifact metadata directly from the context dataclass
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

    def write_model(self):
        if self.ctx.model:
            self.store.save_model(config.MODEL_FILENAME ,self.ctx.model)

    def write_data(self, name, data):
        if data is None:
            return
        if isinstance(data, pd.Series):
            data = data.to_frame()
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        self.store.save_dataframe(name, data)

    def write_explanation(self, name, data, type="df"):
        self.store.save_explanation(name, data, type)

    def load_dataframe(self, name:str) -> pd.DataFrame:
        return self.store.load_dataframe(name)

