import os
import pandas as pd
import numpy as np
import uuid

from datetime import datetime
from dataclasses import asdict

from xai_compare import RunContext
from xai_compare import config
from xai_compare.storage.store_base import DataStore
from xai_compare.storage.filesystem_store import FilesystemStore


class ArtifactStore:
    """ArtifactStore - handles the data storage

     Args:
        ctx (RunContext): The current run configuration.
        store (DataStore): Storage backend. Defaults to FilesystemStore.
    """

    def __init__(self, ctx: RunContext, store: DataStore = None):
        self.ctx = ctx
        self.run_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirname = self._resolve_dirname(ctx)
        self.store = store or FilesystemStore(dirname)


    def _resolve_dirname(self, ctx):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if ctx.run_name:
            return os.path.join(BASE_DIR, config.RUNS_DIR_NAME, ctx.run_name)
        return os.path.join(BASE_DIR, config.RUNS_DIR_NAME, self.run_datetime)


    def write_metadata(self):
        """Writes the artifact metadata directly from the context dataclass
        """
        excluded = {'data', 'model'}
        obj = {k: v for k, v in asdict(self.ctx).items() if k not in excluded}
        obj['datetime'] = self.run_datetime
        obj['run_uuid'] = str(uuid.uuid4())
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
        self.store.save_dataframe(f"{name}.parquet", data)

    def load_dataframe(self, name:str) -> pd.DataFrame:
        return self.store.load_dataframe(f"{name}.parquet")

