import os
import json

from collections import defaultdict
from pathlib import Path
from itertools import combinations

from xailens import config, strings
from xailens.entities.modeleval import ModelEval


class ModelExplorer:

    def __init__(self):
        """Initializes ModelExplorer"""
        base_dir = Path(__file__).resolve().parent.parent
        self.modelevals_dir = os.path.join(base_dir, config.MODEL_DATA_DIR_NAME)
        self._metadata_cache = self._load_all_metadata()
        self._uuid_index = {
            model.get("model_uuid"): model for model in self._metadata_cache
        }

    def get_datasets(self):
        """Gets names of all datasets stored in the models directory

        Returns:
            set: set of distinct dataset names
        """
        datasets = set()
        for modeleval in self._metadata_cache:
            datasets.add(modeleval["dataset_id"])
        return sorted(datasets)

    def get_models_for_dataset(self, dataset: str) -> dict:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            dict of model names and model_data"""
    
        models = {}
        for modeleval in self._metadata_cache:
            if modeleval["dataset_id"] == dataset:
                modeleval_obj = self.get_modeleval_by_uuid(modeleval["model_uuid"])
                if modeleval_obj:
                    models[modeleval_obj.get("model_display_name")] = modeleval["model_uuid"]
        return dict(sorted(models.items()))

    def get_selected_modelevals_for_dataset(self, dataset: str, model_names: list[str]) -> [ModelEval]:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            list of ModelEval (sorted by model_display_name)"""
        modelevals = []
        for modeleval in self._metadata_cache:
            if modeleval["dataset_id"] == dataset and modeleval["model_display_name"] in model_names:
                modelevals.append(self.get_modeleval_by_uuid(modeleval["model_uuid"]))
        return sorted(modelevals, key=lambda m: m.get("model_display_name"))

    def get_modelevals_for_dataset(self, dataset: str) -> [ModelEval]:
        """ Gets all the models for the given dataset name
        Args:
            dataset str: dataset name
        Returns:
            list of ModelEval"""
        modelevals = []
        for modeleval in self._metadata_cache:
            if modeleval["dataset_id"] == dataset:
                modelevals.append(self.get_modeleval_by_uuid(modeleval["model_uuid"]))
        return modelevals

    def is_valid_dataset(self, dataset) -> bool:
        """Checks if dataset is valid
        Args:
            dataset str: dataset name
        Returns:
        """
        for modeleval in self._metadata_cache:
            if modeleval["dataset_id"] == dataset:
                return True
        return False


    def get_modeleval_by_uuid(self, uuid: str) -> ModelEval | None:
        """Gets a Run by its UUID
        Args:
            uuid: Run's UUID

        Returns:
            Run or None"""
        meta = self._uuid_index.get(uuid)
        return ModelEval(path=meta["_path"], metadata=meta) if meta else None

    def get_modelevals_by_uuids(self, uuids: list[str]) -> list[ModelEval]:
        """Gets a list of Runs by their UUIDs
        Args:
            List[uuid]: Run UUIDs

        Returns:
            List[Run] or None"""

        modelevals = []
        for u in uuids:
            meta = self._uuid_index.get(u)
            if not meta:
                continue
            modelevals.append(ModelEval(path=meta["_path"], metadata=meta))
        return modelevals

    #---------------------------
    # Private helper functions
    #-----------------------------
    def _load_all_metadata(self) -> list[dict]:
        """Cache all metadata from disk"""
        if hasattr(self, '_metadata_cache'):
            return self._metadata_cache

        root = Path(self.modelevals_dir)
        modelevals = []
        for meta_file in root.glob(f"*/{config.META_INFO_FILENAME}.json"):
            with open(meta_file, "r") as f:
                meta = json.load(f)
                meta['_path'] = str(meta_file.parent)
                modelevals.append(meta)

        self._metadata_cache = sorted(modelevals, key=lambda x: x.get("datetime", ""), reverse=True)
        return self._metadata_cache