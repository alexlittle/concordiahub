import os
import json

from collections import defaultdict
from pathlib import Path
from itertools import combinations

from xailens import config, strings
from xailens.entities.modeleval import ModelEval
from xailens.formatting import format_model_name


class ModelExplorer:

    def __init__(self):
        """Initializes ModelExplorer"""
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.modelevals_dir = os.path.join(BASE_DIR, config.MODEL_DATA_DIR_NAME)
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
                    models[format_model_name(modeleval_obj)] = modeleval["model_uuid"]
        return dict(sorted(models.items()))

    def get_comparable_modelevals(self, n: int = 50) -> list[list[dict]]:
        """Returns pairs of model_data that share a dataset but have different model types
        Args:
            n: number of comparison groups to return
        Returns:
            list of groups, each group is a pair of metadata dicts with same dataset, different models
        """
        datasets = defaultdict(dict)
        for modeleval in self._load_all_metadata():
            dataset = modeleval["dataset_id"]
            model = modeleval["model_type"]
            if modeleval["model_subtype"]:
                model = f"{model}_{modeleval['model_subtype']}"
            modeleval_obj = self.get_modeleval_by_uuid(modeleval["model_uuid"])
            if model not in datasets[dataset]:
                datasets[dataset][model] = modeleval_obj

        pairs = []
        for models in datasets.values():
            if len(models) > 1:
                for pair in combinations(models.values(), 2):
                    pairs.append(list(pair))

        return sorted(pairs, key=lambda g: max(r.datetime for r in g), reverse=True)[:n]

    def get_dataset_filter_options(self) -> dict[str, str]:
        """Returns dataset filter options based on available datasets in metadata."""
        datasets = sorted({
            modeleval["dataset_id"]
            for modeleval in self._load_all_metadata()
        })

        return {strings.ALL_DATASET_OPTIONS[0]: strings.ALL_DATASET_OPTIONS[1]} | {d: d for d in datasets}

    def get_model_filter_options(self) -> dict[str, str]:
        datasets = defaultdict(list)
        for modeleval in self._load_all_metadata():
            datasets[modeleval["dataset_id"]].append(modeleval["model_type"])

        valid_pairs = set()
        for model_types in datasets.values():
            types = sorted(model_types)
            for a, b in combinations(types, 2):
                valid_pairs.add((a, b))
            for t in set(types):
                if types.count(t) >= 2:
                    valid_pairs.add((t, t))

        options = {strings.ALL_COMP_OPTIONS[0]: strings.ALL_COMP_OPTIONS[1]}
        for a, b in sorted(valid_pairs):
            options[f"{a}_vs_{b}"] = f"{strings.MODEL_DISPLAY_NAME[a]} vs {strings.MODEL_DISPLAY_NAME[b]}"

        return options

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

        runs = []
        for u in uuids:
            meta = self._uuid_index.get(u)
            if not meta:
                continue
            runs.append(ModelEval(path=meta["_path"], metadata=meta))
        return runs

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