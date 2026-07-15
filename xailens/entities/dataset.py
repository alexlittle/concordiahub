"""
Dataset Entity class - used to provide data for display on the dashboard
"""

import pandas as pd

from xailens.exceptions import InvalidSelectionError
from xailens.registry.xai_registry import get_xai_adapter
from xailens.entities.model import Model
from xailens import config

class Dataset:
    """
    Initialises the Dataset object, with the specified dataset and the
        models to include

    Args:
        dataset_id (str): Dataset ID
        models (list[Model]): List of Model objects
    """

    def __init__(self, dataset_id: str, models: list[Model]):
        self.dataset_id = dataset_id
        self.models = models

    def get_recommended_models(self) -> list[str]:
        """
        Returns a list of recommended model names, based on the metric specified
        in the configuration (defaults to f1 weighted)

        Returns a list in case there is a tie between models

        Returns:
            list[str]: recommended model names
        """
        df = self.get_metrics()
        s = df[config.BEST_MODEL_METRIC]
        max_val = s.max()
        best_models = s[s == max_val].index.tolist()

        return best_models

    def get_metrics(self) -> pd.DataFrame:
        """
        Gets the metrics for current dataset

        Returns:
            pd.DataFrame: metrics
        """
        rows = {me.get("model_display_name"): me.get_metrics() for me in self.models}
        df = pd.DataFrame(rows).T
        df.index.name = "model"
        df.columns.name = None
        return df

    def get_instance_data(self, instance_uuid: str) -> pd.Series:
        """
        Gets the instance data for the specified instance uuid

        Args:
            instance_uuid (str): Instance UUID

        Returns:
            pd.Series: instance data
        """

        # only needs the data from the first model - as will be same for all
        return self.models[0].get_instance_data(instance_uuid)

    def _get_explanations(self, instance_uuid: str | None = None) -> pd.DataFrame:
        """
        Private helper function for getting the global explanations (if not instance_uuid given) or
        local explanations if instance_uuid if specified

        Args:
            instance_uuid (str | None): an instance UUID

        Returns:
            pd.DataFrame: explanations

        Raises:
            InvalidSelectionError:
        """
        explanations = {}
        keep_features = set()
        for model in self.models:
            method = model.get("xai_methods")[0]
            adapter = get_xai_adapter(method)
            if instance_uuid is not None:
                df = adapter.load_local_explanation(model, instance_uuid)
            else:
                df = adapter.load_global_explanation(model)

            # remove and leading/trailing whitespace
            df["feature_name"] = df["feature_name"].str.strip()

            # make sure the values are properly numeric
            df["score_normalised_max"] = pd.to_numeric(df["score_normalised_max"], errors="coerce")

            model_name = model.get("model_display_name")
            top_df = (
                df[["feature_name", "score_normalised_max"]]
                .nlargest(config.KEEP_NUM_FEATURES, "score_normalised_max")
                .rename(columns={"score_normalised_max": model_name})
            )
            keep_features.update(top_df["feature_name"].tolist())
            explanations[model_name] = top_df

        if not keep_features:
            raise InvalidSelectionError()

        result = pd.DataFrame({"feature_name": sorted(keep_features)})
        for df in explanations.values():
            result = result.merge(df, on="feature_name", how="left")
        model_cols = [col for col in result.columns if col != "feature_name"]
        result["_mean"] = result[model_cols].abs().fillna(0).mean(axis=1)
        result = (result.sort_values("_mean", ascending=False).drop(columns="_mean").reset_index(drop=True))
        return result[: config.PLOT_MAX_FEATURES].reset_index(drop=True)

    def get_global_explanations(self) -> pd.DataFrame:
        """
        Returns the global explanations for the dataset

        Returns:
            pd.DataFrame: global explanation
        """
        return self._get_explanations()

    def get_local_explanations(self, instance_uuid: str) -> pd.DataFrame:
        """
        Returns the local explanations for the given instance_uuid

        Args:
            instance_uuid (str): instance uuid to get the explanations for

        Returns:
            pd.DataFrame: local explanation
        """
        result = self._get_explanations(instance_uuid)
        model_cols = [col for col in result.columns if col != "feature_name"]
        result[model_cols] = result[model_cols].abs()
        return result

    def get_instance_consensus_ids(self) -> set:
        """
        Returns instance IDs common to all models.

        Returns:
            set: set of instance IDs
        """
        common_ids = None
        for model in self.models:
            y_true, _ = model.load_targets()
            ids = set(y_true[model.get(config.DATA_SCHEMA_ID_COLUMN)].tolist())
            if common_ids is None:
                common_ids = ids
            else:
                common_ids = common_ids.intersection(ids)
        return common_ids

    def get_instance_consensus(self) -> dict:
        """
        Gets metrics on the overall instance consensus across all models for
        this dataset

        Returns:
            dict:
        """
        common_ids = self.get_instance_consensus_ids()
        pred_df, y_true_common = self._get_common_predictions(common_ids)
        correct_df = pred_df.eq(y_true_common, axis=0)
        total = len(common_ids)
        all_correct = int(correct_df.all(axis=1).sum())
        all_incorrect = int((~correct_df).all(axis=1).sum())
        models_disagree = total - all_correct - all_incorrect
        return {
            "total": total,
            "all_correct": all_correct,
            "all_incorrect": all_incorrect,
            "models_disagree": models_disagree,
        }

    def get_all_instances(self) -> pd.DataFrame:
        """
        Gets a dataframe of all the instances for all the models.
            The dataframe includes columns:
            - instance_uuid
            - true_label (ground truth)
            - prediction column for each dataset
            - agreement column (either: allcorrect, allincorrect, disagree)

        Returns:
            pd.DataFrame:
        """

        common_ids = self.get_instance_consensus_ids()
        pred_df, y_true_common = self._get_common_predictions(common_ids)
        correct_df = pred_df.eq(y_true_common, axis=0)
        result = pd.DataFrame({"true_label": y_true_common})
        result = pd.concat([result, pred_df], axis=1)
        result["agreement"] = correct_df.apply(self._agreement, axis=1)

        # sort by true_label
        result = result.sort_values(by=["true_label"], ascending=True)
        result = result.reset_index(names="instance_uuid")
        cols = (
                ["instance_uuid", "true_label"]
                + [c for c in result.columns if
                   c not in ("instance_uuid", "true_label", "agreement")]
                + ["agreement"]
        )
        return result[cols]

    def get_upset_data(self, include_all_correct: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        """
        Builds UpSet-ready data from get_all_instances().

        Args:
            include_all_correct (bool): whether to include all correct/incorrect.
                Defaults to False.


        Returns:
            tuple[pd.DataFrame, pd.Series]: dataframe with one row per unique
                combination of models being wrong, with columns: <model cols...>
                , count, instance_ids and pd.Series total wrong-count per model
        """
        df = self.get_all_instances()

        model_cols = [
            c for c in df.columns
            if c not in ("instance_uuid", "true_label", "agreement")
        ]

        # binary "wrong" matrix: 1 if model's prediction != true_label
        wrong_df = df[model_cols].ne(df["true_label"], axis=0).astype(int)
        wrong_df.index = df["instance_uuid"]

        if not include_all_correct:
            # drop rows where every model got it right, as not needed when
            # wanting to display misclassifications
            wrong_df = wrong_df[wrong_df.sum(axis=1) > 0]

        set_sizes = wrong_df[model_cols].sum().sort_values(ascending=False)

        combo_df = (
            wrong_df.groupby(model_cols)
            .apply(lambda g: list(g.index))
            .reset_index(name="instance_ids")
        )
        combo_df["count"] = combo_df["instance_ids"].apply(len)
        combo_df = combo_df.sort_values("count", ascending=False).reset_index(
            drop=True)

        return combo_df, set_sizes

    def get_instances_for_set(self, set_key: str) -> pd.DataFrame:
        """
        Returns instances where exactly the given models are incorrect.

        Args:
            set_key (str): comma-separated model names, matching the format
                used to build UpSet matrix customdata.

        Returns:
            pd.DataFrame: rows from get_all_instances() matching the combination.
        """
        combo_df, _ = self.get_upset_data()
        model_cols = [c for c in combo_df.columns if
                      c not in ("count", "instance_ids")]
        selected = set(set_key.split(","))

        match = combo_df[
            combo_df[model_cols].apply(
                lambda row: {m for m in model_cols if row[m] == 1} == selected,
                axis=1,
            )
        ]
        if match.empty:
            return self.get_all_instances().iloc[0:0]

        instance_ids = match.iloc[0]["instance_ids"]
        df = self.get_all_instances()
        return df[df["instance_uuid"].isin(instance_ids)]

    #------------------------------
    # Private helper functions
    #-------------------------------
    def _get_common_predictions(self, common_ids: set) -> tuple[pd.DataFrame, pd.Series]:
        """
        Builds prediction (dataframe) and ground-truth (series) for all models,
        filtered to the given common_ids.

        Args:
            common_ids (set): set of the ids in common

        Returns:
            tuple[pd.DataFrame, pd.Series]: pd.DataFrame, one column per model,
                indexed by instance id and pd.Series of ground truth, indexed by instance id
        """
        model_preds = {}
        y_true_common = None
        for model in self.models:
            y_true, y_pred = model.load_targets()
            id_col = model.get(config.DATA_SCHEMA_ID_COLUMN)
            pred_col = model.get(config.DATA_SCHEMA_PREDICTION_COLUMN)
            target_col = model.get(config.DATA_SCHEMA_TARGET_COLUMN)
            y_pred_filtered = (
                y_pred[y_pred[id_col].isin(common_ids)]
                .set_index(id_col)[pred_col]
                .rename(model.get("model_display_name"))
            )
            y_true_filtered = (
                y_true[y_true[id_col].isin(common_ids)]
                .set_index(id_col)[target_col]
            )
            model_preds[model.get("model_display_name")] = y_pred_filtered
            if y_true_common is None:
                y_true_common = y_true_filtered
        return pd.DataFrame(model_preds), y_true_common


    def _agreement(self, row: pd.Series) -> str:
        """
        Private helper function to determine if all models are correct,
        incorrect or disagree.

        Args:
            row (pd.Series): model predictions

        Returns:
            str: either "correct", "allincorrect" or "disagree"
        """
        if row.all():
            return "allcorrect"
        if (~row).all():
            return "allincorrect"
        return "disagree"