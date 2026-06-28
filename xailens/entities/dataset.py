import pandas as pd

from xailens.exceptions import InvalidSelectionError
from xailens.registry.xai_registry import get_xai_adapter
from xailens.entities.modeleval import ModelEval
from xailens import config

class Dataset:

    def __init__(self, dataset_id: str,  modelevals: list[ModelEval]):
        self.dataset_id = dataset_id
        self.modelevals = modelevals

    def get_metrics(self) -> pd.DataFrame:
        rows = {me.get("model_display_name"): me.get_metrics() for me in self.modelevals}
        df = pd.DataFrame(rows).T
        df.index.name = "model"
        df.columns.name = None
        return df

    def get_instance_data(self, instance_uuid: str) -> tuple[pd.Series, str]:
        # only needs the data from the first modeleval - as will be same for all
        return self.modelevals[0].get_instance_data(instance_uuid)

    def _get_explanations(self, instance_uuid: str | None = None) -> pd.DataFrame:
        explanations = {}
        keep_features = set()
        for modeleval in self.modelevals:
            method = modeleval.get("xai_methods")[0]
            adapter = get_xai_adapter(method)
            if instance_uuid is not None:
                df = adapter.load_local_explanation(modeleval, instance_uuid)
            else:
                df = adapter.load_global_explanation(modeleval)

            # remove and leading/trailing whitespace
            df["feature_name"] = df["feature_name"].str.strip()

            # make sure the values are properly numeric
            df["score_normalised_max"] = pd.to_numeric(df["score_normalised_max"], errors="coerce")

            model_name = modeleval.get("model_display_name")
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
        return self._get_explanations()

    def get_local_explanations(self, instance_uuid: str) -> pd.DataFrame:
        result = self._get_explanations(instance_uuid)
        model_cols = [col for col in result.columns if col != "feature_name"]
        result[model_cols] = result[model_cols].abs()
        return result

    def get_instance_consensus_ids(self) -> pd.Index:
        """Returns instance IDs common to all models."""
        common_ids = None
        for me in self.modelevals:
            y_true, _ = me.load_targets()
            ids = set(y_true[me.get(config.DATA_SCHEMA_ID_COLUMN)].tolist())
            if common_ids is None:
                common_ids = ids
            else:
                common_ids = common_ids.intersection(ids)
        return common_ids

    def get_instance_consensus(self) -> dict:
        common_ids = self.get_instance_consensus_ids()

        # For each model, get predictions for the common instance IDs
        model_preds = {}
        y_true_common = None
        for me in self.modelevals:
            y_true, y_pred = me.load_targets()
            id_col = me.get(config.DATA_SCHEMA_ID_COLUMN)
            pred_col = me.get(config.DATA_SCHEMA_PREDICTION_COLUMN)
            target_col = me.get(config.DATA_SCHEMA_TARGET_COLUMN)

            y_pred_filtered = y_pred[y_pred[id_col].isin(common_ids)]
            y_true_filtered = y_true[y_true[id_col].isin(common_ids)]

            model_preds[me.get("model_display_name")] = {
                "preds": y_pred_filtered.set_index(id_col),
                "pred_col": pred_col,
                "target_col": target_col,
            }
            if y_true_common is None:
                y_true_common = y_true_filtered.set_index(id_col)
                target_col_common = target_col

        correct_df = pd.DataFrame({
            model_id: (data["preds"][data["pred_col"]].values == y_true_common[target_col_common].values)
            for model_id, data in model_preds.items()
        }, index=y_true_common.index)

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
        """ Gets a dataframe of all the instances for all the modelevals.
        The dataframe includes columns:
        - instance_uuid
        - true_label (ground truth)
        - prediction column for each dataset
        - agreement column (either: allcorrect, allincorrect, disagree)

        Returns:
            pd.DataFrame
        """

        common_ids = self.get_instance_consensus_ids()

        model_preds = {}
        y_true_common = None
        target_col_common = None

        for me in self.modelevals:
            y_true, y_pred = me.load_targets()
            id_col = me.get(config.DATA_SCHEMA_ID_COLUMN)
            pred_col = me.get(config.DATA_SCHEMA_PREDICTION_COLUMN)
            target_col = me.get(config.DATA_SCHEMA_TARGET_COLUMN)

            y_pred_filtered = y_pred[y_pred[id_col].isin(common_ids)]
            y_true_filtered = y_true[y_true[id_col].isin(common_ids)]

            model_preds[me.get("model_display_name")] = {
                "preds": y_pred_filtered.set_index(id_col),
                "pred_col": pred_col,
            }
            if y_true_common is None:
                y_true_common = y_true_filtered.set_index(id_col)
                target_col_common = target_col

        correct_df = pd.DataFrame({
                model_id: (data["preds"][data["pred_col"]].values == y_true_common[target_col_common].values
                ) for model_id, data in model_preds.items()
            }, index=y_true_common.index)

        pred_df = pd.DataFrame({
                model_id: data["preds"][data["pred_col"]].values for model_id, data in model_preds.items()
            }, index=y_true_common.index)

        result = pd.DataFrame({"true_label": y_true_common[target_col_common]})
        result = pd.concat([result, pred_df], axis=1)
        result["agreement"] = correct_df.apply(self._agreement, axis=1)

        # sort by true_label
        result = result.sort_values(by=["true_label"], ascending=True)
        result = result.reset_index(names="instance_uuid")
        cols = ["instance_uuid", "true_label"] \
               + [c for c in result.columns if c not in ("instance_uuid",
                                                            "true_label",
                                                            "agreement")] + [
                   "agreement"]
        return result[cols]

    # Private helper functions
    def _agreement(self, row: pd.Series) -> str:
        if row.all():
            return "allcorrect"
        if (~row).all():
            return "allincorrect"
        return "disagree"