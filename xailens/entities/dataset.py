import pandas as pd

from xailens.exceptions import InvalidSelectionError
from xailens.registry.xai_registry import get_xai_adapter
from xailens.entities.modeleval import ModelEval
from xailens import constants


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

    def get_global_explanations(self) -> pd.DataFrame:
        explanations = {}
        keep_features = set()

        for modeleval in self.modelevals:
            method = modeleval.get("xai_methods")[0]
            adapter = get_xai_adapter(method)
            df = adapter.load_global_explanation(modeleval)

            schema = modeleval.get_data_schema()
            if schema.hide_zero_global_threshold is not None:
                filtered_df = df[df["score_normalised_max"] > schema.hide_zero_global_threshold]
            else:
                filtered_df = df

            keep_features.update(filtered_df["feature_name"].tolist())
            explanations[modeleval.get("model_display_name")] = df[
                ["feature_name", "score_normalised_max"]].rename(
                columns={
                    "score_normalised_max": modeleval.get("model_display_name")}
            )

        # if no features to show (ie all LLMs) then raise error
        if not keep_features:
            raise InvalidSelectionError()

        all_features_df = pd.DataFrame({"feature_name": sorted(keep_features)})

        result = all_features_df
        for model_id, df in explanations.items():
            result = result.merge(df, on="feature_name", how="left")


        # order by the mean so most important features are at top
        model_cols = [col for col in result.columns if col != "feature_name"]
        result["_mean"] = result[model_cols].mean(axis=1, skipna=True)
        result = result.sort_values("_mean", ascending=False).drop(columns="_mean").reset_index(drop=True)

        return result[:25].reset_index(drop=True)

    def get_instance_consensus_ids(self) -> pd.Index:
        """Returns instance IDs common to all models."""
        common_ids = None
        for me in self.modelevals:
            y_true, _ = me.load_targets()
            ids = set(y_true[me.get(constants.DATA_SCHEMA_ID_COLUMN)].tolist())
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
            id_col = me.get(constants.DATA_SCHEMA_ID_COLUMN)
            pred_col = me.get(constants.DATA_SCHEMA_PREDICTION_COLUMN)
            target_col = me.get(constants.DATA_SCHEMA_TARGET_COLUMN)

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
            id_col = me.get(constants.DATA_SCHEMA_ID_COLUMN)
            pred_col = me.get(constants.DATA_SCHEMA_PREDICTION_COLUMN)
            target_col = me.get(constants.DATA_SCHEMA_TARGET_COLUMN)

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

        return result

    # Private helper functions
    def _agreement(self, row: pd.Series) -> str:
        if row.all():
            return "allcorrect"
        if (~row).all():
            return "allincorrect"
        return "disagree"