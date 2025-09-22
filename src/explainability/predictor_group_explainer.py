import json
import os
import pandas as pd
import shap
import numpy as np
import warnings
from typing import Dict, List, Union
from sklearn.metrics import precision_recall_curve
from src.models.model import EWSModel
from src.models.utils import get_model_features
from src.configs.utils import load_config
from src import constants


class PredictorGroupExplainer:
     """
     PredictorGroupExplainer analyzes model predictions and explains them using SHAP values for a given experimental directory and dataset.
     This class is tailored for experiments organized in a directory (`exp_dir`) with saved model, config, and features.
     Returns a dataframe with predictor group contributions, and top driving factors.
     """

     def __init__(
          self, exp_dir: str, df_path: str, predictor_groups: Union[str, Dict[str, List[str]]],
          threshold: float=None, target_recall: float=None
     ):
          """
          Initializes the PredictorGroupExplainer with paths, thresholds, features, and model.
          :param exp_dir (str): Path to experiment directory containing model, config, and metadata.
          :param df_path (str): Path to the dataset to explain.
          :param predictor_groups (str | dict): JSON file path or dict defining predictor groupings.
          :param threshold (float | None): Classification threshold for label assignment.
          :param target_recall (float | None): If no threshold is provided, this recall is used to compute threshold from val set.
          """
          self.exp_dir = exp_dir
          self.df_path = df_path
          self.threshold = threshold
          self.target_recall = target_recall

          if isinstance(predictor_groups, str):
               with open(predictor_groups, "r") as f:
                    self.predictor_groups = json.load(f)
          else:
               self.predictor_groups = predictor_groups

          # Load metadata
          self.cat_features, self.num_features = get_model_features(self.exp_dir)
          self.all_features = self.cat_features + self.num_features
          self._validate_and_prune_predictor_groups()
          self.config = load_config(os.path.join(exp_dir, constants.ModelArtifacts.CONFIG))
          self.model = self._load_model()
          self.df = self._load_dataframe()          

          if self.threshold is None:
               self.threshold = self._compute_threshold()

     def _validate_and_prune_predictor_groups(self) -> None:
          """
          Validates predictor groups, warns about and removes features not present in the model.
          Raises errors for fatal issues like duplicates or empty resulting groups.
          """
          seen_features = set()
          duplicate_features = set()
          invalid_features = set()

          # First pass: Collect all issues
          for group, feats in self.predictor_groups.items():
               if not isinstance(group, str) or not group.strip():
                    raise ValueError(f"Predictor group name must be a non-empty string. Found: {group!r}")

               if not feats:
                    raise ValueError(f"Predictor group '{group}' is empty.")

               for feat in feats:
                    if feat not in self.all_features:
                         invalid_features.add(feat)

                    if feat in seen_features:
                         duplicate_features.add(feat)
                    seen_features.add(feat)

          # Handle fatal errors first
          if duplicate_features:
               raise ValueError(f"The following features occur in multiple predictor groups: {sorted(duplicate_features)}")

          # Handle non-fatal invalid features
          if invalid_features:
               # Warn the user about the missing features
               warnings.warn(
                    f"The following features in predictor_groups are not in the model's features "
                    f"and will be ignored: {sorted(invalid_features)}",
                    UserWarning
               )

               # --- This is the new pruning logic ---
               pruned_groups = {}
               for group, feats in self.predictor_groups.items():
                    # Keep only the features that are valid
                    valid_feats_in_group = [f for f in feats if f not in invalid_features]

                    # If the group is not empty after pruning, add it to our new dict
                    if valid_feats_in_group:
                         pruned_groups[group] = valid_feats_in_group
               
               # Overwrite the original groups with the pruned version
               self.predictor_groups = pruned_groups

          # Final check: After pruning, are there any features left at all?
          if not self.predictor_groups:
               raise ValueError("After removing features not present in the model, no valid feature groups remain. Please check your predictor_groups.")

     def _compute_threshold(self) -> float:
          """Compute threshold based on target recall or best F1 on val set."""
          val_path = os.path.join(self.exp_dir, constants.ModelArtifacts.VALIDATION)
          val_df = pd.read_pickle(val_path)
          assert constants.ColumnNames.PROBA_1 in val_df.columns, f"Validation dataframe must contain '{constants.ColumnNames.PROBA_1}' column."
          probas = val_df[constants.ColumnNames.PROBA_1].astype(np.float64).values
          labels = val_df[self.config.data.label or constants.ColumnNames.LABEL].astype(int).values

          precisions, recalls, thresholds = precision_recall_curve(labels, probas)

          if self.target_recall is not None:
               idx = np.argmin(np.abs(recalls - self.target_recall))
               return thresholds[idx]
          else:
               f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
               return thresholds[np.argmax(f1s)]

     def _load_model(self):
          """Loads the trained CatBoost model from file."""
          clf = EWSModel(self.exp_dir, self.cat_features, self.config)
          clf.model.load_model(os.path.join(self.exp_dir, constants.ModelArtifacts.MODEL))
          return clf.model

     def _load_dataframe(self) -> pd.DataFrame:
          """Loads the input dataframe and casts numeric features to float, and categorical to string."""
          df = pd.read_pickle(self.df_path)
          df[self.num_features] = df[self.num_features].astype(np.float64)
          df[self.cat_features] = df[self.cat_features].astype(str)
          return df

     def _generate_shap_values(self) -> None:
          """Generates and appends SHAP values for all features."""
          explainer = shap.TreeExplainer(self.model)
          shap_vals = explainer(self.df[self.all_features])
          shap_df = pd.DataFrame(
               shap_vals.values,
               columns=[f"{constants.ColumnNames.SHAP_PREFIX}[{f}]" for f in shap_vals.feature_names],
               index=self.df.index
          )
          self.df = pd.concat([self.df, shap_df], axis=1)

     def _add_group_contributions(self) -> None:
          """Adds group-wise contributions and top drivers for both class labels."""
          def contribs(row):
               result = {}
               for label in (0, 1):
                    label_str = constants.ModelConfig.PREDICTION_LABELS[label]
                    for group, feats in self.predictor_groups.items():
                         shap_feats = [f for f in feats if f"{constants.ColumnNames.SHAP_PREFIX}[{f}]" in self.df.columns]
                         shap_vals = [row[f"{constants.ColumnNames.SHAP_PREFIX}[{f}]"] for f in shap_feats]
                         filtered = [(f, s if label else abs(s)) for f, s in zip(shap_feats, shap_vals) if (s > 0 if label else s <= 0)]
                         total = sum(s for _, s in filtered)
                         top = max(filtered, key=lambda x: x[1])[0] if filtered else None
                         result[f"{group}_label{label}_contrib"] = total
                         result[f"{group}_label{label}_top_feat"] = top
                    sorted_groups = sorted(((g, result[f"{g}_label{label}_contrib"]) for g in self.predictor_groups), key=lambda x: x[1], reverse=True)
                    for i, (grp, _) in enumerate(sorted_groups, 1):
                         base_col_name = f"{constants.ColumnNames.PREDICTOR_GROUP_PREFIX}_{i}_for_{label_str}"
                         result[base_col_name] = grp
                         result[f"{base_col_name}{constants.ColumnNames.TOP_DRIVER_SUFFIX}"] = result[f"{grp}_label{label}_top_feat"]
               return pd.Series(result)
          self.df = pd.concat([self.df, self.df.apply(contribs, axis=1)], axis=1)

     def _apply_threshold_and_label(self) -> None:
          """Applies threshold to probabilities and assigns binary class labels."""
          self.df[constants.ColumnNames.PREDICTION] = (self.df[constants.ColumnNames.PROBA_1] >= self.threshold).astype(int).map(constants.ModelConfig.PREDICTION_LABELS)

     def _select_top_predictors(self) -> None:
          """Adds top predictor groups and their top drivers based on predicted label."""
          def extract(row):
               label = row[constants.ColumnNames.PREDICTION]
               result = {}
               for i in range(1, len(self.predictor_groups) + 1):
                    base_col = f"{constants.ColumnNames.PREDICTOR_GROUP_PREFIX}_{i}"
                    result[base_col] = row.get(f"{base_col}_for_{label}")
                    result[f"{base_col}{constants.ColumnNames.TOP_DRIVER_SUFFIX}"] = row.get(f"{base_col}_for_{label}{constants.ColumnNames.TOP_DRIVER_SUFFIX}")
               return pd.Series(result)
          self.df = pd.concat([self.df, self.df.apply(extract, axis=1)], axis=1)

     def run(self) -> pd.DataFrame:
          """Runs the full SHAP explanation pipeline and returns the enriched dataframe."""
          self._generate_shap_values()
          self._add_group_contributions()
          self._apply_threshold_and_label()
          self._select_top_predictors()
          return self.df