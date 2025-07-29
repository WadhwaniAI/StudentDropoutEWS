import json
import os
import pandas as pd
import shap
from typing import Dict, List
from models.model import CatBoostBinaryClassifier
from models.utils import get_model_features
from utils import load_config


class SHAPPipeline:
     """
     SHAPPipeline analyzes model predictions and explains them using SHAP values for a given experimental directory and dataset.
     This class is tailored for experiments organized in a directory (`exp_dir`) with saved model, config, thresholds, and features.
     Supports feature grouping, threshold-based prediction labeling, and identification of top contributing predictor groups
     and features. 

     Returns:
          pd.DataFrame: Dataframe with SHAP scores, predictor group contributions, top driving features and predicted labels.
     """

     def __init__(
          self, exp_dir: str, df_path: str, predictor_groups: Dict[str, List[str]], target_recall: float, target_ds_name: str
     ):
          """Initializes the SHAPPipeline with paths, thresholds, features, and model."""
          self.exp_dir = exp_dir
          self.df_path = df_path
          self.predictor_groups = predictor_groups
          self.target_recall = target_recall
          self.target_ds_name = target_ds_name

          # Load metadata
          self.cat_features, self.num_features, self.all_features = get_model_features(self.exp_dir)
          self.thresholds = self._load_thresholds()
          self.threshold = self._get_threshold()
          self.model = self._load_model()
          self.df = self._load_dataframe()
          self.config = load_config(os.path.join(exp_dir, "config.json"))

     def _load_thresholds(self) -> Dict:
          """Loads threshold values from file."""
          with open(os.path.join(self.exp_dir, "thresholds.json")) as f:
               return json.load(f)

     def _get_threshold(self) -> float:
          """Retrieves the threshold for the target recall and dataset."""
          key = f"{self.target_ds_name}, recall={self.target_recall}"
          threshold = self.thresholds.get(key)
          if threshold is None:
               raise ValueError(f"No matching threshold found for key: {key}")
          return threshold

     def _load_model(self):
          """Loads the trained CatBoost model from file."""
          clf = CatBoostBinaryClassifier(self.exp_dir, self.cat_features, self.config)
          clf.model.load_model(os.path.join(self.exp_dir, "model.cbm"))
          return clf.model

     def _load_dataframe(self) -> pd.DataFrame:
          """Loads the input dataframe and casts numeric features to float."""
          df = pd.read_pickle(self.df_path)
          df[self.num_features] = df[self.num_features].astype("float64")
          return df

     def _generate_shap_values(self) -> None:
          """Generates and appends SHAP values for all features."""
          explainer = shap.TreeExplainer(self.model)
          shap_vals = explainer(self.df[self.all_features])
          shap_df = pd.DataFrame(shap_vals.values, columns=[f"shap[{f}]" for f in shap_vals.feature_names], index=self.df.index)
          self.df = pd.concat([self.df, shap_df], axis=1)

     def _add_group_contributions(self) -> None:
          """Adds group-wise contributions and top drivers for both class labels."""
          def contribs(row):
               result = {}
               for label in (0, 1):
                    label_str = "dropout" if label else "notdropout"
                    for group, feats in self.predictor_groups.items():
                         shap_feats = [f for f in feats if f"shap[{f}]" in self.df.columns]
                         shap_vals = [row[f"shap[{f}]"] for f in shap_feats]
                         filtered = [(f, s if label else abs(s)) for f, s in zip(shap_feats, shap_vals) if (s > 0 if label else s <= 0)]
                         total = sum(s for _, s in filtered)
                         top = max(filtered, key=lambda x: x[1])[0] if filtered else None
                         result[f"{group}_label{label}_contrib"] = total
                         result[f"{group}_label{label}_top_feat"] = top
                    sorted_groups = sorted(((g, result[f"{g}_label{label}_contrib"]) for g in self.predictor_groups), key=lambda x: x[1], reverse=True)
                    for i, (grp, _) in enumerate(sorted_groups, 1):
                         result[f"predictor_group_{i}_for_{label_str}"] = grp
                         result[f"predictor_group_{i}_for_{label_str}_top_driver"] = result[f"{grp}_label{label}_top_feat"]
               return pd.Series(result)
          self.df = pd.concat([self.df, self.df.apply(contribs, axis=1)], axis=1)

     def _apply_threshold_and_label(self) -> None:
          """Applies threshold to probabilities and assigns binary class labels."""
          self.df["prediction"] = (self.df["preds_proba_1"] >= self.threshold).astype(int).map({0: "notdropout", 1: "dropout"})

     def _select_top_predictors(self) -> None:
          """Adds top predictor groups and their top drivers based on predicted label."""
          def extract(row):
               label = row["prediction"]
               result = {}
               for i in range(1, len(self.predictor_groups) + 1):
                    result[f"predictor_group_{i}"] = row.get(f"predictor_group_{i}_for_{label}")
                    result[f"predictor_group_{i}_top_driver"] = row.get(f"predictor_group_{i}_for_{label}_top_driver")
               return pd.Series(result)
          self.df = pd.concat([self.df, self.df.apply(extract, axis=1)], axis=1)

     def run(self) -> pd.DataFrame:
          """Runs the full SHAP explanation pipeline and returns the enriched dataframe."""
          self._generate_shap_values()
          self._add_group_contributions()
          self._apply_threshold_and_label()
          self._select_top_predictors()
          return self.df