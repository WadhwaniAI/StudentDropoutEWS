import abc
import os
import json
import pandas as pd
from munch import Munch
from typing import Any, Dict, List, Tuple
from src.data.preprocess import DataPreprocessor
from src.data.engineer_attendance_features import EngineerAttendanceFeatures
from src.analysis.metrics import BinaryModelEvaluator
from src.data.utils import extract_academic_year_from_path
from src import constants


class BasePipeline(abc.ABC):
     """An abstract base class for ML pipelines, containing shared logic."""

     def __init__(self):
          """Initializes common state variables for all pipelines."""
          self.exp_dir: str | None = None
          self.config: Munch | None = None
          self.data_path: str | None = None
          self.model = None
          self.df: pd.DataFrame | None = None
          self.datasets: Dict[str, pd.DataFrame] = {}
          self.column_groups: Dict[str, List[str]] | None = None
          self.cat_features: List[str] = []
          self.num_features: List[str] = []
          self.summary_metrics: Dict[str, Any] = {}
          self.json_indent: int = constants.ConfigSchema.JSON_INDENT

     def _load_and_preprocess(self):
          """(Shared) Loads and preprocesses data using the Template Method pattern."""
          raw_df = pd.read_pickle(self.data_path)
          preprocessor = DataPreprocessor()
          self.df, self.column_groups = preprocessor.preprocess(
               df=raw_df, column_filters=self.config.data.column_filters,
               index=self.config.data.index, label=self.config.data.label
          )
          # Call the hook for child-specific post-processing
          self._post_load_and_preprocess_hook()

     @abc.abstractmethod
     def _post_load_and_preprocess_hook(self):
          """(Hook) A method for child classes to implement custom post-processing."""
          raise NotImplementedError

     def _engineer_features(self):
          """(Shared) Generates features for all datasets."""
          feature_engineer = EngineerAttendanceFeatures(
               holidays_calendar_path=self.config.data.holidays_calendar_path,
               index=self.config.data.index, label=self.config.data.label
          )
          feature_engineer.configure_features(**self.config.data.engineer_features)

          for split_name, split_df in self.datasets.items():
               self.datasets[split_name], cat_gen, num_gen = feature_engineer.generate_features(
                    df=split_df, acad_year=extract_academic_year_from_path(self.data_path),
                    column_groups=self.column_groups, drop_columns_or_groups=self.config.data.drop_columns_or_groups
               )
               if not self.cat_features and not self.num_features:
                    self.cat_features, self.num_features = cat_gen, num_gen

     def _evaluate(self):
          """(Shared) Generates predictions and evaluates the model on all datasets."""
          for split, df_split in self.datasets.items():
               df_preds = self.model.predict(x=df_split, features=self.cat_features + self.num_features)
               self.datasets[split] = df_preds
               
               evaluator = BinaryModelEvaluator(
                    df=df_preds, 
                    label_col=self.config.data.label, 
                    proba_1_col=constants.ColumnNames.PROBA_1, 
                    ds_name=split, 
                    save_dir=self.exp_dir, 
                    manual_thresholds=self._manual_thresholds
               )
               evaluator.plot_all()
               self.summary_metrics.update(evaluator.summary_metrics())

     def _save_artifacts(self):
          """(Shared) Saves final datasets and summary metrics to disk."""
          for split_name, final_df in self.datasets.items():
               final_df.to_pickle(f"{self.exp_dir}/{split_name}{constants.FileExtensions.PICKLE}")
          with open(os.path.join(self.exp_dir, constants.ModelArtifacts.SUMMARY_METRICS), "w") as f:
               json.dump(self.summary_metrics, f, indent=self.json_indent)

     @property
     @abc.abstractmethod
     def _manual_thresholds(self) -> Dict[str, float]:
          """An abstract property for child classes to define their thresholds."""
          raise NotImplementedError

     @abc.abstractmethod
     def run(self) -> Tuple[Dict[str, Any], str]:
          """An abstract method to run the specific pipeline steps."""
          raise NotImplementedError