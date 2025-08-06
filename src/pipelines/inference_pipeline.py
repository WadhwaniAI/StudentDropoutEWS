import os
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from .base_pipeline import BasePipeline
from src.configs.config_manager import ConfigManager
from src.models.model import EWSModel
from src.models.utils import get_model_features


class InferencePipeline(BasePipeline):
     """Encapsulates the model inference pipeline, inheriting from BasePipeline."""

     def __init__(self, exp_dir: str, inference_data_path: str):
          """Initializes the inference-specific pipeline."""
          super().__init__()
          self.exp_dir = exp_dir
          self.data_path = inference_data_path
          
          # Use ConfigManager to load and validate the config from the experiment directory
          config_path = os.path.join(self.exp_dir, "config.json")
          self.config = ConfigManager(config_input=config_path).get_validated_config()
          
          self.cat_features, self.num_features = get_model_features(dir=self.exp_dir)
          with open(os.path.join(self.exp_dir, "summary_metrics.json"), "r") as f:
               self.summary_metrics = json.load(f)

     def _post_load_and_preprocess_hook(self):
          """(Hook Implementation) Populates datasets after loading."""
          self.datasets[Path(self.data_path).stem] = self.df
          self.df = None # Clear self.df as it's no longer needed

     def _load_model(self):
          """Loads the pre-trained model from the experiment directory."""
          self.model = EWSModel(exp_dir=self.exp_dir, cat_features=self.cat_features, config=self.config)

     @property
     def _manual_thresholds(self) -> Dict[str, float]:
          """Defines thresholds based on metrics loaded from the training run."""
          return {
               "val_max_f1": self.summary_metrics.get("val_threshold_max_f1"),
               "val_max_lift": self.summary_metrics.get("val_threshold_max_lift")
          }

     def _finalize(self):
          """Saves the final inference artifacts."""
          super()._save_artifacts()

     def run(self) -> Tuple[Dict[str, Any], str]:
          """Executes all steps of the inference pipeline in sequence."""
          self._load_and_preprocess() # Inherited from BasePipeline
          self._engineer_features() # Inherited from BasePipeline
          self._load_model()
          self._evaluate() # Inherited from BasePipeline
          self._finalize()
          return self.summary_metrics, self.exp_dir