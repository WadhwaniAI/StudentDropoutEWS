import os
import shutil
import wandb
from typing import Any, Dict, Tuple
from .base_pipeline import BasePipeline
from src.configs.config_manager import ConfigManager
from src.models.model import EWSModel
from src.models.utils import save_model_features, loss_curves
from src.data.utils import sample_and_split
from src.utils import get_timestamp


# --- Constants ---
CONFIG_FILENAME = "config.json"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

class TrainingPipeline(BasePipeline):
     """Encapsulates the model training pipeline, inheriting from BasePipeline."""

     def __init__(self, config_path: str):
          """Initializes the training-specific pipeline."""
          super().__init__()
          
          # Use ConfigManager to load and validate the config from the experiment directory
          self.config_path = config_path
          self.config = ConfigManager(config_input=self.config_path).get_validated_config()
          
          self.exp_dir = f"{self.config.exp.root_exps}/{self.config.exp.title}_{get_timestamp()}"
          self.data_path = self.config.data.training_data_path

     def _post_load_and_preprocess_hook(self):
          """(Hook Implementation) No action needed after loading for training."""
          pass # The next step, _sample_and_split, will use self.df

     def _setup_experiment(self):
          """Creates experiment directory, saves config, and initializes wandb."""
          os.makedirs(self.exp_dir, exist_ok=True)
          shutil.copyfile(self.config_path, os.path.join(self.exp_dir, CONFIG_FILENAME))
          wandb.init(
               project=self.config.exp.project, config=self.config, 
               name=os.path.basename(self.exp_dir), config_exclude_keys=['exp']
          )
     
     def _sample_and_split(self):
          """Splits the processed dataframe into training and validation sets."""
          df_train, df_val = sample_and_split(
               df=self.df, label=self.config.data.label, sampling_prevalence=self.config.data.sample.p, 
               sample_seed=self.config.data.sample.seed, train_size=self.config.data.split.train_size, 
               split_seed=self.config.data.split.random_state, shuffle=self.config.data.split.shuffle,
          )
          self.datasets = {TRAIN_SPLIT: df_train, VAL_SPLIT: df_val}
          self.df = None # No longer needed after splitting
     
     def _train_model(self):
          """Initializes and trains the model."""
          save_model_features(exp_dir=self.exp_dir, cat_features=self.cat_features, num_features=self.num_features)
          self.model = EWSModel(exp_dir=self.exp_dir, cat_features=self.cat_features, config=self.config)
          
          best_params, val_thresh_max_f1, val_thresh_max_lift = self.model.fit(
               x_train=self.datasets[TRAIN_SPLIT][self.cat_features + self.num_features], 
               y_train=self.datasets[TRAIN_SPLIT][self.config.data.label],
               x_val=self.datasets[VAL_SPLIT][self.cat_features + self.num_features], 
               y_val=self.datasets[VAL_SPLIT][self.config.data.label]
          )
          loss_curves(train_dir=self.exp_dir)
          self.summary_metrics.update({
               "best_params": best_params, "shape_training_data": self.datasets[TRAIN_SPLIT].shape,
               "val_threshold_max_f1": val_thresh_max_f1, "val_threshold_max_lift": val_thresh_max_lift,
               "categorical_features": self.cat_features, "n_categorical_features": len(self.cat_features),
               "numerical_features": self.num_features, "n_numerical_features": len(self.num_features)
          })

     @property
     def _manual_thresholds(self) -> Dict[str, float]:
          """Defines thresholds based on validation set performance."""
          return {
               "max_f1 (val)": self.summary_metrics.get("val_threshold_max_f1"),
               "max_lift (val)": self.summary_metrics.get("val_threshold_max_lift")
          }

     def _finalize(self):
          """Saves artifacts and handles training-specific finalization (wandb)."""
          super()._save_artifacts()
          wandb.log(self.summary_metrics)
          wandb.finish()

     def run(self) -> Tuple[Dict[str, Any], str]:
          """Executes all steps of the training pipeline in sequence."""
          self._setup_experiment()
          self._load_and_preprocess() # Inherited from BasePipeline
          self._sample_and_split()
          self._engineer_features() # Inherited from BasePipeline
          self._train_model()
          self._evaluate() # Inherited from BasePipeline
          self._finalize()
          return self.summary_metrics, self.exp_dir