import json
import numpy as np, pandas as pd, optuna
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from .utils import (
     get_optuna_suggestions, address_device_compatibility,
     filter_valid_params, get_optimal_thresholds
)
from .calibration import IsotonicCalibrator
from src import constants


class EWSModel:
     """Wrapper for CatBoost binary classification with tuning and calibration."""

     def __init__(self, exp_dir: str, cat_features: list, config: dict) -> None:
          """
          Initialize CatBoostBinaryClassifier with config, directory, and categorical features.
          :param exp_dir (str): Directory where model artifacts are stored.
          :param cat_features (list): List of categorical feature column names.
          :param config (dict): Configuration dictionary for model and tuning.
          """
          self.exp_dir = exp_dir
          self.cat_features = cat_features
          self.config = config
          self.best_params = config.model.params.fixed
          self.trial_params = {}

          self.model = CatBoostClassifier(
               train_dir=exp_dir, cat_features=cat_features, **self.best_params
          )
          self.calibrator = IsotonicCalibrator(
               f"{self.exp_dir}/{constants.ModelArtifacts.CALIBRATOR}"
          )

     def fit(self, x_train, y_train, x_val, y_val):
          """
          Trains model, applies tuning and calibration, and saves model artifacts.
          :param x_train (pd.DataFrame): Training feature matrix.
          :param y_train (pd.Series): Training labels.
          :param x_val (pd.DataFrame): Validation feature matrix.
          :param y_val (pd.Series): Validation labels.
          Returns: Tuple of best_params (dict), threshold_max_f1 (float), threshold_max_lift (float)
          """
          if getattr(self.config.model.params, "tune", None) is not None:
               self.tune(x_train, y_train, x_val, y_val)
               with open(f"{self.exp_dir}/{constants.ModelArtifacts.TRIAL_PARAMS}", "w") as f:
                    json.dump(self.trial_params, f, indent=constants.ConfigSchema.JSON_INDENT)

          self.model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=True)
          y_val_score = self.model.predict_proba(x_val)[:, 1]

          df_val = pd.DataFrame(
               {
                    constants.ColumnNames.PROBA_0: 1 - y_val_score,
                    constants.ColumnNames.PROBA_1: y_val_score,
                    self.config.data.label or constants.ColumnNames.LABEL: y_val,
               }
          )
          df_val.to_csv(
               f"{self.exp_dir}/{constants.ModelArtifacts.PRECALIBRATION_SCORES}",
               index=False,
          )

          y_val_score = self.calibrate(y_val, y_val_score)

          df_val = pd.DataFrame(
               {
                    constants.ColumnNames.PROBA_0: 1 - y_val_score,
                    constants.ColumnNames.PROBA_1: y_val_score,
                    self.config.data.label or constants.ColumnNames.LABEL: y_val,
               }
          )
          df_val.to_csv(
               f"{self.exp_dir}/{constants.ModelArtifacts.POSTCALIBRATION_SCORES}",
               index=False,
          )

          th_f1, th_lift = get_optimal_thresholds(
               df=df_val,
               proba_1_col=constants.ColumnNames.PROBA_1,
               target_col=self.config.data.label or constants.ColumnNames.LABEL,
          )
          self.model.set_probability_threshold(th_f1)
          self.optimal_threshold = self.model.get_probability_threshold()
          self.model.save_model(f"{self.exp_dir}/{constants.ModelArtifacts.MODEL}")

          return self.best_params, th_f1, th_lift

     def predict(self, x: pd.DataFrame, features: list) -> pd.DataFrame:
          """
          Predicts probabilities and classes using calibrated model.
          :param x (pd.DataFrame): Input data for prediction.
          :param features (list): Feature columns to use for prediction.
          Returns: Input dataframe with appended prediction columns.
          """
          self.model.load_model(f"{self.exp_dir}/{constants.ModelArtifacts.MODEL}")
          self.optimal_threshold = self.model.get_probability_threshold()

          scores = self.model.predict_proba(x[features])
          scores[:, 1] = self.calibrator.transform(scores[:, 1])
          scores[:, 0] = 1 - scores[:, 1]

          preds = (scores[:, 1] >= self.optimal_threshold).astype(int)

          df_out = pd.DataFrame(
               {
                    constants.ColumnNames.PROBA_0: scores[:, 0],
                    constants.ColumnNames.PROBA_1: scores[:, 1],
                    constants.ColumnNames.PREDICTION: preds,
               }
          )

          return pd.concat([x, df_out], axis=1)

     def tune(self, x_train, y_train, x_val, y_val) -> None:
          """
          Tunes model hyperparameters using Optuna and selects best configuration.
          :param x_train (pd.DataFrame): Training features.
          :param y_train (pd.Series): Training labels.
          :param x_val (pd.DataFrame): Validation features.
          :param y_val (pd.Series): Validation labels.
          Returns: None
          """
          def objective(trial):
               params = dict(train_dir=self.exp_dir, cat_features=self.cat_features)
               params.update(self.config.model.params.fixed)

               for k, v in self.config.model.params.tune.independent.items():
                    params[k] = get_optuna_suggestions(trial, k, v["dtype"], v["tuning_space"])

               for k, v in self.config.model.params.tune.dependent.items():
                    if params[v["dependent_on_param"]] in v["dependent_on_value"]:
                         params[k] = get_optuna_suggestions(trial, k, v["dtype"], v["tuning_space"])

               self.trial_params[trial.number] = params
               model = CatBoostClassifier(**filter_valid_params(address_device_compatibility(params)))
               model.fit(x_train, y_train)

               proba = model.predict_proba(x_val)[:, 1]
               precision, recall, _ = precision_recall_curve(y_val, proba)
               return metrics.auc(recall, precision)

          study = optuna.create_study(direction="maximize")
          study.optimize(objective, n_trials=self.config.model.n_trials)
          self.best_params = study.best_params

          self.model = CatBoostClassifier(
               train_dir=self.exp_dir,
               cat_features=self.cat_features,
               **self.best_params,
          )

     def calibrate(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
          """
          Fits and applies isotonic regression to prediction scores.
          :param y_true (pd.Series): True labels for calibration set.
          :param y_pred (np.ndarray): Uncalibrated prediction scores.
          Returns: Calibrated prediction scores.
          """
          return self.calibrator.fit_transform(
               y_true,
               y_pred,
               self.config.model.calibration_nbins or constants.ModelConfig.CALIBRATION_BINS,
          )