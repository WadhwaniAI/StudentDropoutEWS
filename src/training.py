import os
import pandas as pd
from munch import Munch
from src.data.preprocess import DataPreprocessor
from src.data.engineer_attendance_features import EngineerAttendanceFeatures
from src.models.model import CatBoostBinaryClassifier
from src.models.utils import save_model_features, loss_curves
from src.analysis.metrics import BinaryModelEvaluator
from src.data.utils import sample_and_split, extract_academic_year_from_path


def training_pipeline(config: Munch, exp_dir: str) -> tuple[dict, dict]:
     """
     Runs the full training pipeline and evaluates on training and validation datasets.

     :param config: Parsed configuration as a Munch dictionary for dotted access
     :param exp_dir: Directory to save experiment outputs
     :return: (training_summary, wandb_metrics) - dictionaries with key training info
     """

     # Create output directory
     os.makedirs(exp_dir, exist_ok=True)

     # Load training data
     df = pd.read_pickle(config.data.training_data_path)

     # Preprocess
     preprocessor = DataPreprocessor()
     df, column_groups = preprocessor.preprocess(
          df=df,
          column_filters=config.data.column_filters,
          index=config.data.index,
          label=config.data.label
     )

     # Sample and split
     df_train, df_val = sample_and_split(
          df=df, 
          label=config.data.label, 
          sampling_prevalence=config.data.sample.p, 
          sample_seed=config.data.sample.seed, 
          train_size=config.data.split.train_size, 
          split_seed=config.data.split.random_state, 
          shuffle=config.data.split.shuffle,
     )

     # Initialize and configure feature engineering
     feature_engineer = EngineerAttendanceFeatures(
          holidays_calendar_path=config.data.holidays_calendar_path,
          index=config.data.index,
          label=config.data.label
     )
     feature_engineer.configure_features(**config.data.engineer_features)

     # Generate features for train and val
     datasets = {}
     datasets["train"], cat_features, num_features = feature_engineer.generate_features(
          df_train,
          acad_year=extract_academic_year_from_path(config.data.training_data_path),
          drop_columns_or_groups=config.data.drop_columns_or_groups,
          column_groups=column_groups
     )
     datasets["val"], _, _ = feature_engineer.generate_features(
          df_val,
          acad_year=extract_academic_year_from_path(config.data.training_data_path),
          drop_columns_or_groups=config.data.drop_columns_or_groups,
          column_groups=column_groups
     )

     # Save features
     save_model_features(exp_dir=exp_dir, cat_features=cat_features, num_features=num_features)

     # Train model
     model = CatBoostBinaryClassifier(
          exp_dir=exp_dir,
          cat_features=cat_features,
          config=config
     )
     best_params, val_threshold_max_f1, val_threshold_max_lift = model.fit(
          x_train=datasets["train"][cat_features + num_features],
          y_train=datasets["train"][config.data.label],
          x_val=datasets["val"][cat_features + num_features],
          y_val=datasets["val"][config.data.label]
     )

     # Predict, save, and evaluate on train and val
     wandb_metrics = {}
     for split, df_split in datasets.items():
          df_preds = model.predict(x=df_split, features=cat_features + num_features)
          df_preds.to_pickle(f"{exp_dir}/{split}.pkl")

          evaluator = BinaryModelEvaluator(
               df=df_preds,
               label_col=config.data.label,
               proba_1_col="preds_proba_1",
               ds_name=split,
               save_dir=exp_dir,
               manual_thresholds={
                    "max_f1 (val)": val_threshold_max_f1,
                    "max_lift (val)": val_threshold_max_lift
               }
          )

          metrics = evaluator.summary_metrics()
          evaluator.plot_all()
          wandb_metrics.update(metrics)

     # Additional metrics to log on WandB
     wandb_metrics.update({
          "best_params": best_params,
          "val_threshold_max_f1": val_threshold_max_f1,
          "val_threshold_max_lift": val_threshold_max_lift,
          "shape_train": df_train.shape,
          "shape_val": df_val.shape,
          "categorical_features": cat_features,
          "numerical_features": num_features,
          "n_categorical_features": len(cat_features),
          "n_numerical_features": len(num_features),
          "loss_curves": loss_curves(train_dir=exp_dir)
     })

     # Final summary for CLI / main
     training_summary = {
          "best_params": best_params,
          "val_optimal_thresholds": {
               "val_threshold_max_f1": val_threshold_max_f1, 
               "val_threshold_max_lift": val_threshold_max_lift
          },
          "exp_dir": exp_dir
     }

     return training_summary, wandb_metrics