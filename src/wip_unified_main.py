import argparse
import json
import os
import pandas as pd
import shutil
import wandb
from pathlib import Path
from src.data.preprocess import DataPreprocessor
from src.data.engineer_attendance_features import EngineerAttendanceFeatures
from src.models.model import CatBoostBinaryClassifier
from src.models.utils import save_model_features, get_model_features, loss_curves
from src.analysis.metrics import BinaryModelEvaluator
from src.data.utils import sample_and_split, extract_academic_year_from_path, load_config, get_timestamp, get_config_files


def run_pipeline(
          mode: str, config_path: str=None, exp_dir: str=None, inference_data_path: str=None
) -> tuple[dict, str]:
     """
     Executes the full ML pipeline for training or inference.
     :param mode: Either 'train' or 'infer'.
     :param config_path: Config object for 'train' mode.
     :param exp_dir: Directory of a trained experiment for 'infer' mode.
     :param inference_data_path: Path to inference input file (used in 'infer' mode).     
     Return: A tuple of (metrics dictionary, experiment directory).
     """
     
     # --- Validate Inputs ---
     if mode == 'train':
          if not config_path:
               raise ValueError("For 'train' mode, 'config_path' must be provided.")
          config = load_config(config_path)
          exp_dir = f"{config.exp.root_exps}/{config.exp.title}_{get_timestamp()}"
          os.makedirs(exp_dir, exist_ok=True)
          file_path = config.data.file_path
          wandb.init(
               project=config.exp.project, config=config, name=Path.stem(exp_dir), config_exclude_keys=['exp']
          )
          shutil.copyfile(config_path, os.path.join(exp_dir, "config.json"))

          # Initialize empty lists for features and metrics 
          cat_features, num_features, summary_metrics = [], [], {}

     elif mode == 'infer':
          if not all([exp_dir, inference_data_path]):
               raise ValueError("For 'infer' mode, both 'exp_dir' and 'inference_data_path' must be provided.")
          config = load_config(os.path.join(exp_dir, "config.json"))
          file_path = inference_data_path

          # Load model features and summary metrics from the experiment directory
          cat_features, num_features = get_model_features(dir=exp_dir)
          with open(os.path.join(exp_dir, "summary_metrics.json"), "r") as f:
               summary_metrics = json.load(f)           
                   
     else:
          raise ValueError(f"Invalid mode '{mode}'. Choose 'train' or 'infer'.")

     # --- Load Data ---
     df = pd.read_pickle(file_path)

     # --- Preprocessing ---
     preprocessor = DataPreprocessor()
     df, column_groups = preprocessor.preprocess(
          df=df, 
          column_filters=config.data.column_filters, 
          index=config.data.index, 
          label=config.data.label
     )

     # --- Splitting or Grouping ---
     datasets = {}
     if mode == 'train':
          datasets["train"], datasets["val"] = sample_and_split(
               df=df, 
               label=config.data.label, 
               sampling_prevalence=config.data.sample.p, 
               sample_seed=config.data.sample.seed,
               train_size=config.data.split.train_size, 
               split_seed=config.data.split.random_state, 
               shuffle=config.data.split.shuffle,
          )
     else:
          datasets[Path(file_path).stem] = df

     # --- Feature Engineering ---
     feature_engineer = EngineerAttendanceFeatures(
          holidays_calendar_path=config.data.holidays_calendar_path, index=config.data.index, label=config.data.label
     )
     feature_engineer.configure_features(**config.data.engineer_features)

     for split_name, split_df in datasets.items():
          datasets[split_name], cat_features_gen, num_features_gen = feature_engineer.generate_features(
               df=split_df, 
               acad_year=extract_academic_year_from_path(file_path), 
               column_groups=column_groups, 
               drop_columns_or_groups=config.data.drop_columns_or_groups, 
          )
          if not cat_features and not num_features:
               cat_features, num_features = cat_features_gen, num_features_gen

     # --- Model Initialization ---
     model = CatBoostBinaryClassifier(exp_dir=exp_dir, cat_features=cat_features, config=config)

     # --- Model Training (For 'train' mode) ---
     if mode == 'train':
          print("===> Training model...")
          save_model_features(exp_dir=exp_dir, cat_features=cat_features, num_features=num_features)

          best_params, val_thresh_f1, val_thresh_lift = model.fit(
               x_train=datasets["train"][cat_features + num_features], y_train=datasets["train"][config.data.label],
               x_val=datasets["val"][cat_features + num_features], y_val=datasets["val"][config.data.label]
          )

          loss_curves(train_dir=exp_dir)
          summary_metrics.update({
               "best_params": best_params, "shape_training_data": datasets["train"].shape,
               "val_threshold_max_f1": val_thresh_f1, "val_threshold_max_lift": val_thresh_lift,                        
               "categorical_features": cat_features, "n_categorical_features": len(cat_features), 
               "numerical_features": num_features, "n_numerical_features": len(num_features)
          })          

     # --- Prediction and Evaluation ---
     print("===> Generating predictions and evaluating...")

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
                    "val_max_f1": summary_metrics["val_threshold_max_f1"], 
                    "val_max_lift": summary_metrics["val_threshold_max_lift"]
               }
          )
          evaluator.plot_all()
          summary_metrics.update(evaluator.summary_metrics())

     if mode == 'train':
          with open(os.path.join(exp_dir, "summary_metrics.json"), "w") as f:
               json.dump(summary_metrics, f, indent=5)          
          wandb.log(summary_metrics)
          wandb.finish()

     return summary_metrics, exp_dir

def main():
     """Main CLI entry point for training or inference."""
     parser = argparse.ArgumentParser(description="Run ML pipelines.")
     parser.add_argument("--mode", type=str, required=True, choices=['train', 'infer'], help="Mode to run: 'train' or 'infer'.")
     parser.add_argument("--config_source", type=str, help="Path to config (file or dir). Required for 'train' mode.")
     parser.add_argument("--exp_dir", type=str, help="Experiment directory path. Required for 'infer' mode.")
     parser.add_argument("--inference_data_path", type=str, help="Inference input file. Required for 'infer' mode.")
     args = parser.parse_args()

     if args.mode == 'train':
          if not args.config_source:
               parser.error("--config_source is required for 'train' mode.")

          for config_path in get_config_files(args.config_source):
               print(f"\n===> Running training pipeline for config: {config_path}")
               summary_metrics, exp_dir = run_pipeline(mode='train', config_path=config_path)
               print("Training summary:", summary_metrics)
               print(f"✅ Training pipeline complete. All outputs saved to: {exp_dir}")

     elif args.mode == 'infer':
          if not args.exp_dir or not args.inference_data_path:
               parser.error("--exp_dir and --inference_data_path are required for 'infer' mode.")

          print(f"\n===> Running inference for {args.inference_data_path}")
          print(f"Using model and config from: {args.exp_dir}")
          summary_metrics, _ = run_pipeline(mode='infer', exp_dir=args.exp_dir, inference_data_path=args.inference_data_path)
          print("Inference summary:", summary_metrics)
          print(f"✅ Inference pipeline complete. All outputs saved to: {args.exp_dir}")

if __name__ == "__main__":
     main()