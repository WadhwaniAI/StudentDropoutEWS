import argparse
import os
from pathlib import Path 
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.explainability.predictor_group_explainer import PredictorGroupExplainer
from src.configs.utils import get_config_files
from src import constants


def parse_args():
     """
     Parses CLI arguments for training, inference, or explanation.
     :return: Argument parser instance with populated arguments.
     """
     parser = argparse.ArgumentParser(description="Run ML pipelines.")
     parser.add_argument(
          constants.CliArgs.MODE, 
          type=str, 
          required=True, 
          choices=[constants.CliModes.TRAIN, constants.CliModes.INFER, constants.CliModes.EXPLAIN], 
          help="Mode to run: 'train', 'infer', or 'explain'."
     )

     parser.add_argument(constants.CliArgs.CONFIG_SOURCE, type=str, help="Path to config file or directory (required in 'train' mode).") # train

     parser.add_argument(constants.CliArgs.EXP_DIR, type=str, help="Experiment directory with model and config (for 'infer' and 'explain').") # infer, explain
     parser.add_argument(constants.CliArgs.INFERENCE_DATA_PATH, type=str, help="Dataset to run inference on (for 'infer').") # infer

     parser.add_argument(constants.CliArgs.DF_PATH, type=str, help="Dataset to run SHAP explanations on (for 'explain').") # explain
     parser.add_argument(constants.CliArgs.PREDICTOR_GROUPS, type=str, help="Path to predictor group JSON file (for 'explain').") # explain
     parser.add_argument(constants.CliArgs.TARGET_RECALL, type=float, help="Recall target used to compute threshold (optional, for 'explain').") # explain
     parser.add_argument(constants.CliArgs.THRESHOLD, type=float, help="Custom threshold override for classification (optional, for 'explain').") # explain

     return parser


def training(config_source: str):
     """
     Runs the training pipeline on the provided config file(s).
     :param config_source: Path to a single config file or a directory containing multiple config files.
     """
     for config_path in get_config_files(config_source):
          print(f"\n===> Running training pipeline for config: {config_path}")
          pipeline = TrainingPipeline(config_path=config_path)
          summary_metrics, exp_dir = pipeline.run()
          print("Training summary:", summary_metrics)
          print(f"✅ Training complete. Outputs saved to: {exp_dir}")


def inference(exp_dir: str, inference_data_path: str):
     """
     Runs inference using a trained model.
     :param exp_dir: Path to the experiment directory containing the trained model and config.
     :param inference_data_path: Path to the input dataset for prediction.
     """
     print(f"\n===> Running inference for: {inference_data_path}")
     print(f"Using model from: {exp_dir}")
     pipeline = InferencePipeline(exp_dir=exp_dir, inference_data_path=inference_data_path)
     summary_metrics, _ = pipeline.run()
     print("Inference summary:", summary_metrics)
     print(f"✅ Inference complete. Outputs saved to: {exp_dir}")


def explain(exp_dir: str, df_path: str, predictor_groups: str, threshold: float=None, target_recall: float=None):
     """
     Runs SHAP explanation with predictor group aggregation.
     :param exp_dir: Path to experiment directory with trained model and config.
     :param df_path: Path to the dataset to explain (must be a pickled dataframe).
     :param predictor_groups: JSON file (or dict path) defining predictor groupings.
     :param threshold: Optional classification threshold override for predictions.
     :param target_recall: Optional recall target used to compute threshold.
     """
     print(f"\n===> Running SHAP explanation...")
     print(f"Using model from: {exp_dir}")
     print(f"Explaining  {df_path}")
     explainer = PredictorGroupExplainer(
          exp_dir=exp_dir,
          df_path=df_path,
          predictor_groups=predictor_groups,
          threshold=threshold,
          target_recall=target_recall
     )
     df_explained = explainer.run()
     out_path = os.path.join(exp_dir, f"{Path(df_path).stem}{constants.ModelArtifacts.EXPLAINED_OUTPUT_SUFFIX}{constants.FileExtensions.PICKLE}")
     df_explained.to_pickle(out_path)
     print(f"✅ Explanation complete. Output saved to: {out_path}")


def main():
     """Main CLI entry point. Dispatches to train, infer, or explain mode based on arguments."""
     parser = parse_args()
     args = parser.parse_args()

     if args.mode == constants.CliModes.TRAIN:
          if not args.config_source:
               parser.error(f"{constants.CliArgs.CONFIG_SOURCE} is required for '{constants.CliModes.TRAIN}'.")
          training(args.config_source)

     elif args.mode == constants.CliModes.INFER:
          if not args.exp_dir or not args.inference_data_path:
               parser.error(f"{constants.CliArgs.EXP_DIR} and {constants.CliArgs.INFERENCE_DATA_PATH} are required for '{constants.CliModes.INFER}'.")
          inference(args.exp_dir, args.inference_data_path)

     elif args.mode == constants.CliModes.EXPLAIN:
          if not args.exp_dir or not args.df_path or not args.predictor_groups:
               parser.error(f"{constants.CliArgs.EXP_DIR}, {constants.CliArgs.DF_PATH}, and {constants.CliArgs.PREDICTOR_GROUPS} are required for '{constants.CliModes.EXPLAIN}'.")
          explain(
               exp_dir=args.exp_dir,
               df_path=args.df_path,
               predictor_groups=args.predictor_groups,
               threshold=args.threshold,
               target_recall=args.target_recall,
          )


if __name__ == "__main__":
     main()