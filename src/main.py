import argparse
import os
from pathlib import Path 
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.explainability.predictor_group_explainer import PredictorGroupExplainer
from src.configs.utils import get_config_files


def parse_args():
     """
     Parses CLI arguments for training, inference, or explanation.
     :return: Argument parser instance with populated arguments.
     """
     parser = argparse.ArgumentParser(description="Run ML pipelines.")
     parser.add_argument("--mode", type=str, required=True, choices=["train", "infer", "explain"], help="Mode to run: 'train', 'infer', or 'explain'.")

     parser.add_argument("--config_source", type=str, help="Path to config file or directory (required in 'train' mode).") # train

     parser.add_argument("--exp_dir", type=str, help="Experiment directory with model and config (for 'infer' and 'explain').") # infer, explain
     parser.add_argument("--inference_data_path", type=str, help="Dataset to run inference on (for 'infer').") # infer

     parser.add_argument("--df_path", type=str, help="Dataset to run SHAP explanations on (for 'explain').") # explain
     parser.add_argument("--predictor_groups", type=str, help="Path to predictor group JSON file (for 'explain').") # explain
     parser.add_argument("--target_recall", type=float, help="Recall target used to compute threshold (optional, for 'explain').") # explain
     parser.add_argument("--threshold", type=float, help="Custom threshold override for classification (optional, for 'explain').") # explain

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
     out_path = os.path.join(exp_dir, f"{Path(df_path).stem}_explained_output.pkl")
     df_explained.to_pickle(out_path)
     print(f"✅ Explanation complete. Output saved to: {out_path}")


def main():
     """Main CLI entry point. Dispatches to train, infer, or explain mode based on arguments."""
     parser = parse_args()
     args = parser.parse_args()

     if args.mode == "train":
          if not args.config_source:
               parser.error("--config_source is required for 'train'.")
          training(args.config_source)

     elif args.mode == "infer":
          if not args.exp_dir or not args.inference_data_path:
               parser.error("--exp_dir and --inference_data_path are required for 'infer'.")
          inference(args.exp_dir, args.inference_data_path)

     elif args.mode == "explain":
          if not args.exp_dir or not args.df_path or not args.predictor_groups:
               parser.error("--exp_dir, --df_path, and --predictor_groups are required for 'explain'.")
          explain(
               exp_dir=args.exp_dir,
               df_path=args.df_path,
               predictor_groups=args.predictor_groups,
               threshold=args.threshold,
               target_recall=args.target_recall,
          )


if __name__ == "__main__":
     main()