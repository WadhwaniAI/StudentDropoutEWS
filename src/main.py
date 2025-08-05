import argparse
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.configs.utils import get_config_files


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
               pipeline = TrainingPipeline(config_path=config_path)
               summary_metrics, exp_dir = pipeline.run()
               print("Training summary:", summary_metrics)
               print(f"✅ Training pipeline complete. All outputs saved to: {exp_dir}")

     elif args.mode == 'infer':
          if not args.exp_dir or not args.inference_data_path:
               parser.error("--exp_dir and --inference_data_path are required for 'infer' mode.")

          print(f"\n===> Running inference for {args.inference_data_path}")
          print(f"Using model and config from: {args.exp_dir}")
          pipeline = InferencePipeline(exp_dir=args.exp_dir, inference_data_path=args.inference_data_path)
          summary_metrics, exp_dir = pipeline.run()
          print("Inference summary:", summary_metrics)
          print(f"✅ Inference pipeline complete. All outputs saved to: {args.exp_dir}")

if __name__ == "__main__":
     main()