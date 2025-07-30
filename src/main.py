import argparse
import os
import shutil
import wandb
from src.training import training_pipeline
from src.inference import inference_pipeline
from src.utils import get_timestamp, get_config_files, load_config


def main():
     parser = argparse.ArgumentParser(description="Run training (and optionally inference) pipelines.")
     parser.add_argument("--config_source", type=str, required=True, help="Path to config JSON file or directory of JSON configs.")
     args = parser.parse_args()

     for config_path in get_config_files(args.config_source):
          print(f"\n===> Running training pipeline for config: {config_path}")
     
          config = load_config(config_path)
          exp_dir = f"{config.exp.root_exps}/{config.exp.title}_{get_timestamp()}"
          wandb.init(
               project=config.exp.project, 
               config=config, 
               name=os.path.basename(exp_dir), 
               config_exclude_keys=['exp']
          )

          # Run training pipeline and return metrics on train and val
          training_summary, wandb_metrics = training_pipeline(config, exp_dir)
          print("Training summary:", training_summary)

          # Save config file in experiment directory
          shutil.copyfile(config_path, os.path.join(exp_dir, "config.json"))

          # Run inference pipeline (optional)
          inference_path = getattr(config.data, "inference_data_path", None)
          if inference_path and os.path.exists(inference_path):
               print(f"===> Running inference for {inference_path}")
               inference_summary = inference_pipeline(
                    exp_dir=exp_dir, 
                    inference_data_path=inference_path, 
                    manual_thresholds=training_summary.get("val_optimal_thresholds", None)
               )
               wandb_metrics.update(inference_summary)
               print("Inference summary:", inference_summary)

          wandb.log(wandb_metrics)
          wandb.finish()

if __name__ == "__main__":
     main()