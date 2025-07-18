import json
import os
import numpy as np
import pandas as pd
import re

from sklearn.metrics import precision_recall_curve

from catboost_binary_classifier import CatBoostBinaryClassifier
from dataset import Dataset
from transforms import *
from utils import load_data_file, custom_json_formatter


def predict_on_raw_data(
          file_path: str,
          exp_dir: str,
          config: dict
) -> pd.DataFrame:

     '''
     Description:
          Prediction function that takes in a raw dataframe, transforms it using config and saved transformations if any
          in the experimental directory and returns dataframe with confidence scores and predictions appended as columns. 
     Args:
          file_path: path to the input csv file.
          exp_dir: path to the training directory.
          config: dict of the config file
     Returns:
          Dataframe with confidence scores and predictions as columns.
     '''

     # instance of dataset class
     dataset = Dataset(
          exp_dir=exp_dir,
          config=config,
          file_path=file_path
     )

     # transform numerical features
     transforms = Transforms(
          dataframe=dataset.df,
          num_features=dataset.num_features,
          label=dataset.label,
          feature_groups=dataset.feature_groups,
          exp_dir=exp_dir,
          config_transforms=config.data.post_split_transforms
     )
     dataset.df = transforms.apply()

     # update features post transforms
     dataset.get_features()

     # now predict on transformed dataframe
     dataset.df = predict_on_transformed_dataframe(
          df=dataset.df, 
          features=dataset.features,
          cat_features=dataset.cat_features,
          config=config, 
          exp_dir=exp_dir
     )

     # return dataset object
     return dataset
     
     

def predict_on_transformed_dataframe(
          df: pd.DataFrame,
          features: list,
          cat_features: list,
          config: dict,
          exp_dir: str
) -> pd.DataFrame:

     '''
     Description:
          Prediction function that takes in a transformed dataframe, list of categorical and numerical features, config and
          path to experimental directory and returns dataframe with confidence scores and predictions. 
     Args:
          df: transformed dataframe.
          features: list of all features.
          cat_features: list of all categorical features.
          exp_dir: path to the training (experimental) directory.
          config: dict of the config file.
     Returns:
          Dataframe with confidence scores and predictions as columns.
     '''
     
     # instantiate model using experimental directory
     cb_bin_classifier = CatBoostBinaryClassifier(
          exp_dir=exp_dir,
          cat_features=cat_features,
          config=config
     )

     # return dataframe with confidence scores
     return cb_bin_classifier.predict(
          x=df, 
          features=features
     )



def generate_thresholds_json(
          exp_dirs,
          ds_name,
          recalls,
          output_filename="thresholds.json"
):
     """
     Compute probability thresholds corresponding to given recall levels
     for each experiment directory and save them as a JSON file.

     Parameters
     ----------
     exp_dirs : list of str
          List of experiment directory paths.

     ds_name : str
          Dataset name (without .pkl extension) to load for predictions and targets.

     recalls : list of float
          Recall values for which to find thresholds.

     output_filename : str, optional
          Name of the JSON file to save thresholds. Defaults to "thresholds.json".

     Returns
     -------
     None
          Saves JSON files to disk.
     """

     for exp_dir in exp_dirs:
          print(f"Processing: {exp_dir}")

          # Extract grade
          grade_match = re.findall(r"grade \d", exp_dir)
          if not grade_match:
               raise ValueError(f"Could not extract grade from path: {exp_dir}")
          grade = int(grade_match[0][-1])

          # Load data
          df_path = os.path.join(exp_dir, f"{ds_name}.pkl")
          df = load_data_file(df_path)
          df["preds_proba_1"] = df["preds_proba_1"].astype(np.float64)
          df["target"] = df["target"].astype("int64")

          # Compute precision-recall curve
          ps, rs, ts = precision_recall_curve(
               y_true=df["target"],
               probas_pred=df["preds_proba_1"]
          )

          # Handle sklearn's quirk: length(ts) = len(rs) - 1
          if len(ts) < len(rs):
               rs = rs[:-1]

          # Build thresholds dict
          thresholds = {}
          for recl in recalls:
               idx = np.argmin(abs(recl - rs))
               thresholds[f"{ds_name}, recall={recl}"] = float(ts[idx])

          # Format for JSON
          custom_json = custom_json_formatter(thresholds)

          # Save to JSON
          output_path = os.path.join(exp_dir, output_filename)
          with open(output_path, "w") as f:
               f.write(custom_json)

          print(f"✅ Done: {exp_dir} | Saved: {output_filename}")
