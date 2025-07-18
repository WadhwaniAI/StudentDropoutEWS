import json
import numpy as np
import os
import pandas as pd
import pickle
import re
import shap
from utils import get_config_files, load_config, load_data_file
from catboost_binary_classifier import CatBoostBinaryClassifier



def generate_and_save_shap_values(
          exp_dir,
          df,
          ds_name,
          cohort_name,
          config_filename_index=1,
          output_suffix="shap_values",
):
     """
     Compute and save SHAP values for a single cohort dataframe.

     The function:
          - Loads model configuration and trained CatBoost model
          - Computes SHAP values for the provided dataframe
          - Appends SHAP values as new columns to the dataframe
          - Saves the resulting dataframe to disk

     Parameters
     ----------
     exp_dir : str
          Path to the experiment directory containing model and configs.

     df : pandas.DataFrame
          Dataframe on which to compute SHAP values. Must include the
          necessary feature columns used by the model.

     ds_name : str
          Base name of the dataset (without extension).
          E.g. "prod[ay, fullsat]"

     cohort_name : str
          Label to include in the output filename to distinguish different cohorts
          (e.g. "public" or "private" or any custom group name).

     config_filename_index : int, optional
          Index of the config JSON file in the list returned by
          `get_config_files`. Defaults to 1.

     output_suffix : str, optional
          Suffix used in output filenames for saved dataframes with SHAP values.
          Defaults to "shap_values".

     Returns
     -------
     None
          The function saves the dataframe but does not return objects.
     """

     # Load list of numerical features
     num_features_path = os.path.join(exp_dir, "num_features.pkl")
     with open(num_features_path, "rb") as f:
          num_features = pickle.load(f)

     # Load list of categorical features
     cat_features_path = os.path.join(exp_dir, "cat_features.pkl")
     with open(cat_features_path, "rb") as f:
          cat_features = pickle.load(f)

     # Combine all feature names used for predictions
     features = cat_features + num_features

     # Load model configuration
     config_files = get_config_files(config_dir=exp_dir, ext="json")
     config_path = config_files[config_filename_index]
     config = load_config(config_path)

     # Initialize CatBoost classifier
     clf = CatBoostBinaryClassifier(
          exp_dir=exp_dir,
          cat_features=cat_features,
          config=config,
     )

     # Load trained CatBoost model
     model_path = os.path.join(exp_dir, "model.cbm")
     clf.model.load_model(model_path)

     # Ensure numeric features are float64
     df[num_features] = df[num_features].astype(np.float64)

     # Create SHAP explainer
     explainer = shap.TreeExplainer(clf.model)

     # Compute SHAP values
     shap_values = explainer(df[features])

     # Extract feature names from SHAP values object
     feature_names = shap_values.feature_names

     # Generate new column names for SHAP values
     shap_columns = [f"shap[{name}]" for name in feature_names]

     # Create DataFrame for SHAP values
     shap_df = pd.DataFrame(
          shap_values.values,
          columns=shap_columns,
          index=df.index,
     )

     # Concatenate original dataframe with SHAP values
     df_out = pd.concat([df, shap_df], axis=1)

     # Construct output filename
     output_filename = f"{ds_name}[{cohort_name}][{output_suffix}].pkl"
     output_path = os.path.join(exp_dir, output_filename)

     # Save the updated dataframe
     df_out.to_pickle(output_path)

     print(
          f"Saved SHAP values for cohort '{cohort_name}' "
          f"to: {output_path} | Rows: {len(df_out)}"
     )



def add_shap_group_contributions(df, predictor_groups):
     """
     Add SHAP group-level contributions and top-driving features for both labels (0 and 1).
     
     Parameters:
          df (pd.DataFrame): DataFrame with SHAP values (e.g., 'shap_feature1', ...)
          predictor_groups (dict): {group_name: [feature1, feature2, ...]}
          shap_prefix (str): Prefix for SHAP value columns. Default is 'shap_'
     
     Returns:
          pd.DataFrame: Original df with 48 additional columns.
     """

     def compute_row_contributions(row):
          result = {}
          contribs_by_label = {0: {}, 1: {}}
          tops_by_label = {0: {}, 1: {}}

          for label in [0, 1]:
               for group, features in predictor_groups.items():
                    # Filter to features that exist in the DataFrame
                    valid_features = [f for f in features if f"shap[{f}]" in df.columns]
                    shap_scores = [row[f"shap[{f}]"] for f in valid_features]

                    if label == 1:
                         filtered = [(f, s) for f, s in zip(valid_features, shap_scores) if s > 0]
                         contrib = sum(s for _, s in filtered)
                    else:
                         filtered = [(f, abs(s)) for f, s in zip(valid_features, shap_scores) if s <= 0]
                         contrib = sum(s for _, s in filtered)

                    top_feat = max(filtered, key=lambda x: x[1])[0] if filtered else None

                    result[f"{group}_label{label}_contrib"] = contrib
                    result[f"{group}_label{label}_top_feat"] = top_feat

                    contribs_by_label[label][group] = contrib
                    tops_by_label[label][group] = top_feat

          # Add sorted groups and their top drivers
          for label in [0, 1]:
               label_str = 'dropout' if label == 1 else 'notdropout'
               sorted_groups = sorted(contribs_by_label[label].items(), key=lambda x: x[1], reverse=True)
               for i, (group, _) in enumerate(sorted_groups, 1):
                    result[f"predictor_group_{i}_for_{label_str}"] = group
                    result[f"predictor_group_{i}_for_{label_str}_top_driver"] = tops_by_label[label][group]

          return pd.Series(result)

     # Apply row-wise logic
     df_contribs = df.apply(compute_row_contributions, axis=1)
     return pd.concat([df, df_contribs], axis=1)



def select_predictor_groups_by_class(df, prediction_col):
     """
     Selects the appropriate 12 predictor group and top driver columns based on the prediction class.
     
     Parameters:
          df (pd.DataFrame): DataFrame with full SHAP-related columns.
          prediction_col (str): Name of the column containing binary prediction class (0 or 1).
     
     Returns:
          pd.DataFrame: Original df with 12 additional columns: selected_predictor_group_* and top_driver.
     """
     def pick_relevant_columns(row):
          label_str = row[prediction_col]
          result = {}
          for i in range(1, 7):
               group_col = f"predictor_group_{i}_for_{label_str}"
               driver_col = f"{group_col}_top_driver"
               result[f"predictor_group_{i}"] = row.get(group_col)
               result[f"predictor_group_{i}_top_driver"] = row.get(driver_col)
          return pd.Series(result)

     df_selected = df.apply(pick_relevant_columns, axis=1)
     return pd.concat([df, df_selected], axis=1)



def process_and_save_shap_group_contributions(
          exp_dirs,
          ds_names,
          predictor_groups,
          output_suffix="predictors"
):
     """
     For each experiment directory and dataset name, load the SHAP-enriched
     DataFrame, compute SHAP group contributions using predictor_groups,
     and save the resulting DataFrame to disk.

     Parameters
     ----------
     exp_dirs : list of str
          List of experiment directory paths.

     ds_names : list of str
          List of dataset filenames (without .pkl) that include SHAP values.

     predictor_groups : dict
          Dictionary mapping group names to lists of predictor (feature) names.

     output_suffix : str, optional
          Suffix to append in the output filenames. Defaults to "predictors".

     Returns
     -------
     None
          Saves the updated DataFrames to disk.
     """

     for exp_dir in exp_dirs:
          print(f"Processing: {exp_dir}")

          # Extract grade number from directory name
          grade_match = re.findall(r"grade \d", exp_dir)
          if not grade_match:
               raise ValueError(f"Could not extract grade from: {exp_dir}")
          grade = int(grade_match[0][-1])

          # Load feature lists
          with open(os.path.join(exp_dir, "num_features.pkl"), "rb") as f:
               num_features = pickle.load(f)
          with open(os.path.join(exp_dir, "cat_features.pkl"), "rb") as f:
               cat_features = pickle.load(f)

          for ds_name in ds_names:
               # Load DataFrame with SHAP values
               df_path = os.path.join(exp_dir, f"{ds_name}.pkl")
               df = load_data_file(df_path)

               # Identify SHAP value columns
               shap_columns = [col for col in df.columns if "shap[" in col]

               # Ensure correct data types
               df[num_features] = df[num_features].astype("float64")
               df[shap_columns] = df[shap_columns].astype("float64")

               # Add SHAP group contributions
               df_final = add_shap_group_contributions(df, predictor_groups)

               # Save updated DataFrame
               output_path = os.path.join(exp_dir, f"{ds_name}[{output_suffix}].pkl")
               df_final.to_pickle(output_path)

               print(f"✅ grade {grade} | ds: {ds_name} | shape: {df_final.shape}")



def apply_threshold_and_save_selected_predictors(
          exp_dirs,
          target_recall,
          target_ds_name,
          ds_names,
          output_suffix="chosen_predictors"
):
     """
     For each experiment directory and dataset name:
          - Load model thresholds
          - Apply probability threshold to create predictions
          - Map predictions to class labels
          - Select predictor groups for each class
          - Save resulting dataframe

     Parameters
     ----------
     exp_dirs : list of str
          List of experiment directory paths.

     target_recall : float
          Desired recall value used to look up the threshold.

     target_ds_name : str
          Dataset name string used as key for the thresholds dictionary.

     ds_names : list of str
          Dataset names (without .pkl) to process.

     output_suffix : str, optional
          Suffix to append in saved filenames. Defaults to "chosen_predictors".

     Returns
     -------
     None
          Saves the updated DataFrames to disk.
     """

     for exp_dir in exp_dirs:
          print(f"Processing: {exp_dir}")

          # Extract grade number
          grade_match = re.findall(r"grade \d", exp_dir)
          if not grade_match:
               raise ValueError(f"Could not extract grade from path: {exp_dir}")
          grade = int(grade_match[0][-1])

          # Load thresholds from JSON
          thresholds_path = os.path.join(exp_dir, "thresholds.json")
          with open(thresholds_path, "r") as file:
               thresholds = json.load(file)

          # Determine threshold value
          key = f"{target_ds_name}, recall={target_recall}"
          if key not in thresholds:
               raise ValueError(f"Threshold not found for key: {key}")
          t = thresholds[key]

          for ds_name in ds_names:
               # Load DataFrame
               df_path = os.path.join(exp_dir, f"{ds_name}.pkl")
               df = load_data_file(df_path)

               # Ensure correct dtype
               df["preds_proba_1"] = df["preds_proba_1"].astype(np.float64)

               # Compute predictions
               df["prediction"] = (df["preds_proba_1"] >= t).astype(int)
               num_dropouts = df["prediction"].sum()

               # Map numeric predictions to class names
               df["prediction"] = df["prediction"].map({0: "notdropout", 1: "dropout"})

               # Select chosen predictor groups
               df_final = select_predictor_groups_by_class(df, prediction_col="prediction")

               # Build output path
               output_path = os.path.join(exp_dir, f"{ds_name}[{output_suffix}].pkl")

               # Save final DataFrame
               df_final.to_pickle(output_path)

               print(
                    f"✅ grade {grade} | ds: {ds_name} | "
                    f"shape: {df_final.shape} | # dropouts: {num_dropouts}"
               )



def run_shap_pipeline(
          exp_dirs,
          ds_name,
          df_cohorts_dict,
          predictor_groups,
          target_recall,
          target_ds_name,
          config_filename_index=1,
          shap_output_suffix="shap_values",
          group_output_suffix="predictors",
          final_output_suffix="chosen_predictors"
):
     """
     Runs the full SHAP analysis pipeline across cohorts:
          - Compute SHAP values and save
          - Append group-level SHAP contributions
          - Apply threshold and save top driving predictors

     Parameters
     ----------
     exp_dirs : list of str
          List of experiment directories (one per cohort/model setup).

     ds_name : str
          Base name of the dataset, used for filenames (e.g. "prod[ay, fullsat]").

     df_cohorts_dict : dict
          Dictionary mapping cohort names to dataframes (e.g. {"public": df1, "private": df2}).

     predictor_groups : dict
          Dictionary mapping group names to feature lists.

     target_recall : float
          Recall value used to determine probability threshold.

     target_ds_name : str
          Name used as key to find the threshold in thresholds.json.

     config_filename_index : int, optional
          Index for selecting model config JSON. Default is 1.

     shap_output_suffix : str, optional
          Suffix for saved SHAP values file. Default is "shap_values".

     group_output_suffix : str, optional
          Suffix for group contributions file. Default is "predictors".

     final_output_suffix : str, optional
          Suffix for final predictors file. Default is "chosen_predictors".

     Returns
     -------
     None
          Saves files at each stage within the respective experiment directories.
     """

     # Step 1: Generate and save SHAP values for each cohort
     for exp_dir in exp_dirs:
          for cohort_name, df in df_cohorts_dict.items():
               print(f"\n🔹 Generating SHAP values for cohort: {cohort_name}")
               generate_and_save_shap_values(
                    exp_dir=exp_dir,
                    df=df.copy(),
                    ds_name=ds_name,
                    cohort_name=cohort_name,
                    config_filename_index=config_filename_index,
                    output_suffix=shap_output_suffix,
               )

     # Step 2: Append SHAP group contributions
     process_and_save_shap_group_contributions(
          exp_dirs=exp_dirs,
          ds_names=[f"{ds_name}[{cohort}][{shap_output_suffix}]" for cohort in df_cohorts_dict],
          predictor_groups=predictor_groups,
          output_suffix=group_output_suffix,
     )

     # Step 3: Apply thresholds and select top predictor groups
     apply_threshold_and_save_selected_predictors(
          exp_dirs=exp_dirs,
          target_recall=target_recall,
          target_ds_name=target_ds_name,
          ds_names=[f"{ds_name}[{cohort}][{shap_output_suffix}][{group_output_suffix}]" for cohort in df_cohorts_dict],
          output_suffix=final_output_suffix,
     )

     print("\n✅ SHAP pipeline completed.")
