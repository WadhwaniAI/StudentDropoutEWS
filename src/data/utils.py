import json
import math
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Union, List, Dict, Tuple, Optional, Mapping
from collections import defaultdict
from collections.abc import Mapping
from src import constants


def build_attendance_replacement_map(
     source: Union[str, Path, Dict[str, List[str]]]
) -> Dict[str, str]:
     """
     Builds a replacement map from a semantic attendance map.

     The source can be a file path or a pre-loaded dictionary. It transforms
     a map like `{"present": ["1"]}` into a replacement map like `{"1": "p"}`
     using the characters defined in `constants.Attendance.Status`.

     :param source: Path to the semantic JSON map or the dictionary itself.
     :return: A dictionary mapping raw values to replacement characters.
     """
     if isinstance(source, (str, Path)):
          with open(source, "r") as f:
               semantic_map = json.load(f)
     elif isinstance(source, dict):
          semantic_map = source
     else:
          raise TypeError(f"Source must be a path or a dictionary, not {type(source).__name__}")

     status_map = {
          k.lower(): v
          for k, v in vars(constants.Attendance.Status).items()
          if not k.startswith("__")
     }

     allowed_keys = set(status_map.keys())
     if set(semantic_map.keys()) != allowed_keys:
          raise ValueError(
               "Invalid keys in attendance replacement map. "
               f"Expected exactly {sorted(list(allowed_keys))}, but found {sorted(list(semantic_map.keys()))}."
          )

     return {
          value: status_map[status_name]
          for status_name, values in semantic_map.items()
          for value in values
     }


def determine_columns_to_drop(
          df: pd.DataFrame,
          drop_columns_or_groups: Optional[List[str]]=None,
          column_groups: Optional[Dict[str, List[str]]]=None
) -> List[str]:
     """
     Determines the columns to drop based on column group names or individual column names.
     :param df: DataFrame to validate individual column existence
     :param drop_columns_or_groups: List of group names or individual column names to drop
     :param column_groups: Mapping from group names to list of column names
     :return: Set of columns_to_drop, sorted alphabetically
     """
     drop_columns_or_groups = drop_columns_or_groups or []
     column_groups = column_groups or {}
     columns_to_drop = []

     for key in drop_columns_or_groups:
          if key in column_groups:
               columns_to_drop.extend(column_groups[key])
          elif isinstance(key, str) and key in df.columns:
               columns_to_drop.append(key)

     return sorted(set(columns_to_drop))


def split_features_by_dtype(
          df: pd.DataFrame, index: str="aadhaaruid", label: Optional[str]="target", 
) -> Tuple[List[str], List[str]]:
     """
     Identify columns as Categorical with dtype 'object' and Numerical with dtype np.float64 only.     
     Excludes index and label columns from both.
     :param df: Input DataFrame
     :param index: Name of index column to exclude
     :param label: Optional. Name of label/target column to exclude
     Returns: Tuple of (categorical_features, numerical_features)
     """
     exclude = {label, index}
     categorical_features = [col for col in df.select_dtypes(include=constants.DtypeCastMap.STR).columns if col not in exclude]
     numerical_features = [col for col in df.select_dtypes(include=constants.DtypeCastMap.FLOAT).columns if col not in exclude]

     return categorical_features, numerical_features


def holidays_academic_year_wise(holidays_path: str) -> dict:
     """
     Generate {academic_year: [month_day]} mapping from detailed holiday metadata.
     :param holidays_path: Path to JSON file containing holiday data
     """
     with open(holidays_path, "r") as f:
          holidays = json.load(f)

     months = list(range(6, 13)) + list(range(1, 6))
     return {
          str(ay): sorted(
               {
                    f"{m}_{d}" 
                    for m in months if str(m) in month_data
                    for days in month_data[str(m)].values()
                    for d in days
               }
          )
          for ay, month_data in holidays.items()
     }


def sample_and_split(
          df, label, sampling_prevalence, sample_seed, train_size, split_seed, shuffle
) -> dict:
     """
     Samples the dataset based on given prevalence, then splits into train and validation sets.
     :param df (pd.DataFrame): DataFrame containing features and label
     :param label (str): Name of the label column to sample on
     :param sampling_prevalence (float or str): Desired prevalence for the label ('actual' for current prevalence)
     :param sample_seed (int): Seed for random sampling
     :param train_size (float): Proportion of the dataset to include in the train split
     :param split_seed (int): Seed for train-test split
     :param shuffle (bool): Whether to shuffle the data before splitting     
     Returns: Tuple of DataFrames (df_train, df_val)
     """
     actual_p = df[label].mean()
     p = actual_p if sampling_prevalence == "actual" else float(sampling_prevalence)

     num_pos = df[label].sum()
     num_neg = len(df) - num_pos

     if p < actual_p:
          sample_sizes = {
               0: int(math.ceil(num_neg)),
               1: int(math.ceil(num_neg * p / (1 - p)))
          }
     else:
          sample_sizes = {
               0: int(math.ceil(num_pos * (1 - p) / p)),
               1: int(math.ceil(num_pos))
          }

     sampled_indices = []
     for label_val in [0, 1]:
          label_indices = df.index[df[label] == label_val].tolist()
          sampled_indices.extend(label_indices[:sample_sizes[label_val]])

     sampled_df = df.loc[sampled_indices].sample(frac=1, random_state=sample_seed)

     X, y = sampled_df.drop(columns=[label]), sampled_df[label]

     X_train, X_val, y_train, y_val = train_test_split(
          X, y, train_size=train_size, stratify=y,
          random_state=split_seed, shuffle=shuffle
     )

     return (
          pd.concat([X_train, y_train], axis=1).reset_index(drop=True),
          pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
     )


def extract_academic_year_from_path(path: str) -> str:
     """Extract academic year (e.g., '2223') from path like 'ay2223_grade3.pkl'."""
     filename = Path(path).name
     match = re.search(r"ay(\d{4})_grade\d+.*\.pkl", filename)
     if not match:
          raise ValueError(f"Could not extract academic year from path: {path}")
     return match.group(1)


def generate_column_groups_from_schema(schema_input: Union[str, Path, Mapping]):
     """
     Generates column groups by reading group info from a dataset schema..
     :param schema_input: Path to the dataset_schema.json (str or pathlib.Path) or a schema dictionary/mapping (dict, Munch, etc.).
     :return: A dictionary mapping each group to its list of columns.
     """
     schema = {}
     if isinstance(schema_input, (str, Path)):
          with open(schema_input, 'r') as f:
               schema = json.load(f)
     elif isinstance(schema_input, Mapping):
          schema = schema_input
     else:
          raise TypeError("Input must be a file path (str or pathlib.Path) or a mapping (e.g., dict).")

     column_groups = defaultdict(list)
     for column, attributes in schema.items():
          if len(attributes) > 2 and attributes[2]:
               column_groups[attributes[2]].append(column)

     return dict(column_groups)