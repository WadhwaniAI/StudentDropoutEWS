import json
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from src.utils import resolve_path
from .utils import generate_column_groups_from_schema


class DataPreprocessor:
     """Class to perform basic preprocessing on a pandas DataFrame based on schema and feature group metadata."""

     def __init__(
               self,
               schema_path: str="metadata/dataset_schema.json",
               attendance_replacement_map: Optional[dict]=None
     ):
          """
          :param schema_path: Path to the JSON schema file defining column data types.
          :param column_groups_path: Path to the JSON file defining column groups.
          :param attendance_replacement_map: Mapping for replacing attendance values.
          """
          self.schema_path = resolve_path(schema_path)
          self.schema = self._load_json(self.schema_path)
          self.column_groups = generate_column_groups_from_schema(self.schema_path)
          self.all_attendance_cols = self.column_groups["all_attendances"]
          self.attendance_replacement_map = attendance_replacement_map or {"nan":"m", "5":"m", "0":"m", "1":"p", "2":"a"}

     def _load_json(self, path: str) -> dict:
          """Loads and returns JSON content from a file."""
          with open(path, "r", encoding="utf-8") as f:
               return json.load(f)

     def preprocess(
               self, df: pd.DataFrame, column_filters: Optional[Dict[str, List[str]]]=None,
               index: str="aadhaaruid", label: str="target"
     ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
          """Lowercases column names, removes duplicates, casts dtypes, fixes exam attendance, and applies filters.
          :param df (pd.DataFrame): Input DataFrame to preprocess.
          :param index (str): Column name to use as index.
          :param label (str): Column name for the target variable.
          Returns: Tuple (Preprocessed DataFrame (pd.DataFrame), Filtered column_groups (dict))"""
          # Lowercasing column names
          df.columns = map(str.lower, df.columns)

          # Remove duplicates based on index
          df = df.drop_duplicates(subset=index)

          # Filter metadata based on DataFrame columns
          self.schema, self.column_groups = self._validate_and_filter_metadata(
               df, self.schema, self.column_groups, reserved_columns={index, label}
          )

          # Update attendance columns to those present in the dataframe
          self.all_attendance_cols = [col for col in self.all_attendance_cols if col in df.columns]

          # Cast columns to schema data types
          cast_map = {"str": str, "int": "int64", "float": np.float64}
          dtype_map = {dtype: [col for col in df.columns if self.schema[col][0] == dtype] for dtype in cast_map}
          converted_parts = [df[cols].astype(cast_map[dtype]) for dtype, cols in dtype_map.items() if cols]
          all_typed_cols = sum(dtype_map.values(), [])
          df = pd.concat([df.drop(columns=all_typed_cols)] + converted_parts, axis=1)

          # Replace values in attendance columns
          df.loc[:, self.all_attendance_cols] = df.loc[:, self.all_attendance_cols].replace(self.attendance_replacement_map)

          # Rectify attendance using exam scores
          for col_attnd in self.column_groups["exam_attnd_subwise"]:
               col_score = col_attnd.replace("attnd", "score")
               if col_score in df.columns:
                    df.loc[:, col_attnd] = df[col_score].apply(lambda x: "m" if pd.isnull(x) else "p" if x > 0 else "a")

          # Fill missing values with nan for categorical columns and 0 for numerical columns
          df.loc[:, dtype_map["str"]] = df.loc[:, dtype_map["str"]].fillna("nan")
          df.loc[:, dtype_map["float"]] = df.loc[:, dtype_map["float"]].fillna(0.0)

          self.df = df

          if column_filters:
               self._apply_column_filters(column_filters)

          return self.df, self.column_groups

     def _apply_column_filters(self, column_filters: dict) -> None:
          """Applies filtering based on column_filters with 'in' and/or 'notin' keys.
          :param column_filters: Dict with optional 'in' and 'notin' sub-dicts specifying filtering logic."""
          for mode, op in {'in': lambda x, y: x.isin(y), 'notin': lambda x, y: ~x.isin(y)}.items():
               for col, values in column_filters.get(mode, {}).items():
                    if col in self.df.columns:
                         self.df = self.df[op(self.df[col], values)]
          self.df.reset_index(drop=True, inplace=True)

     def _validate_and_filter_metadata(
               self, df: pd.DataFrame, schema: dict, column_groups: dict, reserved_columns: Set[str]=None
     ) -> Tuple[dict, dict]:
          """Validate and filter schema and feature group metadata based on DataFrame columns.
          :param df: Input DataFrame
          :param schema: Original schema dictionary
          :param column_groups: Original column_groups dictionary
          :param reserved_columns: Columns to exclude from validation
          Returns: Tuple of (Filtered schema dictionary, Filtered column_groups dictionary)
          Raises: ValueError if required columns are missing"""
          reserved_columns = reserved_columns or set()
          df_columns = set(df.columns)

          # Validate schema completeness
          missing_in_schema = df_columns - set(schema.keys())
          if missing_in_schema:
               raise ValueError(f"Columns missing in schema: {sorted(missing_in_schema)}")

          # Validate column_groups coverage (excluding reserved columns)
          non_reserved_cols = df_columns - reserved_columns
          feature_group_cols = {col for group in column_groups.values() for col in group}
          missing_in_groups = non_reserved_cols - feature_group_cols
          if missing_in_groups:
               raise ValueError(f"Columns missing in column_groups: {sorted(missing_in_groups)}")

          print("✅ All DataFrame columns found in schema and column_groups.")

          # Filter schema to used columns
          schema_filtered = {col: val for col, val in schema.items() if col in df_columns}

          # Filter column_groups to used columns
          column_groups_filtered = {group: [f for f in feats if f in non_reserved_cols] for group, feats in column_groups.items()}

          return schema_filtered, column_groups_filtered