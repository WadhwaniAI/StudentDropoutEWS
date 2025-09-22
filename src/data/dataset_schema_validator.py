import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union
from src import constants


class DatasetSchemaValidator:
     """Validates a dataset schema against a set of predefined rules."""

     def __init__(self, schema: Union[str, Path, Dict[str, Any]]):
          """
          Initializes the validator with the schema.
          :param schema: A dictionary, a string path, or a Path object representing the schema.
          """
          if isinstance(schema, (str, Path)):
               with open(schema, "r") as f:
                    self.schema = json.load(f)
          else:
               self.schema = schema
          
          self.errors: List[str] = []
          self._valid_dtypes = {
               v for k, v in vars(constants.DtypeCastMap).items() if not k.startswith("__")
          }

     def validate(self) -> bool:
          """
          Runs all validation checks and returns a boolean indicating success.
          It collects all errors found in the schema.
          :return: True if the schema is valid, False otherwise.
          """
          self._validate_label_column()
          
          for col_name, col_props in self.schema.items():
               if self._validate_entry_structure(col_name, col_props):
                    self._validate_pattern_based_rules(col_name, col_props)
          
          return not self.errors

     def _validate_label_column(self):
          """Ensures the label column exists and has the correct integer data type."""
          label_col = constants.ColumnNames.LABEL
          if label_col not in self.schema:
               self.errors.append(f"Mandatory label column '{label_col}' not found in schema.")
               return

          label_props = self.schema[label_col]
          if not (isinstance(label_props, list) and label_props):
               return

          label_dtype = label_props[0]
          if label_dtype != constants.DtypeCastMap.INT:
               self.errors.append(
                    f"Label column '{label_col}' must have data type '{constants.DtypeCastMap.INT}', but found '{label_dtype}'."
               )

     def _validate_entry_structure(self, col_name: str, col_props: List[Any]) -> bool:
          """Validates the basic structure of a schema entry, returning True if valid."""
          is_valid = True
          if not (2 <= len(col_props) <= 3):
               self.errors.append(f"Column '{col_name}': Entry must have 2 or 3 elements, but found {len(col_props)}.")
               is_valid = False

          dtype = col_props[0]
          if dtype not in self._valid_dtypes:
               self.errors.append(f"Column '{col_name}': Invalid data type '{dtype}'. Must be one of {self._valid_dtypes}.")
               is_valid = False

          description = col_props[1]
          if not isinstance(description, str) or not description:
               self.errors.append(f"Column '{col_name}': Description must be a non-empty string.")
               is_valid = False
               
          return is_valid

     def _check_type_and_group(
          self, col_name: str, dtype: str, group: str, expected_dtype: str, expected_group: str
     ):
          """Helper to check dtype and group for a column."""
          if dtype != expected_dtype:
               self.errors.append(
                    f"Column '{col_name}': Expected data type '{expected_dtype}', but found '{dtype}'."
               )
          if group != expected_group:
               self.errors.append(
                    f"Column '{col_name}': Expected to belong to group '{expected_group}', but found '{group}'."
               )

     def _validate_pattern_based_rules(self, col_name: str, col_props: List[Any]):
          """Validates rules based on column name patterns."""
          dtype, _, group = (col_props + [None])[:3]

          if re.match(constants.Attendance.PATTERN, col_name):
               self._check_type_and_group(
                    col_name, dtype, group, 
                    constants.DtypeCastMap.STR, 
                    constants.ColumnGroups.ALL_ATTENDANCES
               )
          elif col_name.endswith("_score"):
               self._check_type_and_group(
                    col_name, dtype, group, 
                    constants.DtypeCastMap.FLOAT, 
                    constants.ColumnGroups.EXAM_SCORE_SUBWISE
               )
          elif col_name.endswith("_agg_attnd"):
               self._check_type_and_group(
                    col_name, dtype, group,
                    constants.DtypeCastMap.FLOAT,
                    constants.ColumnGroups.MONTH_AGG_ATTND
               )
          elif col_name.endswith("_attnd"):
               self._check_type_and_group(
                    col_name, dtype, group, 
                    constants.DtypeCastMap.STR, 
                    constants.ColumnGroups.EXAM_ATTND_SUBWISE
               )

     def get_errors(self) -> List[str]:
          """Returns the list of validation errors."""
          return self.errors