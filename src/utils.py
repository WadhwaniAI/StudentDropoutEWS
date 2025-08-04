import json
import numpy as np
import os
import pytz
from pathlib import Path
from datetime import datetime
from pathlib import Path
from munch import Munch, munchify
from typing import Union, Dict


# Define the root of this repo (2 levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def add_preds_threshold(df, preds_proba_col, threshold, preds_col='preds_threshold'):

     '''
     Description:
          Computes predictions based on a given threshold
     Args:
          df: input dataframe.
          preds_proba_col: name of column with confidence scores of minority class (class 1)
          threshold: chosen threshold beyond which instance would be classified as 1. 
          preds_col: name of predictions column.
     Returns:
          Dataframe with new predictions column based on the given threshold.
     '''

     # conditions for new predictions
     c1 = (df[preds_proba_col] >= threshold)
     conditions, choices = [c1], [1]

     # add column with new predictions
     df[preds_col] = np.select(conditions, choices, default=0)

     return df
     

def load_config(config_input: Union[str, Dict, Munch]) -> Munch:
     """Loads a configuration and ensures it is returned as a Munch object."""
     if isinstance(config_input, str):
          try:
               with open(config_input, 'r') as f:
                    return munchify(json.load(f))
          except FileNotFoundError:
               raise FileNotFoundError(f"Config file not found: '{config_input}'")
          except json.JSONDecodeError:
               raise ValueError(f"Could not parse JSON from file: '{config_input}'")
     elif isinstance(config_input, (dict, Munch)):
          return munchify(config_input)
     else:
          raise TypeError("Input must be a dictionary, Munch object, or a file path (string).")


def resolve_config_paths(config: dict, keys_to_resolve: list) -> dict:
     """
     Resolves relative or list-of-relative paths in nested config using PROJECT_ROOT.

     Args:
          config (dict): Munch/dict-like config with potential relative paths.
          keys_to_resolve (list): List of dotted keys (e.g., "data.tst_file") to resolve.

     Returns:
          dict: Config with resolved absolute paths (as strings).
     """
     for dotted_key in keys_to_resolve:
          try:
               value = get_nested(config, dotted_key)
          except (KeyError, AttributeError):
               continue  # or raise warning if desired

          if value is None:
               continue

          if isinstance(value, str):
               path = Path(value)
               if not path.is_absolute():
                    resolved = str((PROJECT_ROOT / path).resolve())
                    set_nested(config, dotted_key, resolved)

          elif isinstance(value, list):
               resolved_list = []
               for item in value:
                    item_path = Path(item)
                    if not item_path.is_absolute():
                         item_path = (PROJECT_ROOT / item_path).resolve()
                    resolved_list.append(str(item_path))
               set_nested(config, dotted_key, resolved_list)

          else:
               raise TypeError(f"Expected str or list for key '{dotted_key}', but got {type(value)}")

     return config



def get_nested(config, key_path):
     """
     Retrieves a value from a nested dictionary or munch object using a dot-delimited key path.

     Args:
          config (dict or munch.Munch): The configuration dictionary or Munch object.
          key_path (str): Dot-delimited string representing the nested key, e.g., "data.tst_file".

     Returns:
          The value found at the nested key path.

     Example:
          config = {"data": {"tst_file": "file.pkl"}}
          get_nested(config, "data.tst_file")  # returns "file.pkl"
     """
     keys = key_path.split(".")
     for k in keys:
          config = config[k]
     return config


def set_nested(config, key_path, value):
     """
     Sets a value in a nested dictionary or munch object using a dot-delimited key path.

     Args:
          config (dict or munch.Munch): The configuration dictionary or Munch object.
          key_path (str): Dot-delimited string representing the nested key, e.g., "data.tst_file".
          value: The value to set at the specified nested key path.

     Example:
          config = {"data": {"tst_file": "old.pkl"}}
          set_nested(config, "data.tst_file", "new.pkl")
          # config now becomes: {"data": {"tst_file": "new.pkl"}}
     """
     keys = key_path.split(".")
     for k in keys[:-1]:
          config = config[k]
     config[keys[-1]] = value


def round_to_n(x, n):
     """Rounds x to n significant figures."""
     mant, exp = f"{x:.{n}e}".split('e')
     mant = str(round(float(mant) * 10**(n-1)) / 10**(n-1))
     return float(f"{mant}e{exp}")


def get_config_files(config_path: str) -> list:
     """Returns list of .json config file paths from a file or directory."""
     if not os.path.exists(config_path):
          print("No configs to run.")
          return []
     elif os.path.isfile(config_path) and config_path.endswith(".json"):
          return [config_path]
     elif os.path.isdir(config_path):
          return [os.path.join(config_path, f) for f in os.listdir(config_path) if f.endswith(".json")]
     else:
          raise ValueError("Expected a .json file or directory containing .json files.")


def replace_value_in_nested_dict(d, target, replacement):
    
     '''
     Description:
          Replaces a target value in a nested dictionary with a replacement value.
     Args:
          d: input dictionary.
          target: value to be replaced.
          replacement: value to replace the target with.
     Returns:
          New dictionary with target value replaced with replacement value.
     '''

     # Create a new dictionary to avoid modifying the original one during recursion
     new_dict = {}

     for key, value in d.items():
          if isinstance(value, dict):
               # Recursively call the function for nested dictionaries
               new_dict[key] = replace_value_in_nested_dict(value, target, replacement)
          elif value == target:
               # Replace the target value with the replacement
               new_dict[key] = replacement
          elif isinstance(value, list):
               # Replace the target value with the replacement in lists
               new_dict[key] = [replacement if item == target else item for item in value]
          else:
               # Keep the value as is if it doesn't match the target
               new_dict[key] = value

     return new_dict


def custom_json_formatter(data: dict, indent: int = 5) -> str:
     """Formats a dict into JSON with 5-space indentation and inline lists."""

     def format_item(item, level):
          """Recursively formats dicts and lists."""
          pad = ' ' * level
          if isinstance(item, dict):
               if not item:
                    return '{}'
               body = [f"{' ' * (level + indent)}{json.dumps(k)}: {format_item(v, level + indent)}" for k, v in item.items()]
               return f"{{\n" + ",\n".join(body) + f"\n{pad}}}"
          elif isinstance(item, list):
               return '[' + ', '.join(format_item(v, level) for v in item) + ']'
          return json.dumps(item)

     return format_item(data, 0)


def sort_nested_dict(d):
     
     '''
     Description:   
          Recursively sorts a nested dictionary by its keys.
     Args:
          d: input dictionary.
     Returns:
          Sorted dictionary.
     '''

     if isinstance(d, dict):
          # Sort the dictionary by keys and recursively sort nested dictionaries
          return {k: sort_nested_dict(v) for k, v in sorted(d.items())}
     
     elif isinstance(d, list):
          # If the value is a list, recursively sort any dictionaries within the list
          return [sort_nested_dict(item) for item in d]
     
     else:
          # Return the value as is if it's not a dictionary or list
          return d
     

def get_timestamp(timezone: str='Asia/Kolkata') -> str:
     """Returns current timestamp as YYYY-MM-DD_HH:MM:SS."""
     now = datetime.now(pytz.timezone(timezone))
     return f"{now.strftime('%Y-%m-%d_%H:%M:%S')}"


def resolve_path(path_str: str) -> Path:
     """Returns absolute path, resolving relative ones from PROJECT_ROOT."""
     path = Path(path_str)
     return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()