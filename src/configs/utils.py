import json
import os
from munch import Munch, munchify
from typing import Union, Dict


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