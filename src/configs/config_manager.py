from pathlib import Path
from typing import Union, Dict, Any
from munch import Munch, munchify
from .utils import load_config
from src.utils import custom_json_formatter
from src import constants


class ConfigManager:
     """A class to load, validate, update, and manage a Config, providing a validated Munch object for experiments."""

     def __init__(
               self,
               config_input: Union[str, Dict, Munch],
               config_schema: Union[
                   str, Dict, Munch
               ] = constants.MetadataPaths.CONFIG_SCHEMA,
     ):
          """
          Initializes the manager, loads config & schema, and runs validation.
          :param config_input (Union[str, Dict, Munch]): The Config, as a path, dict, or Munch.
          :param config_schema (Union[str, Dict, Munch]): The schema to validate against.
          """
          self.schema = load_config(config_schema)
          self.config = load_config(config_input)
          self.errors = []
          self._validate()

     @property
     def is_valid(self) -> bool:
          """Returns True if the current Config is valid against the schema."""
          return not self.errors

     @staticmethod
     def _get_expected_type(schema_value: Any) -> type:
          """Infers the expected Python type from a schema's placeholder value."""
          if isinstance(schema_value, (dict, Munch)):
               return (dict, Munch)
          if isinstance(schema_value, list):
               return list
          if isinstance(schema_value, str):
               if any(
                   ph in schema_value
                   for ph in constants.ConfigSchema.TYPE_PLACEHOLDERS["INT"]
               ):
                    return int
               if any(
                   ph in schema_value
                   for ph in constants.ConfigSchema.TYPE_PLACEHOLDERS["FLOAT"]
               ):
                    return float
               if any(
                   ph in schema_value
                   for ph in constants.ConfigSchema.TYPE_PLACEHOLDERS["BOOL"]
               ):
                    return bool
               if any(
                   ph in schema_value
                   for ph in constants.ConfigSchema.TYPE_PLACEHOLDERS["FLOAT_OR_STR"]
               ):
                    return (str, float)
          return str

     @staticmethod
     def _deep_update(source_dict: Munch, update_dict: Munch) -> Munch:
          """
          Recursively updates a Munch object with values from another.
          :param source_dict (Munch): The original Config to update.
          :param update_dict (Munch): The new keys and values to include.
          """
          for key, value in update_dict.items():
               if isinstance(value, Munch) and key in source_dict and isinstance(source_dict.get(key), Munch):
                    source_dict[key] = ConfigManager._deep_update(source_dict[key], value)
               else:
                    source_dict[key] = value
          return source_dict

     def _recursive_validate(self, user_conf: Munch, schema: Munch, path: str):
          """
          Recursively compares the user config against the schema, handling mandatory and optional (<key>) parameters.
          :param user_conf (Munch): The user's Config.
          :param schema (Munch): The schema to validate against.
          :param path (str): The current dot-notation path for error reporting.
          """
          # Check for MISSING MANDATORY keys (by iterating through the schema)
          for schema_key in schema.keys():
               if schema_key.startswith("<") and schema_key.endswith(">"):
                    continue
               if schema_key not in user_conf:
                    self.errors.append(f"Missing key at '{path}': '{schema_key}'")

          # Validate all keys PRESENT in the user's config
          for key, user_val in user_conf.items():
               current_path = f"{path}.{key}"
               schema_val = None

               if key in schema:
                    schema_val = schema[key]
               else:
                    optional_key = f"<{key}>"
                    if optional_key in schema:
                         schema_val = schema[optional_key]
                    else:
                         # This finds a generic placeholder like '<param_name>'
                         generic_placeholder = next((k for k in schema.keys() if k.startswith("<")), None)
                         if generic_placeholder:
                              schema_val = schema[generic_placeholder]

               # If no schema rule was found after all strategies, it's an extra key
               if schema_val is None:
                    self.errors.append(f"Extra key not in schema at '{path}': '{key}'")
                    continue

               # Now that we have the correct schema rule, validate the type
               expected_type = self._get_expected_type(schema_val)
               if not isinstance(user_val, expected_type) and user_val is not None:
                    expected_name = getattr(expected_type, '__name__', str(expected_type))
                    self.errors.append(
                         f"Type mismatch at '{current_path}': Expected {expected_name}, got {type(user_val).__name__}."
                    )
                    continue # Skip deeper validation if type is wrong

               # If the value is a dictionary, recurse deeper
               if isinstance(user_val, Munch):
                    self._recursive_validate(user_val, schema_val, current_path)

     def _validate(self) -> bool:
          """Internal method to clear errors and re-run the validation logic."""
          self.errors = []
          self._recursive_validate(self.config, self.schema, "root")
          return self.is_valid

     def update(self, additions: Union[Dict, Munch]):
          """Deeply updates the Config with new values and re-validates."""
          self.config = self._deep_update(self.config, munchify(additions))
          self._validate()

     def save(self, path: str, indent: int = constants.ConfigSchema.JSON_INDENT):
          """
          Validates and saves the current Config using custom JSON formatting.
          :param path (str): The full path where the Config file will be saved.
          :param indent (int): The number of spaces to use for indentation.
          Raises ValueError: If the Config is invalid against the schema.
          """
          config_to_save = self.get_validated_config()
          Path(path).parent.mkdir(parents=True, exist_ok=True)
          with open(path, "w") as f:
               f.write(custom_json_formatter(config_to_save, indent=indent))
          print(f"✅ Config successfully saved to: {path}")

     def get_validated_config(self) -> Munch:
          """
          The primary method to get a Config for an experiment run.
          Return (Munch): The validated Config munch (dictionary) object.
          Raises ValueError: If the Config is invalid against the schema.
          """
          if not self.is_valid:
               error_summary = "\n- ".join(self.errors)
               raise ValueError(f"Config is invalid. Please fix the following errors:\n- {error_summary}")
          return self.config