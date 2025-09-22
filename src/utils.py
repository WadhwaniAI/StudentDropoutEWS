import json
import pytz
from pathlib import Path
from datetime import datetime
from pathlib import Path
from src import constants


# Define the root of this repo (2 levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def replace_value_in_nested_dict(d, target, replacement):    
     """
     Replaces a target value in a nested dictionary with a replacement value.
     :param d (dict): input dictionary.
     :param target (Any): value to be replaced.
     :param replacement (Any): value to replace the target with.
     Returns: New dictionary with target value replaced with replacement value.
     """
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


def custom_json_formatter(data: dict, indent: int = constants.ConfigSchema.JSON_INDENT) -> str:
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
     

def get_timestamp(timezone: str=constants.DateTime.TIMEZONE) -> str:
     """Returns current timestamp as YYYY-MM-DD_HH:MM:SS."""
     now = datetime.now(pytz.timezone(timezone))
     return f"{now.strftime(constants.DateTime.TIMESTAMP_FORMAT)}"


def resolve_path(path_str: str) -> Path:
     """Returns absolute path, resolving relative ones from PROJECT_ROOT."""
     path = Path(path_str)
     return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()