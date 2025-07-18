import geopy.distance
import inspect
import json
import munch
import numpy as np
import os
import pandas as pd
import pickle
import pytz
import random
import re
import toml
import tomli

from datetime import datetime, date
from pathlib import Path

from catboost import CatBoostClassifier


# Define the root of this repo (2 levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def append_cols(df1, df2, cols: list, index_col):

     '''
     Description:
          Append columns from second dataframe to first dataframe.
     Args:
          df1: Dataframe to which colums are to be added.
          df2: Dataframe from which columns are to be added to df1.
          cols: Which colums from df2 to be added to df1.
          index_col: The unique ID using which the instances would be matched before appending the columns.
     Returns:
          Dataframe df1 with new columns added.
     '''

     if index_col not in cols:
          cols.append(index_col)

     assert index_col in df1.columns
     assert index_col in df2.columns

     return pd.merge(left=df1, right=df2[cols], how='left', on=index_col)



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
     


def load_data_file(file_path):
     
     '''
     Description:
          General function to load a csv, excel or pickle tabular file.
     '''

     # get extension of file
     ext = file_path.split(".")[-1]
     
     # load file based on extension
     if ext == "csv": 
          df = load_csv(file_path=file_path)
     if ext == "pkl" : 
          df = load_pickle(file_path=file_path)
     if ext == "xlsx": 
          df = load_excel(file_path=file_path)
     
     return df
     


def load_pickle(file_path: str):
     
     '''
     Description:
          Function to a pickle tabular file, lower-case all column names and convert them to string.
     '''

     # load pickle file
     df = pd.read_pickle(file_path)

     return df



def load_csv(file_path: str):

     '''
     Description:
          Function to a csv tabular file, lower-case all column names and convert them to string.
     '''

     # read first two rows to get column names
     df = pd.read_csv(filepath_or_buffer=file_path, nrows=2)
     
     # load dataframes in chunks
     chunks = pd.read_csv(
          filepath_or_buffer=file_path, 
          usecols=list(df.columns), 
          chunksize=1000, dtype={x: str for x in list(df.columns)}
     )
     
     # concatenate all chunks
     df = pd.concat([chunk for chunk in chunks], axis = 0)
     
     return df



def load_excel(file_path: str):

     '''
     Description:
          Function to a excel tabular file, lower-case all column names and convert them to string.
     '''
     
     # load excel file
     df = pd.read_excel(file_path)

     return df



def oversample_df(df, factor: float):

     idx = {}
     for key in [0, 1]:
          idx[f'{key}'] = df.index[df['target']==key].tolist()
          random.shuffle(idx[f'{key}'])

     df_oversampled = pd.concat([*[df.loc[idx['1']] for i in range(0, factor)], df.loc[idx['0']]], axis=0)

     return df_oversampled.sample(frac=1)



def generate_npy_files(df, df_type, save_folder_path, labels, num_features=None, bin_features=None, cat_features=None,):
     
     '''
     Description:
          Save numpy files of an input dataframe for use in TabR experiments.
     '''

     if not os.path.exists(save_folder_path):
          os.makedirs(save_folder_path)

          if num_features is not None:
               np.save(f"{save_folder_path}/X_num_{df_type}.npy", np.asarray(df[num_features], dtype=np.float32))
          
          if cat_features is not None:
               np.save(f"{save_folder_path}/X_cat_{df_type}.npy", np.asarray(df[cat_features], dtype=np.str_))
          
          if bin_features is not None:
               np.save(f"{save_folder_path}/X_bin_{df_type}.npy", np.asarray(df[bin_features], dtype=np.float32))
          
          np.save(f"{save_folder_path}/Y_{df_type}.npy", np.asarray(df[labels], dtype=np.int64)) 
     


def load_config(
          file_path: str, 
          keys_to_resolve: list=["exp.root_data", "exp.root_exps", "data.trn_file", "data.tst_file", "data.holidays"]
) -> munch.Munch:
     
     '''
     Description:
          Loads a toml or json config file. 
     Args:
          file_path: path to the config file
     Returns:
          A munchify object which is a dictionary.
     '''

     # check if extension is valid
     ext = file_path.split(".")[-1]
     assert ext in ["toml", "json"]
     
     # load config file
     if ext == "toml":
          with open(file_path, mode="rb") as f:
               config = tomli.load(f)
     elif ext == "json":
          with open(file_path, mode="r") as f:
               config = json.load(f)

     # Resolve specified path keys
     if keys_to_resolve:
          config = resolve_config_paths(config, keys_to_resolve)

     return munch.munchify(config)



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


     
def postprocess_csv(df, groupby_columns: list, agg_column: str, save_csv: str, threshold=None):
     
     '''
     Description:
          Extracts a dataframe based on pandas groupby conditions.
     Args:
          df: input dataframe.
          groupby_columns: columns in decreasing order of priority to group instances in the dataframe.
          agg_column: could be mean or count or sum, the function to aggegate with.
          save_csv: save the resultant dataframe to a csv.
          threshold: if predictions are for a particular threshold, incorporates in column name.
     Returns:
          Returns None. Saves the extracted sample to a file derived from the source file.
     '''

     df.groupby(groupby_columns)[agg_column]\
          .agg(['mean', 'sum', 'count'])\
          .reset_index()\
          .rename(columns={"mean":f"predicted prevalence (threshold={threshold})", "sum":"number of predicted dropouts", "count":"total number of students"})\
          .to_csv(f"{save_csv}", index=True)



def aerial_distance(coordinates1: tuple=None, coordinates2: tuple=None):
     
     '''
     Description:
          Computes the geodesic distance between a pair of coordinates.
     Args:
          coordinates1: a tuple with latitude and longitude of point 1.
          coordinates2: a tuple with latitude and longitude of point 2.
     Returns:
          The geodesic distance between the two points in km.
     '''

     # compute distance
     return round(geopy.distance.geodesic(coordinates1, coordinates2).km, 5)



def round_to_n(x, n):

     '''
     Description:
          Rounds a number to have `n` significant figures.
          This function formats the input number in scientific notation to isolate the mantissa and exponent, 
          and then rounds the mantissa to the specified number of significant digits. The final result is converted back to a float.
     Args:
          x (float): The number to round.
          n (int): The number of significant figures to retain.
     Returns:
          float: The input number rounded to `n` significant figures.
     Example:
          >>> round_to_n(0.0123456, 2)
          0.012

          >>> round_to_n(12345.6, 3)
          12300.0
     '''
     # gives 1.n figures
     fmt = '{:1.' + str(n) + 'e}'    
     
     # get mantissa and exponent
     p = fmt.format(x).split('e')    

     # round "extra" figure off mantissa
     p[0] = str(round(float(p[0]) * 10**(n-1)) / 10**(n-1))
     
     # convert str to float
     return float(p[0] + 'e' + p[1]) 
          


def generate_cols_to_exclude(holidays_filepath: str, output_filepath: str = None):
     """
     Description:
          Reads a holidays JSON file, generates columns to exclude, and writes to a JSON file.
     Args:
          holidays_filepath (str): Path to the holidays.json file.
          output_filepath (str, optional): Path to save the cols_to_exclude.json file. 
               If None, saves in the same directory as holidays_filepath.
     Returns:
          None: The function saves the output to a file and prints a confirmation message.
     """
     with open(holidays_filepath, 'r') as f:
          holidays = json.load(f)

     months = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
     cols_to_exclude = {}

     for ay, monthly_data in holidays.items():
          ay = str(ay)
          cols_to_exclude[ay] = []
          for m in months:
               m_str = str(m)
               if m_str not in monthly_data:
                    continue
               for holiday_type, days in monthly_data[m_str].items():
                    cols_to_exclude[ay].extend([f"{m}_{d}" for d in days])
          # remove duplicates and sort
          cols_to_exclude[ay] = sorted(list(set(cols_to_exclude[ay])))

     if output_filepath is None:
          output_filepath = holidays_filepath.replace('holidays.json', 'cols_to_exclude.json')

     with open(output_filepath, 'w') as f:
          json.dump(cols_to_exclude, f, indent=5)

     print(f"Saved cols_to_exclude to {output_filepath}")



def get_config_files(config_path: str, ext: str) -> list:
     '''
     Description:
          Returns a list of config file(s) based on the given path and extension.

          - If `config_path` is a directory, it returns all files in that directory
               with the specified extension.
          - If `config_path` is a single file path, it returns a list containing only that file,
               regardless of extension.

     Args:
          config_path (str): Path to a directory or a full path to a config file.
          ext (str): Extension of the config files (e.g., "json", "yaml").

     Returns:
          list: List of config file paths.
     '''

     # If the path is a file, return it directly
     if os.path.isfile(config_path):
          return [config_path]

     # If the path is a directory, list and filter files by extension
     if os.path.isdir(config_path):
          files = [
               os.path.join(config_path, f)
               for f in os.listdir(config_path)
               if f.endswith(f".{ext}")
          ]
          return files

     # Invalid path
     return []



def get_academic_year(ds_name: str) -> str:

     '''
     Description:
          Returns the academic year based on the dataset name.
     Args:
          ds_name: name of the dataset.
     Returns:
          academic year as string.
     '''
     
     # academic year based on dataset name
     if "prod" in ds_name: 
          year = "2425"
     elif "srv" in ds_name:
          year = "2324"
     elif "trn" in ds_name or "train" in ds_name or "tst" in ds_name:
          year = "2223"
     
     # assert year is valid
     assert year in ["2223", "2324", "2425"]

     return year



def word_label(x):

     '''
     Description:
          Converts a binary label to a word label.
     Args:
          x: binary label.
     Returns:
          word label.
     '''

     # check if x is binary
     assert x in [0, 1]

     # convert binary to word
     if x == 0: 
          return "NO"
     if x == 1: 
          return "YES"



def make_demo_config(
          config: munch.Munch,
          save_config_dir: str,
          save_ext: str="toml"
) -> None:

     '''
     Description:
          Generates a demo config file for a given grade.
     Args:
          config: munchified config file (dictionary).
          save_config_dir: to make demo of this directory.
     Returns:
          None. Saves the demo config file in the destination directory.
     '''

     # save config directory
     save_config_dir = f"{save_config_dir} demo"
     os.makedirs(save_config_dir, exist_ok=True)

     # title for demo
     config.exp.title = f"{config.exp.title}, demo"

     # n_trials for demo
     config.model.n_trials = 1

     # iterations
     config.model.params.int.iterations.low = 5
     config.model.params.int.iterations.high = 30
     config.model.params.int.iterations.step = 5

     # device (gpu)
     config.model.params.fixed.devices = "7"

     # save config based on extn
     assert save_ext in ["toml", "json"]
     if save_ext == "toml":
          with open(f"{save_config_dir}/{config.exp.title}.toml", "w") as f:
               toml.dump(config, f)
     elif save_ext == "json":
          custom_json = custom_json_formatter(config)
          with open(f"{save_config_dir}/{config.exp.title}.json", "w") as file:
               file.write(custom_json)
     


def get_full_dir_from_dir(
          dir: str,
          root_dir: str="/ews-all-data/experiments",
) -> str:

     '''
     Description:
          Returns the full path of a directory.
     Args:
          dir: directory.
     Returns:
          full path of the directory.
     '''

     # grade from dir
     grade = int(re.findall("grade \d", dir)[0][-1])

     # base directory of experiments
     exp_dir = dir[:dir.index(', grade')]

     # dataset
     full_dir = f"{root_dir}/{exp_dir}/grade {grade}/{dir}"

     return full_dir, grade
     


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



def custom_json_formatter(dictionary, indent=5):

     '''
     Description:
          Custom JSON formatter that indents for new keys but not for lists.
          Works for nested dictionaries and lists.
     Args:
          dictionary: input dictionary.
          indent: number of spaces to indent.
     Returns:
          JSON formatted string.          
     '''

     def format_item(item, level, max_indent_level=25):

          '''
          Description:
               Helper function to format each item in the dictionary.
          Args:
               item: input item.
               level: current level of indentation.
               max_indent_level: maximum level of indentation.
          Returns:
               JSON formatted string for the item.
          '''

          # check for level
          if level > max_indent_level:
               return json.dumps(item)
          else:
               if isinstance(item, dict):
                    if len(item.keys()) > 0:
                         return "{\n" + ",\n".join(
                              f"{' ' * (level + indent)}{json.dumps(k)}: {format_item(v, level + indent)}"
                              for k, v in item.items()
                         ) + "\n" + " " * level + "}"
                    else:
                         return "{ " + ", ".join(
                              f"{json.dumps(k)}: {format_item(v, level + indent)}"
                              for k, v in item.items()
                         ) + " }"
               elif isinstance(item, list):
                    return "[" + ", ".join(format_item(i, level) for i in item) + "]"
               else:
                    return json.dumps(item)

     return format_item(dictionary, 0)



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
     


def get_discard_ds_name(ds_name: str) -> str: 

     '''
     Description:
          Returns the dataset name to discard samples based on the input dataset name.
     Args:
          ds_name: input dataset name.
     Returns:
          Dataset name to discard.
     '''

     # dataset name to discard
     if ds_name in ["prod", "prod[sat]", "prod[fullsat]"]:
          return "prod"
     elif ds_name in ["srv", "srv-upd", "srv-upd[fullsat]"]:
          return "srv-upd"
     elif ds_name in ["trn", "train", "train[fullsat]"]:
          return "train"



def get_model_features(dir: str) -> tuple:

     '''
     Description:          
          Returns the numerical, categorical and all features from an experimental directory.
     Args:
          dir: experimental directory.
     Returns:
          Numerical, categorical and all features.
     '''

     # numerical features
     with open(f"{dir}/num_features.pkl","rb") as f:
          num_features = pickle.load(f)

     # categorical features
     with open(f"{dir}/cat_features.pkl","rb") as f:
          cat_features = pickle.load(f)
     
     # all features
     features = [*num_features, *cat_features]
          
     # return features
     return cat_features, num_features, features



def save_model_features(
          exp_dir: str,
          cat_features: list,
          num_features: list,
) -> None:

     '''
     Description:
          Saves the numerical and categorical features as pickle files.
     Args:
          exp_dir: experimental directory.
          cat_features: categorical features.
          num_features: numerical features.
     Returns:
          None. Saves the features as pickle files.
     '''

     # save categorical features
     with open(f"{exp_dir}/cat_features.pkl","wb") as f:
          pickle.dump(cat_features, f)

     # save numerical features
     with open(f"{exp_dir}/num_features.pkl","wb") as f:
          pickle.dump(num_features, f)
     

def get_timestamp(
          timezone: str='Asia/Kolkata'
) -> str:

     '''
     Description:
          Returns the current timestamp in the format: YYYY-MM-DD_HH:MM:SS.
     Returns:
          Current timestamp.
     '''

     # Set the timezone to Indian Standard Time (IST)
     india_timezone = pytz.timezone(timezone)

     # Get the current time in IST
     now = datetime.now(india_timezone)

     # Format the timestamp with the desired format
     timestamp = now.strftime(f"{date.today().strftime('%Y-%m-%d')}_{now.strftime('%H:%M:%S')}")

     # return timestamp
     return timestamp



def get_students_to_discard( 
          students_options: dict,
          ay: str,
          grade: int,
          discard_period: str,
          discard_options: list,
): 

     '''
     Description:
          Identifies and returns a set of students to discard based on given discard options.
          This function loops through a list of discard criteria and aggregates the corresponding student sets from the `students_options` dictionary. 
          Each key in `students_options` is expected to follow the format: "ay{ay}, grade{grade}, {discard_period}[frac_m]={discard_option}"
     Args:
          students_options (dict): A dictionary mapping formatted keys to sets of student IDs.
          ay (str): Academic year string (e.g., '2425' for academic year 2024-25).
          grade (int): The grade/class level of students.
          discard_period (str): The period or time-window label for discarding logic.
          discard_options (list): A list of values for the discard option (typically fractions or thresholds).
     Returns:
          set: A set of student IDs that are to be discarded, aggregated from all discard options.
     '''

     # initialise students to discard
     students_to_discard = set()

     # loop over all discard options
     for discard_option in discard_options:
          
          current_students = students_options[f"ay{ay}, grade{grade}, {discard_period}[frac_m]={discard_option}"]
          students_to_discard = students_to_discard.union(current_students)
     
     return students_to_discard



def public_school_filter(df: pd.DataFrame):

     '''
     Description:
          Filters the input dataframe to include only public schools.
     Args:
          df: input dataframe.
     Returns:
          Filtered dataframe with only public schools.
     '''

     # filter for public schools
     df.columns = map(str.lower, df.columns)
     df = df[df["schcat"].isin(["1", "2", "3", "4", "5", "6", "7"])]
     df = df[~df["schmgt"].isin(["5", "92", "93", "94", "95", "97", "101"])]

     return df



def public_private_school_filter(df: pd.DataFrame):

     '''
     Description:
          Filters the input dataframe to into student from public schools and private schools.
     Args:
          df: input dataframe.
     Returns:
          Two filtered dataframes, first of students in public schools and second of students in private schools.
     '''

     # ensure all columns are in lower case
     df.columns = map(str.lower, df.columns)

     # Define the public school condition
     public_condition = df["schcat"].isin(["1", "2", "3", "4", "5", "6", "7"]) & ~df["schmgt"].isin(["5", "92", "93", "94", "95", "97", "101"])

     # Public school students
     public_students = df[public_condition]

     # Private (non-public) school students — the rest
     private_students = df[~public_condition]

     # Return the filtered dataframes
     return public_students, private_students



def filter_valid_params(params):
     """
     Description:
          Filters the input parameters dictionary to include only valid CatBoostClassifier parameters.  
     Args:
          params (dict): Dictionary of parameters to filter.
     Returns:
          dict: Filtered dictionary containing only valid CatBoostClassifier parameters.
     """

     valid_keys = inspect.signature(CatBoostClassifier.__init__).parameters.keys()
     valid_keys = set(valid_keys) - {"self"}
     return {k: v for k, v in params.items() if k in valid_keys}



def resolve_path(path_str: str) -> Path:
    """
    Resolves a given path string to an absolute path.

    If the input path is already absolute, it is returned as-is.
    If the input path is relative, it is resolved relative to the PROJECT_ROOT.

    Args:
        path_str (str): The input path string (can be relative or absolute).

    Returns:
        Path: An absolute Path object pointing to the resolved location.
    """
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

