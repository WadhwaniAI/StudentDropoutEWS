import json
import pandas as pd
import numpy as np

from utils import resolve_path



def basic_preprocess(
          df: pd.DataFrame,
          index: str = "aadhaaruid",
          label: str = "target",
          column_dictionary_path: str="data/column_dictionary.json",
          feature_groups_path: str="data/feature_groups.json"
) -> pd.DataFrame:

     '''
     Description:
          Basic preprocessing of the data which includes the following operations.
          - removing duplicate rows based on a unique index column.
          - correcting the datatypes of all columns.
          - rectifying the attendances using exam scores and attendances of questions of the same examination (LO-wise datasets).
          - establishing feature groups for bundled use.
     Args:
          df: input dataframe.
          index: the unique index column of the dataframe.
          label: the label column of the dataframe.
          column_dictionary_path: path to the column dictionary JSON file.
          feature_groups_path: path to the feature groups JSON file.
     Returns:
          A basic preprocessed dataframe.
     '''

     # Resolve paths relative to project root if needed
     column_dictionary_path = resolve_path(column_dictionary_path)
     feature_groups_path = resolve_path(feature_groups_path)

     # lower-case all column names
     df.columns = map(str.lower, df.columns)

     # remove duplicate rows based on the index column
     df = df.drop_duplicates(subset=index)

     # load the master column dictionary
     with open(column_dictionary_path, "r", encoding="utf-8") as f:
          column_dictionary = json.load(f)

     # load the master feature groups
     with open(feature_groups_path, "r", encoding="utf-8") as f:
          feature_groups = json.load(f)

     # update the column dictionary and feature groups with the columns present in the dataframe
     column_dictionary, feature_groups = validate_and_filter_metadata(df, column_dictionary, feature_groups, reserved_columns={index, label})

     # initialize lists to hold columns based on their datatypes
     cols_to_int, cols_to_float, cols_to_str = [], [], []

     # categorize columns based on their datatypes
     for col in df.columns:
          if column_dictionary[col][0] == "int":
               cols_to_int.append(col)
          elif column_dictionary[col][0] == "float":
               cols_to_float.append(col)
          elif column_dictionary[col][0] == "str":
               cols_to_str.append(col)

     # convert columns to appropriate datatypes
     df[cols_to_str] = df[cols_to_str].astype(str)
     df[cols_to_float] = df[cols_to_float].astype(np.float64)
     df[cols_to_int] = df[cols_to_int].astype("int64")

     # replace all missing values in daily attendances with 'm', presence and absence with 'p' and 'a' respectively
     df[feature_groups["all_attendances"]] = df[feature_groups["all_attendances"]].replace({'nan':'m', '5':'m', '0':'m', '1':'p', '2':'a'})

     # loop over exam_attnd columns to rectify them using corresponding exam_scores
     for col_attnd in [*feature_groups["exam_attnd_subwise"], *feature_groups["exam_attnd_lowise"]]:
          col_score = col_attnd.replace("attnd", "score")
          df[col_attnd] = df[col_score].apply(rectify_attendance_using_score)
     
     # fillna in exam_scores after using them to rectify exam_attendance
     df[feature_groups["exam_score_subwise"]] = df[feature_groups["exam_score_subwise"]].fillna(0)
     df[feature_groups["exam_score_lowise"]] = df[feature_groups["exam_score_lowise"]].fillna(0)

     # rectify attendances of an examination using attendances of the questions of the same examination (using LO-wise data)
     subjects = set([col.split("_")[0] for col in feature_groups["exam_attnd_subwise"]])
     for sub in subjects:
          # get all relevant columns
          cols = [col for col in [*feature_groups["exam_attnd_subwise"], *feature_groups["exam_attnd_lowise"]] if sub in col]
          
          # rectify the attendance columns using the other attendance columns
          df = rectify_attendance_using_attendances(df, cols)

     # return the basic preprocessed dataframe
     return df, feature_groups



def rectify_attendance_using_score(x: float) -> str:

     '''
     Description:
          Rectifies exam attendance columns using corresponding exam scores of students. 
          Students who attend an exam don't get the score 0. 
          If a student has score 0, most likely the student missed the exam.
          In case, none of the above conditions are met, the attendance is considered missing.
     Args:
          x: exam score.
     Returns:
          Rectified attendance.
     '''

     if x > 0:
          return "p"
     elif x == 0:
          return "a"
     else:
          return "m"



def rectify_attendance_using_attendances(
          df: pd.DataFrame, 
          cols: list
) -> pd.DataFrame: 
     
     '''
     Description:
          Use given attendances to rectify each of them based on inconsistencies.
          Primarily used to rectify attendance columns in Learning-Outcome wise datasets by looking at consistency in question-wise attendances.
          This is a second-level check after the first-level check using exam scores.
     Args:
          df: the input dataframe.
          cols: list of relevant attendance columns to rectified.
     Returns:
          Dataframe. This would be the corrected columns. 
     '''

     # concatenated attendances
     dummy = "concatenated_attendance"

     # combine attendance columns. 
     #df[dummy] = df[cols].agg(''.join, axis=1)
     df = pd.concat([df.drop(columns=[dummy], errors='ignore'), df[cols].agg(''.join, axis=1).rename(dummy)], axis=1)

     # rectify each column based on the dummy column
     cols_to_concat = {}
     for col in cols:          
          cols_to_concat[col] = df[dummy].apply(rectify_attendance_using_concatenated_attendance)
     
     # add the new columns to the dataframe
     cols_to_concat_df = pd.DataFrame(cols_to_concat, index=df.index)
     df = pd.concat([df.drop(columns=cols_to_concat_df.columns, errors="ignore"), cols_to_concat_df], axis=1)
     
     # drop the dummy column
     df.drop(columns=[dummy], inplace=True)

     # return the dataframe
     return df



def rectify_attendance_using_concatenated_attendance(x: str) -> str:

     '''
     Description:
          Rectifies attendance columns using a the column of concatenated attendances. 
          This works for Learning-Outcome wise datasets. 
          For a particular examination, we get individual attendances for each question. We assume that if a student is present for one, he / she is present for the rest too.
          Thus, 
               if a "p" is present in the concatenated attendance, we assume that the student is present for the examination.
               if an "a" is present in the concatenated attendance, we assume that the student is absent for the examination.
               if neither "p" nor "a" is present, we assume that the student is missing for the examination.
     Args:
          x: column of concatenated attendances.
     Returns:
          Rectified attendance.
     '''

     if "p" in x:
          return "p"
     elif "a" in x:
          return "a"
     else:
          return "m"



def validate_and_filter_metadata(
          df,
          column_dictionary,
          feature_groups,
          reserved_columns=None
):
     """
     Validates that:
          - all df columns exist in column_dictionary
          - all df columns (excluding reserved_columns) exist in feature_groups

     Then removes any keys from column_dictionary and feature_groups
     that are not present in the DataFrame.

     Parameters
     ----------
     df : pd.DataFrame
          The DataFrame whose columns should be validated.

     column_dictionary : dict
          Dictionary mapping column names to metadata or data types.

     feature_groups : dict
          Dictionary mapping group names to lists of features.

     reserved_columns : set, list, or tuple, optional
          Columns that should be ignored in feature_groups validation.

     Returns
     -------
     column_dictionary_filtered : dict
          Filtered dictionary containing only relevant columns present in df.

     feature_groups_filtered : dict
          Filtered dictionary with groups containing only features present in df.

     Raises
     ------
     ValueError
          If any required columns are missing in the dictionaries.
     """

     # Set of all DataFrame columns
     df_columns = set(df.columns)

     # 1) Check all df columns are in column_dictionary
     dict_columns = set(column_dictionary.keys())
     missing_in_dict = df_columns - dict_columns
     if missing_in_dict:
          raise ValueError(
               f"Columns missing in column_dictionary: {sorted(missing_in_dict)}"
          )

     # 2) Check all non-reserved df columns are in feature_groups
     df_columns_nonreserved = df_columns - set(reserved_columns)

     group_columns = set()
     for feat_list in feature_groups.values():
          group_columns.update(feat_list)

     missing_in_groups = df_columns_nonreserved - group_columns
     if missing_in_groups:
          raise ValueError(
               f"Columns missing in feature_groups: {sorted(missing_in_groups)}"
          )

     print("✅ All DataFrame columns found in column_dictionary, "
               "and all non-reserved columns found in feature_groups.")

     # Filter column_dictionary
     column_dictionary_filtered = {
          col: val for col, val in column_dictionary.items() if col in df_columns
     }

     # Filter feature_groups
     feature_groups_filtered = {}
     for group, feat_list in feature_groups.items():
          filtered_feats = [f for f in feat_list if f in df_columns_nonreserved]
          feature_groups_filtered[group] = filtered_feats

     return column_dictionary_filtered, feature_groups_filtered
