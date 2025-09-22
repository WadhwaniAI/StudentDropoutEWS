import itertools
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .utils import (
     determine_columns_to_drop,
     split_features_by_dtype,
     holidays_academic_year_wise,
)
from src.utils import resolve_path
from src import constants


class EngineerAttendanceFeatures:
     def __init__(self,
               holidays_calendar_path: str = constants.MetadataPaths.HOLIDAYS_CALENDAR,
               index: str = constants.ColumnNames.INDEX,
               label: Optional[str] = constants.ColumnNames.LABEL,
     ):
          """
          Initializes the feature engineer with holiday mappings.
          :param holidays_calendar_path: Path to the holidays calendar file.
          :param index: Name of the index column in the DataFrame.
          :param label: Optional. Name of the label column if present in the DataFrame.
          """
          self.holidays = holidays_academic_year_wise(
               resolve_path(holidays_calendar_path)
          )
          self.index = index
          self.label = label

          # State variables - will be set during configuration
          self._configured = False
          self._pattern_strings: List[str] = []
          self._feature_config: Dict = {}
          self.attendance_chars = constants.Attendance.CHARS

     def configure_features(
               self,
               groups_of_months: Dict[str, List[int]] = constants.FeatureEngineering.GROUPS_OF_MONTHS,
               combs_of_chars: List[List] = constants.FeatureEngineering.CHAR_COMBINATIONS,
               partitions: List[int] = constants.FeatureEngineering.PARTITIONS,
               disc_cols_miss_frxn: float = constants.FeatureEngineering.MISSING_FRACTION,
               months_for_binary: List[int] = constants.FeatureEngineering.MONTHS_FOR_BINARY,
               absence_thresholds: List[int] = constants.FeatureEngineering.ABSENCE_THRESHOLDS
     ) -> None:
          """
          Configure the feature generation parameters. This should be called once before processing dataframes.
          :param groups_of_months: Dictionary mapping group names to month lists for pattern features
          :param combs_of_chars: Character combinations for pattern generation
          :param partitions: List of partition numbers for pattern features
          :param disc_cols_miss_frxn: Maximum allowed fraction of 'm' (missing) values to pick valid attendance columns
          :param months_for_binary: Months to consider for binary features
          :param absence_thresholds: Thresholds for binary absence features
          """
          self._feature_config = {
               "groups_of_months": groups_of_months,
               "combs_of_chars": combs_of_chars,
               "partitions": partitions,
               "disc_cols_miss_frxn": disc_cols_miss_frxn,
               "months_for_binary": months_for_binary,
               "absence_thresholds": absence_thresholds,
          }

          # Pre-compute pattern features
          self._pattern_strings = self._formulate_pattern_strings(
               self._feature_config["combs_of_chars"]
          )
          self._configured = True

     def _extract_month(self, column_name: str) -> Optional[str]:
          """
          Extracts the month number from an attendance column name using the defined pattern.
          :param column_name: Name of the attendance column
          :return: Month number as string if pattern matches, None otherwise
          """
          match = re.match(constants.Attendance.PATTERN, column_name)
          if match:
               return match.group(1)
          return None

     def _get_valid_attendance_columns(
          self, df: pd.DataFrame, acad_year: str, disc_cols_miss_frxn: float = 1.0
     ) -> tuple[List[str], List[str]]:
          """
          Identifies valid attendance columns by:
          - Matching the pattern 'digit_digit' (e.g., '12_15')
          - Excluding holiday columns for the given academic year
          - Filtering out columns with excessive missing 'm' values
          :param df: Input DataFrame containing attendance columns
          :param acad_year: Academic year identifier
          :return: Tuple of (all_attendance_columns, valid_attendance_columns)
          """
          all_attendances = [col for col in df.columns if self._extract_month(col) is not None]
          holiday_cols = set(self.holidays.get(acad_year, []))

          valid_attendances = [
               col for col in all_attendances
               if col not in holiday_cols
               and df[col].value_counts(normalize=True).get(constants.Attendance.Status.MISSING, 0) < disc_cols_miss_frxn
          ]
           
          return all_attendances, valid_attendances

     def _num_of_patterns(self, s: str, pattern: str) -> int:
          """Counts overlapping occurrences of a pattern in a string."""
          return 0 if not pattern.strip() else len(re.findall(f'(?={re.escape(pattern.strip())})', s.strip()))

     def _formulate_pattern_strings(self, char_combos: List[List]) -> List[str]:
          """
          Generates all string patterns based on max length and character combinations.
          :param char_combos: List of [max_len, characters] format
          :return: Sorted list of pattern strings
          """
          return sorted(
               ''.join(p)
               for max_len, chars in char_combos
               for l in range(1, int(max_len) + 1)
               for p in itertools.product(chars, repeat=l)
          )

     def _scaling_factor(self, L: int, s: str) -> int:
          """
          Calculates scaling factor to normalize pattern frequency.
          :param L: Length of the sequence
          :param s: Pattern string
          :return: Integer scaling factor
          """
          overlap = max((i for i in range(1, len(s)) if s[:i] == s[-i:]), default=0)
          unit = s[overlap:]
          return 1 + (L - len(s)) // len(unit) if unit else L // len(s)

     def _add_pattern_features(self, df: pd.DataFrame, cols: List[str], group_name: str) -> pd.DataFrame:
          """
          Adds normalized pattern frequency features from attendance strings over specified partitions.
          :param df: Input DataFrame
          :param cols: Attendance columns to process
          :param group_name: Group label (e.g. sem1)
          :return: DataFrame with new pattern features
          """
          partitions = self._feature_config['partitions']
          pattern_features = []

          for n in partitions:
               d, r = divmod(len(cols), n)
               for i in range(n):
                    part_cols = cols[i * d: (i + 1) * d + (r if i == n - 1 else 0)]
                    dummy = df[part_cols].agg(''.join, axis=1)
                    L = len(dummy.iloc[0])

                    for feat in self._pattern_strings:
                         counts = dummy.apply(lambda x: self._num_of_patterns(x, feat))
                         scaled = (counts / self._scaling_factor(L, feat)).astype(constants.DtypeCastMap.FLOAT)
                         colname = f"{group_name}_partns_{n}_partn_{i+1}_frac_{feat}"
                         pattern_features.append(pd.DataFrame({colname: scaled}, index=df.index))

          return pd.concat([df, *pattern_features], axis=1) if pattern_features else df

     def _add_last_occurrence_features(self, df: pd.DataFrame, cols: List[str], group_name: str) -> pd.DataFrame:
          """
          Adds features capturing the last occurrence of 'a', 'm', 'p' in attendance string.
          :param df: Input DataFrame
          :param cols: Attendance columns to combine
          :param group_name: Group label
          :return: DataFrame with added last-occurrence features
          """
          dummy = df[cols].agg(''.join, axis=1)
          last_occurrence_data = {
               f"{group_name}_last_{ch}": dummy.apply(lambda x: (x.rfind(ch) + 1) / (len(x) + 1e-10))
               for ch in self.attendance_chars
          }
          return pd.concat([df, pd.DataFrame(last_occurrence_data, index=df.index)], axis=1)

     def _add_binarised_features(self, df: pd.DataFrame, valid_attendances: List[str]) -> pd.DataFrame:
          """
          Adds binary features indicating absence in the last N days.
          :param df: Input DataFrame
          :param valid_attendances: List of valid attendance columns
          :return: Modified DataFrame with binary absence features
          """
          months_for_binary = self._feature_config['months_for_binary']
          absence_thresholds = self._feature_config['absence_thresholds']

          months_str = list(map(str, months_for_binary))
          cols = [col for col in valid_attendances if self._extract_month(col) in months_str]

          if not cols:
               return df  # Return unchanged if no relevant columns

          dummy = df[cols].astype(constants.DtypeCastMap.STR).agg(''.join, axis=1)
          binarised_data = {
               f"binary_absence_{t}": dummy.apply(
                    lambda x: 0.0 if constants.Attendance.Status.PRESENT in x[-t:] else 1.0
               ).astype(constants.DtypeCastMap.FLOAT)
               for t in absence_thresholds
          }

          return pd.concat([df, pd.DataFrame(binarised_data, index=df.index)], axis=1)

     def generate_features(
               self,
               df: pd.DataFrame,
               acad_year: str,
               drop_columns_or_groups: List[str] = None,
               column_groups: Dict[str, List[str]] = None
     ) -> pd.DataFrame:
          """
          Generates all configured features for the given dataframe and academic year.
          :param df: Input DataFrame with attendance data
          :param acad_year: Academic year identifier
          :return: DataFrame with engineered features (original attendance columns removed)
          """
          if not self._configured:
               raise ValueError("Features not configured. Call configure_features() first.")

          # Get valid attendance columns for this dataframe and academic year
          all_attendances, valid_attendances = self._get_valid_attendance_columns(
               df, acad_year, self._feature_config["disc_cols_miss_frxn"]
          )

          if not valid_attendances:
               raise ValueError(f"No valid attendance columns found in the dataframe for academic year {acad_year}.")

          result_df = df.copy()

          # Add pattern and last occurrence features
          groups_of_months = self._feature_config["groups_of_months"]
          for group_name, months in groups_of_months.items():
               months_str = list(map(str, months))
               cols = [col for col in valid_attendances if self._extract_month(col) in months_str]
               if cols:
                    result_df = self._add_pattern_features(result_df, cols, group_name)
                    result_df = self._add_last_occurrence_features(result_df, cols, group_name)

          # Add binary features
          result_df = self._add_binarised_features(result_df, valid_attendances)

          # Drop raw daily attendance columns + other configured drop features
          drop_columns = set(all_attendances).union(
               set(determine_columns_to_drop(result_df, drop_columns_or_groups, column_groups))
          )
          result_df.drop(columns=drop_columns, inplace=True, errors="ignore")

          cat_features, num_features = split_features_by_dtype(result_df, self.index, self.label)
          return result_df, cat_features, num_features