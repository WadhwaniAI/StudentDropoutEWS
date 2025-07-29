import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def get_drift(
          ref_df: pd.DataFrame,
          cur_df: pd.DataFrame,
          ref_name: str,
          cur_name: str,
          features: list,
          save_dir: str,
) -> None:
     """
     Computes column-wise data drift between two datasets using Evidently.

     :param ref_df (pd.DataFrame): Reference dataset (e.g., training data).
     :param cur_df (pd.DataFrame): Current dataset (e.g., inference/test data).
     :param ref_name (str): Name for the reference dataset (used in output filenames).
     :param cur_name (str): Name for the current dataset (used in output filenames).
     :param features (list): List of feature columns to evaluate drift on.
     :param save_dir (str): Directory where drift reports (HTML & Excel) will be saved.

     Returns None. Saves drift report as an HTML file and an Excel summary table.
     """
     report = Report(metrics=[DataDriftPreset()])
     report.run(reference_data=ref_df[features], current_data=cur_df[features])

     path = f"{save_dir}/{ref_name} vs {cur_name}"
     report.save_html(f"{path}.html")
     report.as_dataframe()['DataDriftTable'].to_excel(f"{path}.xlsx", index=False)