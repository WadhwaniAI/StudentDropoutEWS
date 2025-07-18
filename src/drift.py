import pandas as pd

from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
from evidently.report import Report



def get_drift(
          ref_df: pd.DataFrame,
          cur_df: pd.DataFrame,
          ref_name: str,
          cur_name: str,
          features: list,
          save_dir: str,
) -> None:
     
     '''
     Description:
          Computes the drift between two dataframes.
     Args:
          ref_df: reference dataframe.
          cur_df: current dataframe.
          ref_name: name of reference dataset.
          cur_name: name of current dataset.
          features: features to compute drift on.
          save_dir: directory to store the drift results.
     Returns:
          None. Saves the drift between the two dataframes in html and spreadsheet formats.
     '''

     # initialise report
     # Report represents an object to present in a browser (html presentation) and metrics shows what would be depicted. 
     # Here, we would show drift between the two datasets, column-wise.
     report = Report(metrics=[DataDriftPreset()])

     # compute drift
     report.run(reference_data=ref_df[features], current_data=cur_df[features])
     
     # name of file to save
     save_path = f"{save_dir}/{ref_name} vs {cur_name}"

     # save html version
     report.save_html(f"{save_path}.html")

     # save spreadsheet version
     report.as_dataframe()['DataDriftTable'].to_excel(f"{save_path}.xlsx", index=False)    