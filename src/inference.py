import os
import pandas as pd
from src.data.preprocess import DataPreprocessor
from src.data.engineer_attendance_features import EngineerAttendanceFeatures
from src.models.model import CatBoostBinaryClassifier
from src.models.utils import get_model_features
from src.analysis.metrics import BinaryModelEvaluator
from src.utils import load_config
from src.data.utils import extract_academic_year_from_path


def inference_pipeline(
          exp_dir: str, inference_data_path: str, manual_thresholds: dict=None
) -> dict:
     """
     Runs the full inference pipeline.

     :param: exp_dir (str): Experiment directory where config and models are saved.
     :param: inference_data_path (str): Path to the inference data file (pickle format).
     :param: manual_thresholds (dict): Optional manual thresholds for evaluation.
     :returns: Dict containing summary metrics.
     """

     # Load config and features from experiment directory
     config = load_config(f"{exp_dir}/config.json")
     cat_features, num_features = get_model_features(dir=exp_dir)

     # Load inference data
     df = pd.read_pickle(inference_data_path)

     # Preprocess
     preprocessor = DataPreprocessor()
     df, column_groups = preprocessor.preprocess(
          df=df,
          column_filters=config.data.column_filters, 
          index=config.data.index, 
          label=config.data.label
     )

     # Initialize and configure feature engineering
     feature_engineer = EngineerAttendanceFeatures(
          holidays_calendar_path=config.data.holidays_calendar_path,
          index=config.data.index, 
          label=config.data.label
     )
     feature_engineer.configure_features(**config.data.engineer_features)
     df, _, _ = feature_engineer.generate_features(
          df,
          acad_year=extract_academic_year_from_path(inference_data_path),
          drop_columns_or_groups=config.data.drop_columns_or_groups,
          column_groups=column_groups
     )

     # Prediction
     model = CatBoostBinaryClassifier(
          exp_dir=exp_dir, 
          cat_features=cat_features, 
          config=config
     )
     result_df = model.predict(
          x=df, 
          features=cat_features + num_features
     )

     # Summary metrics and plots
     ds_name = os.path.splitext(os.path.basename(inference_data_path))[0]
     evaluator = BinaryModelEvaluator(
          df=result_df,
          label_col=config.data.label,
          proba_1_col="preds_proba_1",
          ds_name=ds_name,
          save_dir=exp_dir,
          manual_thresholds=manual_thresholds
     )
     evaluator.plot_all()

     # Save result dataframe
     result_df.to_pickle(f"{exp_dir}/{ds_name}.pkl")

     return evaluator.summary_metrics()