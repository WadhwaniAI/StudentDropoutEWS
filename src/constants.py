"""
Constants used across this project.
This file serves as the central source of truth for all constants used throughout the project.
It uses classes as namespaces to provide better organization, autocompletion, and static analysis.
"""

class MetadataPaths:
     """File paths for metadata and configuration files."""
     DATASET_SCHEMA = "metadata/dataset_schema.json"
     CONFIG_SCHEMA = "metadata/config_schema.json"
     HOLIDAYS_CALENDAR = "metadata/holidays_calendar.json"
     ATTENDANCE_REPLACEMENT_MAP = "metadata/attendance_replacement_map.json"

class ModelArtifacts:
     """File names for all model-related artifacts."""
     CONFIG = "config.json"
     MODEL = "model.cbm"
     CALIBRATOR = "isotonic_regression.pkl"
     PRECALIBRATION_SCORES = "val_precalibration_confidence_scores.csv"
     POSTCALIBRATION_SCORES = "val_postcalibration_confidence_scores.csv"
     TRIAL_PARAMS = "trial_params.json"
     CATBOOST_TRAINING = "catboost_training.json"
     LOSS_CURVES = "loss_curves.png"
     CAT_FEATURES = "cat_features.pkl"
     NUM_FEATURES = "num_features.pkl"
     VALIDATION = "val.pkl"
     SUMMARY_METRICS = "summary_metrics.json"
     EXPLAINED_OUTPUT_SUFFIX = "_explained_output"
     LOSS_CURVE_TRAIN_REMAPPED = "trn"
     LOSS_CURVE_VALIDATION_REMAPPED = "val"

class ColumnNames:
     """Standardized column names used across dataframes."""
     INDEX = "studentid"
     LABEL = "target"
     PROBA_0 = "preds_proba_0"
     PROBA_1 = "preds_proba_1"
     PREDICTION = "preds"
     SHAP_PREFIX = "shap"
     PREDICTOR_GROUP_PREFIX = "predictor_group"
     TOP_DRIVER_SUFFIX = "_top_driver"

class ColumnGroups:
    """Standardized names for column groups used in the schema."""
    ALL_ATTENDANCES = "all_attendances"
    EXAM_SCORE_SUBWISE = "exam_score_subwise"
    EXAM_ATTND_SUBWISE = "exam_attnd_subwise"
    MONTH_AGG_ATTND = "month_agg_attnd"
    ATTND = "attnd"
    SCORE = "score"

class DtypeCastMap:
     """Data type casting map for preprocessing."""
     STR = "string"
     INT = "int"
     FLOAT = "float"

class Attendance:
     """Constants related to student attendance data."""
     class Status:
          ABSENT = 'a'
          MISSING = 'm'
          PRESENT = 'p'

     CHARS = [Status.ABSENT, Status.MISSING, Status.PRESENT]
     PATTERN = r"^(\d+)_(\d+)$"

class FeatureEngineering:
     """Default parameters for feature engineering."""
     GROUPS_OF_MONTHS = {"full": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]}
     MONTHS_FOR_BINARY = [12, 1, 2, 3, 4]
     ABSENCE_THRESHOLDS = [10, 15, 20, 30, 40, 50, 60]
     CHAR_COMBINATIONS = [[1, ['a', 'm', 'p']]]
     PARTITIONS = [3]
     MISSING_FRACTION = 1.0

class ModelConfig:
     """Parameters for model training, evaluation, and prediction."""
     CALIBRATION_BINS = 20
     EPSILON = 1e-8
     PREDICTION_LABELS = {0: "notdropout", 1: "dropout"}
     
     class GpuIncompatibleParams:
          """Parameters that are not compatible with GPU training."""
          RANDOM_STRENGTH = "random_strength"
          RSM = "rsm"
          DIFFUSION_TEMPERATURE = "diffusion_temperature"
          SAMPLING_FREQUENCY = "sampling_frequency"
          APPROX_ON_FULL_HISTORY = "approx_on_full_history"
          LANGEVIN = "langevin"

     GPU_INCOMPATIBLE_DEFAULTS = {
          GpuIncompatibleParams.RANDOM_STRENGTH: 1,
          GpuIncompatibleParams.RSM: 1,
          GpuIncompatibleParams.DIFFUSION_TEMPERATURE: 10000,
          GpuIncompatibleParams.SAMPLING_FREQUENCY: "PerTreeLevel",
          GpuIncompatibleParams.APPROX_ON_FULL_HISTORY: False,
          GpuIncompatibleParams.LANGEVIN: False
     }

class PlotConfig:
     """Configurations for generating plots."""
     FIGURE_SIZE = (7, 6)
     DPI = 500
     COLORS = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
     RECALL_K_POINTS = 1000
     CALIBRATION_BINS = 20

     class Suffixes:
          """Suffixes for plot filenames."""
          PRECISION_RECALL = "precision_recall"
          ROC_CURVE = "roc_curve"
          CALIBRATION = "calibration"
          ERROR_DIST = "error_dist"
          PPV_NPV = "ppv_npv_vs_threshold"
          RECALL_AT_K = "recall_at_k"
          PROBA_DIST = "proba_dist"
          PROBA_DIST_WITH_LABELS = "proba_dist_with_labels"
          DROPOUT_RATE = "dropout_rate_vs_threshold"

class SuccessMessages:
     """Standardized success messages for console output."""
     METADATA_VALIDATION = "✅ All DataFrame columns found in schema and column_groups."

class InternalStrings:
    """Internal or reserved strings used in the codebase."""
    SELF = "self"
    NONE = "None"

class FileExtensions:
    """Standardized file extensions used across the project."""
    PICKLE = ".pkl"

class ConfigSchema:
     """Constants for configuration schema validation."""
     TYPE_PLACEHOLDERS = {
          "INT": ["<int>", "<num>"],
          "FLOAT": ["<float>"],
          "BOOL": ["<true|false>", "<bool>"],
          "FLOAT_OR_STR": ["'actual' | float"]
     }
     JSON_INDENT = 5

class SplitNames:
    """Standardized names for data splits."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

class WandB:
    """Constants related to Weights & Biases integration."""
    PROJECT = "project"
    EXCLUDE_KEYS = "exp"

class CliModes:
    """CLI modes for main.py."""
    TRAIN = "train"
    INFER = "infer"
    EXPLAIN = "explain"

class CliArgs:
    """CLI arguments for main.py."""
    MODE = "--mode"
    CONFIG_SOURCE = "--config_source"
    EXP_DIR = "--exp_dir"
    INFERENCE_DATA_PATH = "--inference_data_path"
    DF_PATH = "--df_path"
    PREDICTOR_GROUPS = "--predictor_groups"
    TARGET_RECALL = "--target_recall"
    THRESHOLD = "--threshold"

class DateTime:
    """Constants related to date and time formatting."""
    TIMEZONE = "Asia/Kolkata"
    TIMESTAMP_FORMAT = "%Y-%m-%d_%H:%M:%S"

class SummaryMetricKeys:
    """Keys for the summary_metrics dictionary."""
    BEST_PARAMS = "best_params"
    SHAPE_TRAINING_DATA = "shape_training_data"
    VAL_THRESHOLD_MAX_F1 = "val_threshold_max_f1"
    VAL_THRESHOLD_MAX_LIFT = "val_threshold_max_lift"
    CATEGORICAL_FEATURES = "categorical_features"
    N_CATEGORICAL_FEATURES = "n_categorical_features"
    NUMERICAL_FEATURES = "numerical_features"
    N_NUMERICAL_FEATURES = "n_numerical_features"
    MANUAL_THRESHOLD_MAX_F1 = "max_f1 (val)"
    MANUAL_THRESHOLD_MAX_LIFT = "max_lift (val)"