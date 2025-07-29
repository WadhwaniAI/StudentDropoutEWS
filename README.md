<details>
<summary><h1>🗂️ Project</h1></summary>

- EWS (Early Warning System) is a tabular binary classification problem to identify students at the risk of dropping out.
- We receive the following kinds of data:
  - Enrollment data: Information about students collected during enrollment
  - Attendance data: Daily attendance data collected throughout the academic year
  - Assessment data: Semester-1 Assessment Tests (SAT-1) that capture examination attendance and scores 
- These are combined into a single dataset file per grade wherein:
  - a row represents a student (identified using a PII ID as index)
  - a column comes from either the enrollment data, attendance data or assessment data.
- CatBoost is used to model the datasets.
- Separate models are trained for each grade (grades 3 to 8)
- Results shared include the prediction class and contributions of predictor groups and features to guide interventions.
</details>

<details>
<summary><h1>🧠 Index</h1></summary>

- **Setup instructions**  
  Learn how to clone the repository, create a virtual environment, and install required packages.

- **Important metadata**  
  Covers mandatory metadata such as calendar of holidays, dataset schema, column groups and predictor groups

- **Mandatory aspects of a dataset file**  
  Details the columns, formats, and naming conventions expected in input data files.

- **Config template**  
  Explains how the JSON configuration defines experiment parameters, datasets, and model settings.

- **Training a model**  
  Shows how to train a model based on a given Config and experiment (logging) directory.

- **Inference**  
  Shows how to generate predictions on new data using trained models and update config files as needed.

- **Executing main.py**  
  Understand how to run model training using the CLI, configure paths and options, evaluate over test datasets and compute drifts.

- **Explainability**  
  Shows how to generate predictor groups and their top driving features using SHAP scores for each student.
</details>

<details>
<summary><h1>🔧 Setup instructions</h1></summary>

- Clone the repository
```
$ git clone https://github.com/WadhwaniAI/StudentDropoutEWS.git
$ git checkout main
$ cd StudentDropoutEWS
```
- Create a virtual environment and install the required packages
```
$ conda create --name venv python==3.12
$ conda activate venv
$ pip install -r requirements.txt
```
</details>

<details>
<summary><h1>📊 Important metadata</h1></summary>

The ```metadata/``` directory contains mandatory auxiliary data aspects needed to train models, run inference, and obtain predictors.

```metadata/holidays_calendar.json```
- This nested dictionary stores non-working day metadata for each academic year (e.g., "2223"), and each month ("6", "7", etc.) 
- It maps to categories like "sundays", "festive", "vacation", or custom labels (e.g., "pravesh utsav"), listing relevant day integers.
- Example: `{"2223": {"6": {"sundays": [5, 12], "vacation": [1, 2], "pravesh utsav": [13, 14]}}}`
- For subsequent academic years, this dictionary must be updated.

```data/schema.json```
- This is the universal dictionary with column names as keys and values as a list of appropriate datatype and description.
- It is preferred that column names aren't modified.
  - Modifying them would require modifying certain hardcoded aspects of scripts and configs as well.

```data/column_groups.json```
- This dictionary groups columns for combined use such as common preprocessing operations.

```data/predictor_groups.json```
- This dictionary enlists predictor groups used to explain predictions and guide interventions using SHAP.
</details>

<details>
<summary><h1>🧩 Mandatory aspects of a dataset file</h1></summary>

**Schema:**
A dataframe to use in training and inference pipelines must have a schema consistent with `data/schema.json`.

**File type:**
- A dataset (dataframe) file must be of pickle type. Example: `dataset/ay2223_grade3.pkl`

**Naming format:**
- The basename of any dataset file must follow the pattern: `ay<academic_year>_grade<grade>.pkl`. Example: `ay2223_grade3.pkl`
</details>

<details>
<summary><h1>📘 Config template</h1></summary>

- A JSON Configuration file is used to define all aspects for running an experiment. Example template is shown below.

---

```javascript
{
     "exp": {
          "title": "<experiment_title>",                             // Descriptive name for the experiment. Eg: "baseline - grade3"
          "project": "<project_name>",                               // Project grouping identifier. Eg: "ews"
          "root_exps": "<path_to_experiment_outputs>"                // Directory to save all experiment outputs. Eg: "exps/baseline/grade3"
     },
     "data": {
          "training_data_path": "<path_to_training_data>",           // Pickle or CSV path for training data. Eg: "datasets/ay2223_grade3.pkl"
          "inference_data_path": "<path_to_inference_data>",         // Path for data to infer on (optional). Eg: "datasets/ay2324_grade3.pkl"
          "index": "<unique_id_column>",                             // Unique ID column. Eg: "aadhaaruid"
          "label": "<target_column>",                                // Target label column name. Eg: "target"
          "holidays_calendar_path": "<path_to_holidays_calendar>",   // JSON with academic holidays metadata. Eg: "metadata/holidays_calendar.json"
          "column_filters": {
               "in": { "<col>": ["<val1>", "<val2>"] },              // Include rows where column values are in list. Eg: { "schcat": ["1", "2"] }
               "notin": { "<col>": ["<val1>", "<val2>"] }            // Exclude rows where column values are in list. Eg: { "schmgt": ["92", "93"] }
          },
          "sample": {
               "p": "<'actual' | float>",                            // Sampling ratio or 'actual' to keep original. Eg: 0.5 or "actual"
               "seed": <int>                                         // Random seed for reproducibility. Eg: 5
          },
          "split": {
               "train_size": <float>,                                // Train split ratio. Eg: 0.7
               "random_state": <int>,                                // Random seed for split. Eg: 42
               "shuffle": <true|false>                               // Shuffle before splitting. Eg: true
          },
          "engineer_features": {
               "groups_of_months": { "<group>": [<months>] },        // Month groupings. Eg: { "full": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4] }
               "combs_of_chars": [[<partn>, ["m", "p", "a"]]],       // Attendance char combinations per partition. Eg: [[1, ["m", "p", "a"]]]
               "partitions": [<int>],                                // No. of time partitions. Eg: [3]
               "disc_cols_miss_frxn": <float>,                       // Missingness threshold for discretization. Eg: 0.5
               "months_for_binary": [<months>],                      // Months used for binary features. Eg: [6, 7, 8, 9, 10]
               "absence_thresholds": [<ints>]                        // Thresholds to define binary absence. Eg: [10, 15, 30]
          },
          "drop_columns_or_groups": [
               "<col_or_group1>", "<col_or_group2>"                  // Drop any columns or groups. Eg: "schoolid", "[full][#partns=3][partn_3, frac_p]"
          ]
     },
     "model": {
          "n_trials": <int>,                                         // No. of hyperparameter tuning trials. Eg: 50
          "calibration_nbins": <int>,                                // Bins for probability calibration. Eg: 20
          "params": {
               "fixed": {                                            // Fixed parameters (Not to be tuned)
                    "loss_function": "Logloss",                      // Objective function. Eg: "Logloss"
                    "random_seed": <int>,                            // Seed for model reproducibility. Eg: 0
                    "task_type": "<CPU|GPU>",                        // Hardware to use. Eg: "CPU"
                    "devices": "<GPU_ids>",                          // GPU device string (optional). Eg: "0", "0,1"
                    "auto_class_weights": "<a valid value>"          // Class imbalance handling. Eg: "Balanced"
               },
               "tune": {                                             // Hyperparameter tuning
                    "independent": {                                 // Independent hyperparameters
                         "<param_name>": {
                              "dtype": "<int|float|categorical>",     // Type of the hyperparameter. Eg: "float"
                              "tuning_space": {
                                   "low": <num>,                      // Lower bound. Eg: 0.01
                                   "high": <num>,                     // Upper bound. Eg: 1.0
                                   "step": <optional_int>,            // Step size (optional). Eg: 1
                                   "log": <optional_bool>,            // Log scale? Eg: true
                                   "choices": ["<cat1>", "<cat2>"]    // Categories (if categorical). Eg: ["Ordered", "Plain"]
                              }
                         }
                    },
                    "dependent": {                                    // Dependent hyperparameters
                         "<param_name>": {
                              "dependent_on_param": "<other_param>",  // Param this depends on. Eg: "grow_policy"
                              "dependent_on_value": ["<trig_val>"],   // Values that trigger it. Eg: ["Depthwise"]
                              "dtype": "<int|float>",
                              "tuning_space": {
                                   "low": <num>,                      // Eg: 3
                                   "high": <num>                      // Eg: 10
                              }
                         }
                    }
               }
          }
     }
}
```
</details>

<details>
<summary><h1>🎯 Training a model</h1></summary>

The `training_pipeline` trains a CatBoost model using the given config and saves outputs to the specified experiment directory.

```
from training import training_pipeline
training_summary, metrics_summary = training_pipeline(
     config=config dictionary,
     exp_dir=path/to/exp/dir
)
```
</details>

<details>
<summary><h1>🎯 Inference</h1></summary>

The ```inference_pipeline``` performs prediction on new data using the trained CatBoost model from a given experiment directory. 
It returns the input dataframe with predicted probabilities and binary labels (based on either learned or manual thresholds).

```
from inference import inference_pipeline
results = inference_pipeline(
     exp_dir=path/to/exp/dir,
     inference_data_path="datasets/test_set.pkl",
     manual_thresholds={"test": 0.75} 
)
# Access outputs
probas = results["preds_proba_1"]     # Series of predicted probabilities
labels = results["predictions"]       # Series of "dropout" / "notdropout" labels
```
</details>

<details>
<summary><h1>📉 Executing main.py</h1></summary>

This run the training (and optionally inference) pipelines for given JSON configs. Example: 

```
python main.py --config_source

Arguments:
----------
config_source: Path to config JSON file or directory containing JSON configs.
```
</details>

<details>
<summary><h1>💡 Explainability</h1></summary>

The `SHAPPipeline` explains model predictions using SHAP values by grouping feature contributions and identifying the top predictor groups and drivers behind each prediction.

```
from explainability.shap_pipeline import SHAPPipeline
shap_pipeline = SHAPPipeline(
     exp_dir=path/to/exp/dir,
     df_path=path/to/df_with_predictions",
     predictor_groups=path/to/predictor_groups.json,
     target_recall=0.4,
     target_ds_name="test"
)
df_explained = shap_pipeline.run()
df_explained[["predictor_group_1", "predictor_group_1_top_driver"]].head()
```
</details>