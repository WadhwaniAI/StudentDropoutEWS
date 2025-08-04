<details>
<summary><span style="font-size: 20px">🗂️ Project</span></summary>

- EWS (Early Warning System) is a tabular binary classification problem to identify students at the risk of dropping out.
- A student, identified using a unique Student ID is a
  - Dropout (label 1): If the student ID is enrolled in a given academic year but is absent in the following academic year
  - Not a dropout (label 0): If the student ID is enrolled in both (successive) the academic years.
- We receive the following kinds of data:
  - Enrollment data: Information about students collected during enrollment
  - Attendance data: Daily attendance data collected throughout the academic year
  - Assessment data: Semester-1 Assessment Tests (SAT-1) that capture examination attendance and scores 
- These are combined into a single dataset file for each grade wherein:
  - a row represents a student (identified using a Student ID as index)
  - columns are sourced from the enrollment data, attendance data, assessment data, or are engineered features
- CatBoost is used to build six prediction models, one for each grade (3 to 8) using the above generated dataset files.
  - The input is a set of categorical and numerical features obtained from given datasets
  - The output is probability scores indicative of the risk of a student dropping out
- Results shared include the prediction class and contributions of predictor groups and features to guide interventions.

</details>

<details>
<summary><span style="font-size: 20px">🧠 Index</span></summary>

- **Setup**  
  Learn how to clone the repository, create a virtual environment, and install required packages.

- **Metadata**  
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
<summary><span style="font-size: 20px">🔧 Setup</span></summary>

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
<summary><span style="font-size: 20px">🧩 Metadata</span></summary>

The ```metadata/``` directory contains mandatory auxiliary data aspects needed to train models, run inference, and obtain predictors.

```metadata/holidays_calendar.json```
- Example: `{"2223": {"6": {"sundays": [5, 12], "vacation": [1, 2], "pravesh utsav": [13, 14]}}}`
- This nested dictionary stores non-working day metadata for each academic year (e.g., "2223" representing academic year 2022-23), and for each month within the year ("6", "7" representing June and July.) 
- It maps to sub-categories like "sundays", "festive", "vacation", or custom labels (e.g., "pravesh utsav"), listing relevant dates as integers.
- An example of this file for the academic years from 2022-23 to 2024-25 for the state of Gujarat is [here](metadata/holidays_calendar.json)
- Please edit the dictionary within this file for the academic years of your interest.

```metadata/schema.json```
- This dictionary represents the schema for a dataset.
- Each valid column name is a keys and value is a list of appropriate datatype and description.

```metadata/column_groups.json```
- This dictionary groups columns for combined use such as common preprocessing operations.

```metadata/predictor_groups.json```
- This dictionary enlists predictor groups used to explain predictions and guide interventions using SHAP.
</details>

<details>
<summary><span style="font-size: 20px">📊 Dataset</span></summary>

- A valid dataset for training and inference must have a schema consistent with [dataset schema](metadata/dataset_schema.json). 
- The columns in a usable dataset must be a subset of the columns in [dataset schema](metadata/dataset_schema.json). 
- If the columns in the dataset have different names, please modify [dataset schema](metadata/dataset_schema.json) before use.
- The format of a dataset (dataframe) file must be pickle. Example: `dataset/ay2223_grade3.pkl`. 
- Currently, support for other file formats is not provided.
- The basename of a dataset file is important to extract "academic year" and "grade" using regex.
- The basename must follow the pattern: `ay<academic_year>_grade<grade>.pkl`. Eg: `ay2223_grade3.pkl`.

</details>

<details>
<summary><span style="font-size: 20px">📘 Config</span></summary>

- A new JSON Configuration file is used to define all aspects for training a model.
- An existing JSON configuration file (from a previous experiment) is used to run inference on a new dataset.
- `Config Schema` is shown below. Comments explain valid entries: **// datatype: description; example**.

---

```javascript
{
     "exp": {
          "title": "<experiment_title>",                             // str: Descriptive name for the experiment; Eg: "baseline_grade3"
          "project": "<project_name>",                               // str: Project name on W&B for logging; Eg: "ews"
          "root_exps": "<path_to_experiment_outputs>"                // str: Directory to save all experiment outputs; Eg: "exps/baseline/grade3"
     },
     "data": {
          "training_data_path": "<path_to_training_data>",           // str: Pickle or CSV path of training data; Eg: "datasets/ay2223_grade3.pkl"
          "index": "<unique_id_column>",                             // str: Unique ID column; Eg: "aadhaaruid"
          "label": "<target_column>",                                // str: Target label column name; Eg: "target"
          "holidays_calendar_path": "<path_to_holidays_calendar>",   // str: JSON with academic holidays metadata; Eg: "metadata/holidays_calendar.json"
          "column_filters": {                                        
               "in": { "<col>": ["<val1>", "<val2>"] },              // dict[str, list[str]]: Include rows where column values are in list; Eg: { "schcat": ["1", "2"] }
               "notin": { "<col>": ["<val1>", "<val2>"] }            // dict[str, list[str]]: Exclude rows where column values are in list; Eg: { "schmgt": ["92", "93"] }
          },
          "sample": {
               "p": "<'actual' | float>",                            // str or float: Sampling ratio or 'actual' to keep original; Eg: 0.5 or "actual"
               "seed": <int>                                         // int: Random seed for reproducibility; Eg: 5
          },
          "split": {
               "train_size": <float>,                                // float: Train split ratio; Eg: 0.7
               "random_state": <int>,                                // int: Random seed for split; Eg: 42
               "shuffle": <true|false>                               // boolean: Shuffle before splitting into train and val; Eg: true
          },
          "engineer_features": {
               "groups_of_months": { "<group>": [<months>] },        // dict[str, list[int]]: Month groupings; Eg: { "full": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4] }
               "combs_of_chars": [[<max len>, ['a','m','p']]],       // list[list[int, list[str]]]: Max length of permutation, subset of ("a", "m", "p") to use; Eg: [[1, ["m", "p", "a"]]]
               "partitions": [<int>],                                // int: Number of partitions to split each month group; Eg: [3]
               "disc_cols_miss_frxn": <float>,                       // float: Permitted max limit of fraction of missing attendance entries; Eg: 0.9
               "months_for_binary": [<months>],                      // list[int]: Months used for binary features; Eg: [6, 7, 8, 9, 10]
               "absence_thresholds": [<ints>]                        // list[int]: Thresholds (days of continuous absenteeism) to define binary absence. Eg: [10, 15, 30]
          },
          "drop_columns_or_groups": [
               "<col_or_group1>", "<col_or_group2>"                  // list[str]: Drop any columns or groups; Eg: ["schoolid", "[full][#partns=3][partn_3, frac_p], "exam_attnd_subwise"]
          ]
     },
     "model": {                                                      
          "n_trials": <int>,                                         // int: Number of hyperparameter tuning trials; Eg: 50
          "calibration_nbins": <int>,                                // int: Bins for probability calibration; Eg: 20
          "params": {                                                
               "fixed": {                                            // Fixed parameters (Are not tuned); Mandatory
                    "loss_function": "Logloss",                      // str: Objective function; Eg: "Logloss"
                    "random_seed": <int>,                            // int: Seed for model reproducibility; Eg: 0
                    "task_type": "<CPU|GPU>",                        // str: Hardware to use; Eg: "CPU"
                    "devices": "<GPU_ids>",                          // str: GPU ID device string (optional); Eg: "0", "0,1"
                    "auto_class_weights": "<a valid value>"          // str: Class imbalance handling; Eg: "Balanced"
               },
               "tune": {                                             // Specify only for Hyperparameter tuning
                    "independent": {                                 // Independent hyperparameters
                         "<param_name>": {
                              "dtype": "<int|float|categorical>",    // str: DataType of the hyperparameter; Eg: "float"
                              "tuning_space": {
                                   "low": <num>,                     // int or float (as per dtype): Min val of the tuning space; Eg: 0.01
                                   "high": <num>,                    // int or float (as per dtype): Max val of the tuning space; Eg: 1.0
                                   "step": <optional_int>,           // int: Step size (optional); Eg: 2
                                   "log": <optional_bool>,           // boolean: Log scale to use or not?; Eg: true
                                   "choices": ["<cat1>", "<cat2>"]   // list[str]: Categories (if categorical); Eg: ["Ordered", "Plain"]
                              }
                         }
                    },
                    "dependent": {                                   // Dependent hyperparameters
                         "<param_name>": {
                              "dependent_on_param": "<other_param>", // str: Param this depends on; Eg: "grow_policy"
                              "dependent_on_value": ["<trig_val>"],  // list[str]: Values that trigger it; Eg: ["Depthwise"]
                              "dtype": "<int|float>",                // str: DataType of the hyperparameter; Eg: "int"
                              "tuning_space": {
                                   "low": <num>,                     // int or float (as per dtype): Min val of the tuning space; Eg: 0.1
                                   "high": <num>                     // int or float (as per dtype): Max val of the tuning space; Eg: 10
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
<summary><span style="font-size: 20px">📉 Training</span></summary>

- To train a model, execute `main.py` using `train` mode as illustrated below.
- All artifacts are saved in the created experiment directory (created using `config.exp.root_exps`).
- If a directory of JSON configs is provided, experiments run in a loop.

```
python -m src.main --mode train --config_source <path/to/config> 

Arguments:
----------
mode (str): 'train'
config_source (str): Path to config JSON file or directory of JSON configs. 
```

</details>

<details>
<summary><span style="font-size: 20px">🎯 Inference</span></summary>

- To run inference on a new dataset, execute `main.py` using `infer` mode as illustrated below.
- Output dataframe with features and predicted probabilities is saved in `exp_dir`.

```
python -m src.main --mode infer --exp_dir <path/to/exp_dir> --inference_data_path <path/to/inference_data> 

Arguments:
----------
mode (str): 'infer'
exp_dir (str): Path to the experiment directory (to use model and other optional artifacts).
inference_data_path (str): Path to the inference data file.
```

</details>

<details>
<summary><span style="font-size: 20px">💡 Explainability</span></summary>

- The `SHAPPipeline` explains model predictions using [SHAP](https://shap.readthedocs.io/en/latest/) scores.
- Features present in `df_path` are manually grouped into [predictor groups](metadata/predictor_groups.json) to combine contributions.
- Output dataframe with columns pertaining to predictor groups and top driving factors is saved in `exp_dir`.

```
from explainability.shap_pipeline import SHAPPipeline
shap_pipeline = SHAPPipeline(
     exp_dir=path/to/exp/dir,                          // str: path to experiment directory (to use model and optional artifacts)
     df_path=path/to/df_with_predictions",             // str: path to dataframe containing prediction columns
     predictor_groups=path/to/predictor_groups.json,   // str: path to JSON defining groupings of features
     threshold=0.4,                                    // float (optional): Threshold to generate prediction class column
     target_recall=0.6                                 // float (optional): Recall on val set to compute threshold (if not provided/known)
)
df_explained = shap_pipeline.run()
df_explained[["predictor_group_1", "predictor_group_1_top_driver"]].head()
```

</details>