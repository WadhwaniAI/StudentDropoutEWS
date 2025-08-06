<details>
<summary><span style="font-size: 20px">🗂️ OVERVIEW</span></summary>

---

### Problem
- Many students in India drop out of schools due to diverse social, economic and geographical factors.
- Students enrolled in a given academic year (AY) but **failing to re-enroll** in the *next* AY are dropouts.
- *Education gaps* lead to unskilled labour and are linked to poor health—impeding a nation's development.

### Motivation
- **Dropout indicators** are present in social traits, attendance patterns and performance in assessments.
- This project aims to build an **Early Warning System (EWS)** using *machine learning (ML)* techniques to predict students who are at risk of dropping out of school.
- These predictions could potentially be used to cognize and design interventions to mitigate student dropouts.

### Data Sources
- The project has been developed using the following data that was provided by *Vidya Samiksha Kendra (VSK)—Samagra Shiksha, Department of Education, state of Gujarat, India*.
  - Enrollment data: Delineates a student's regional and socioeconomic factors.
  - Daily attendance data: Delineates a student's daily attendance (present, absent or missing entry).
  - Semester assessment data: Delineates a student's attendance and performance in examinations.
- Data from the three sources is merged into a *unified dataset* with each row representing information pertaining to one student.
- **Customizable**: Anyone with similar data could use this project by suitably modifying the [Dataset Schema](metadata/dataset_schema.json).

### Formulation
- EWS is formulated as a *Binary Classification* ML problem (dropout: label 1, not-dropout: label 0).
- For a given AY, a binary *Target* for each student is derived using the enrollment data of the following AY.
- The **Input** to the pipeline is the unified dataset (with the target column).
- The resulting **Output** is a dataframe that includes the final set of features used in modeling and dropout probabilities for each student.
- [SHAP](https://shap.readthedocs.io/en/latest/) is used to *explain the model's predictions*.

---

</details>

<details>
<summary><span style="font-size: 20px">🔧 SETUP</span></summary>

---

- Clone the repository
```bash
git clone https://github.com/WadhwaniAI/StudentDropoutEWS.git
git checkout main
cd StudentDropoutEWS
```

- Create a virtual environment and install the required packages
```bash
conda create --name ews python==3.12
conda activate ews
pip install -r requirements.txt
```

---

</details>

<details>
<summary><span style="font-size: 20px">🧩 METADATA</span></summary>

---

The [metadata](metadata) directory contains mandatory files that define the schema necessary to use this repository.

[Calendar of holidays](metadata/holidays_calendar.json)
- This is a *mandatory* nested JSON dictionary holding information about holidays in AYs. 
- Example format: `{"2223": {"6": {"sundays": [5, 12, 19, 26], "vacation": [1, 2]}}}`
  - Stores non-working dates for each AY (e.g., "2223" for AY 2022-23) and month (e.g., "6" for June, and "7" for July). 
  - Dates are integers under categories like "sundays", "festive", "vacation", or others (e.g., "pravesh utsav").
- An example of this file for the AYs from 2022-23 to 2024-25 for the state of Gujarat is [here](metadata/holidays_calendar.json).
- Please edit the dictionary within this file for the AYs of your interest.
- This file could either be manually populated from a PDF or parsed from a CSV notified by the administration.

[Dataset Schema](metadata/dataset_schema.json)
- This is a *mandatory* JSON dictionary defining the structure of a usable dataset.
- Each key is a column name and the corresponding value is a list of datatype, description, and grouping.
- Valid datatypes are `str` for categorical columns, `float` for numerical columns, and `int` for target column.
- Description is a piece of text briefly explaining the information the column contains.
- Grouping enables combined use of columns such as in common preprocessing operations.
- Modify [Dataset Schema](metadata/dataset_schema.json) if a dataset has different column names, datatypes, descriptions or groupings.

[Config Schema](metadata/config_schema.json)
- This is a *mandatory* nested JSON dictionary illustrating the valid schema of a `Config` file.
- A new Config (for training) or an existing Config (for inference) must follow this schema.
- In [Config Schema](metadata/config_schema.json), all *optional* parameters are denoted as `<key>` and valid datatypes are in placeholders.
- [Config Schema](metadata/config_schema.json) is **not to be deleted**.
  - A copy of this file needs to be made by the user for their own experiments.
- [Config Schema](metadata/config_schema.json) is elaborated upon in the **Config** section of this README.

[Predictor groups](metadata/predictor_groups.json)
- This is a JSON dictionary categorizing similar features into predictor groups.
- Predictor groups are used to explain a model's predictions and guide interventions.
- Features are manually organized into predictor groups—there is no script to generate them.
- The features in [Predictor groups](metadata/predictor_groups.json) must be a subset of the features used in modeling.
- They are used in [PredictorGroupExplainer](src/explainability/predictor_group_explainer.py). They are not required in the training or inference pipelines.
- Modify [Predictor groups](metadata/predictor_groups.json) for different group explanations.

---

</details>

<details>
<summary><span style="font-size: 20px">📊 DATASET</span></summary>

---

- A valid dataset for training and inference must have a schema *consistent* with [Dataset Schema](metadata/dataset_schema.json). 
  - The columns in a dataset must be a subset of the columns in [Dataset Schema](metadata/dataset_schema.json). 
  - If the names of columns in the dataset are different, please modify [Dataset Schema](metadata/dataset_schema.json) before use.
- The format of an input dataset file must be pickle. Example: `dataset/ay2223_grade3.pkl`. 
  - Currently, support for other file formats is not provided.
- The stem of a dataset file name is important to extract "academic year" and "grade" using regex.
  - It must follow the pattern: `ay<academic_year>_grade<grade>`. Eg: `ay2223_grade3`.

---

</details>

<details>
<summary><span style="font-size: 20px">📘 CONFIG</span></summary>

---

- A new JSON Configuration file is used to define all aspects for training a model.
- An existing JSON configuration file (from a previous experiment) is used to run inference on a new dataset.
- [Config Schema](metadata/config_schema.json) is explained below. Comments explain valid entries: **// datatype: description; example**.
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
               "shuffle": <true|false>                               // bool: Shuffle before splitting into train and val; Eg: true
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
               "<tune>": {                                           // [Optional] Specify only for Hyperparameter tuning
                    "independent": {                                 // Independent hyperparameters
                         "<param_name>": {
                              "dtype": "<int|float|categorical>",    // str: DataType of the hyperparameter; Eg: "float"
                              "tuning_space": {
                                   "low": <num>,                     // int or float (as per dtype): Min val of the tuning space; Eg: 0.01
                                   "high": <num>,                    // int or float (as per dtype): Max val of the tuning space; Eg: 1.0
                                   "step": <optional_int>,           // int: Step size (optional); Eg: 2
                                   "log": <optional_bool>,           // bool: Log scale to use or not?; Eg: true
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

---

</details>

<details>
<summary><span style="font-size: 20px">📉 TRAINING</span></summary>

---

- To train a model, run `main.py` in `train` mode:
```bash
python -m src.main \
     --mode train \
     --config_source path/to/config_or_config_dir
```
```bash
Arguments:
----------
mode (str): Must be set to "train" to activate TRAINING mode.
config_source (str): Path to either a single config file or a directory containing multiple config JSONs.
```
- Training generates the following artifacts in the experiment directory (created using `config.exp.root_exps`):
  - Training and validation dataframes with dropout predictions
  - Metric plots, and
  - JSON file containing loss values over epochs.
- If a directory of JSON configs is provided, experiments run in a loop.

---

</details>

<details>
<summary><span style="font-size: 20px">🎯 INFERENCE</span></summary>

---

- To run inference on a new dataset with a trained model, run `main.py` in `infer` mode:
```bash
python -m src.main \
     --mode infer \
     --exp_dir path/to/exp_dir \
     --inference_data_path path/to/inference_data.pkl
```
```bash
Arguments:
----------
mode (str): Must be set to "infer" to activate INFERENCE mode.
exp_dir (str): Path to a previous experiment directory (to use trained model and config).  
inference_data_path (str): Path to the inference dataset file in pickle format.
```
- Inference generates and saves a dataframe with features and predicted probabilities in exp_dir.

---

</details>

<details>
<summary><span style="font-size: 20px">💡 EXPLAINABILITY</span></summary>

---

- To explain results, run `main.py` in `explain` mode:
```bash
python -m src.main \
    --mode explain \
    --exp_dir path/to/exp_dir \
    --df_path path/to/input_data.pkl \
    --predictor_groups path/to/predictor_groups.json \
    [--threshold 0.6] \
    [--target_recall 0.4]
```
```bash
Arguments:
----------
mode (str): Must be set to "explain" to activate explainability mode.
exp_dir (str): Path to the experiment directory with trained model, config and optional artifacts.
df_path (str): Path to the results dataset (containing prediction columns) to be explained (`.pkl` format).
predictor_groups (str OR Dict[str, List[str]]): Path to the JSON file containing mapping of features to groups OR the loaded dictionary.
threshold (float): (Optional) Manually specify the threshold for binary classification to generate output predictions.
target_recall (float): (Optional) Recall on validation set to compute threshold (if not provided/known)
```
- The resulting output of this pipeline is a dataframe saved in `exp_dir` with SHAP values for each predictor group and top driving feature(s) for each prediction.
---

</details>

<details>
<summary><span style="font-size: 20px">🙏 ACKNOWLEDGEMENTS [wip]</span></summary>

---

- We extend our sincere gratitude to *Vidya Samiksha Kendra (VSK), state of Gujarat, India* for their support. 
- All the original data that was used to build the project has been provided by VSK.
- We would like to thank *UNICEF India* for expert guidance to drive Model Explainability for interventions.

---

</details>