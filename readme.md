
# 🗂️ Project
<details>
<summary>[click to expand]</summary>

- EWS (Early Warning System) is a tabular binary classification problem.
- Datasets used are the following:
  - Enrollment datasets: Information about students collected during enrollment
  - Attendance datasets: Daily attendance data collected throughout the academic year
  - Assessment datasets: Semester-1 Assessment Tests (SAT-1) that capture examination attendance and scores 
- Modeling is done for students in six grades (3 to 8).
- CatBoost is used to model the datasets.
- Results are presented by sharing the prediction class and features contributing to that prediction for each student.
- Highlighting the contributing features is an attempt to guide interventions.
</details>

# 🧠 Index
<details>
<summary>[click to expand]</summary>

- **Setup instructions**  
  Learn how to clone the repository, create a virtual environment, and install required packages.

- **Executing main.py**  
  Understand how to run model training using the CLI, configure paths and options, evaluate over test datasets and compute drifts.

- **Structure of a config file**  
  Explains how the JSON configuration defines experiment parameters, datasets, and model settings.

- **Structure of a dataset file**  
  Details the columns, formats, and naming conventions expected in input data files.

- **Supporting data**  
  Covers auxiliary files like holidays data, year-wise columns to drop lists, students options, and a sample data dictionary.

- **Inference**  
  Shows how to generate predictions on new data using trained models and update config files as needed.

- **Generating predictors**  
  Shows how to generate predictor groups and their top driving features using SHAP scores.


</details>

# 🔧 Setup instructions
<details>
<summary>[click to expand]</summary>

- Clone the repository
- Create a virtual environment
- Install the required packages using ```pip install -r requirements.txt```
</details>

# 📉 Executing main.py
<details>
<summary>[click to expand]</summary>

```
python train.py --config_path <> --ext <> --students_options <>

Arguments:
----------
config_path: Is either the path to a directory containing configs or a single file path to one specific config file.
ext: Extension of the config files, valid extensions are json and toml.
students_options: A pickle file containing a dictionary of description as key and list of unique student IDs as values. 
     [This is not required for training. It is used to compute baseline metrics and compare them with those obtained by trained models. If this file is not specified or does not exist, the evaluation plots such as precision-recall and recall@k will not have baselines.]
```

**Model Calibration**
- Model calibration is done using IsotonicRegression (IR) which is embedded in the fit method of the CatBoostBinaryClassifier class.
- As IR is currently the most suitable technique for this project, a separate class with options has not been defined for Calibration.
</details>

# 📘 Structure of a config file
<details>
<summary>[click to expand]</summary>

- A configuration file is used to define all parameters for running an experiment.
- Below is a structured explanation of each section of a config file. 
- A line in the Config is explained with a comment before it.

---

```javascript
{
     // metadata of experimental details
     "exp": {

          // Human-readable title of the experiment
          "title": "public schools [fullay, attendance labels, sat, drift>0.5], grade 3",

          // Name of the project folder to log results onto W&B
          "project": "ews",

          // Base path for all input datasets
          "root_data": "data",

          // Base path where experiment outputs will be saved
          "root_exps": "exps/public schools [fullay, attendance labels, sat, drift]/grade 3"
     },

     // space to add notes if any
     "notes": {

          // Free-text description of the experiment
          "description": "public schools [fullay, attendance labels]"          
     },

     // all data aspects
     "data": {

          // Grade of the dataset [3 to 8]
          "grade": 3,

          // Path to training data file(.pkl). 
          // The file name must be with a ' | ' separator. The last term, that is, "train[fullsat]" is used to log metrics.
          // The last term (obtained using .split(' | ') is also referred to as the ds_name (dataset_name).
          "trn_file": "data/grade 3/2223 | grade 3 | master | sem1sem2 | train[fullsat].pkl",

          // List of eval or test files. The file names here must follow the same format as mentioned above for the trn_file.
          "tst_file": ["data/grade 3/2324 | grade 3 | master | sem1sem2 | srv-upd[fullsat].pkl"],

          // A test file from tst_files on which metrics would be reported.
          "chosen_tst_file": "",

          // Unique identifier column
          "index": "aadhaaruid",

          // Name of target column used as label
          "label": "target",

          // Path to json file enlisting holidays of an academic year, with key as AY and value as list of dates (month_day).
          "holidays": "data/cols_to_exclude.json",

          // List of names of datasets to save post evaluation.
          "datasets_to_save": ["trn", "val", "srv-upd[fullsat]"],
          
          // Some processing operations [for now, only dropping not-required features]
          "processing": {
               // List of features or feature_groups to drop
               "drop_feature_groups": ["month_agg_attnd", "schoolid", "villageid", "class", "studentage", "freetransportfacility", "academicyear", "[full][#partns=3][partn_3, frac_m]", "[full][#partns=3][partn_3, frac_p]", "[full][last_m]"]
          },

          // sampling parameters
          "sample": {

               // Prevalence to use in training file. Could either be a string "actual" or a float in [0, 1].
               "p": "actual",
               // Random seed for sampling
               "seed": 5
          },

          // train-val Splitting parameters
          "split": {

               // Fraction of samples to use for training
               "train_size": 0.7,
               // Random seed for splitting
               "random_state": 42,
               // Whether to shuffle the data before splitting
               "shuffle": true
          },

          // Transformations to implement on numerical columns.
          // Dictionary with key as name of transform and
          "post_split_transforms": {  },

          // Encodings to implement on categorical columns.
          "post_split_encodings": {  },

          // Parameters to engineer features using daily attendance entries
          "feature_engineering": {

               // Dictionary with list of months as values and nomenclature for month groups as keys.
               // Here, "full" refers to the entire academic year.
               // Instead of "full", could also use "sem1": [6, 7, 8, ..., 11] or "sem2": [12, 1, .., 4].
               "groups_of_months": {
                    "full": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
               },

               // To discard those columns with overall fraction of missing daily attendance entries equal 1.0. [basically, all entries are "m"]
               "disc_cols_miss_frxn": 1.0,

               // Length of the combination and characters to use to compute features.
               // For example, the following default initialisation would generate "aa", "am", "ap", "ma", "mm", "mp", "pa", "pm", "pp" features.
               "combs_of_chars": [[2, ["m", "p", "a"]]],

               // Partion an academic year (or the school working oeriod) into sections to compute features.
               // For example, if the academic year consists of 240 days, 3 partitions would provide 80 days each. 
               // We could carry out this process for multiple partitions hence we use a list.
               "partitions": [3],

               // Boolean to include last occurence of a|m|p as features or not. 
               // If number of days in the section of school working period is 30 and the last "a" occures on 25th day, "last_a" would be 25/30.
               "last_a_m_p_features": true,

               // To drop columns showing daily attendances or not.
               // In general, once we have used the daily attendances to generate features, not including them in modeling is best.
               "drop_all_attendances": true
          },
          
          // Label engineering columns are also used as features.
          "label_engineering": {          

               // Names of the datasets to implement label engineering on.
               "implement_on": ["train[fullsat]", "srv-upd[fullsat]"],

               // The name of task to use. This is default. Since the Config was also used for Regression, we specify.
               "task": "altered classification",

               // Months of attendances to use to compute features.
               "months": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4],

               // POAs (periods of absence) to use to compute features.
               "poas": [10, 15, 20, 30, 40, 50, 60],

               // To discard those columns with fraction of missing values equal 1.0.
               "disc_cols_miss_frxn": 1.0,

               // To drop columns showing daily attendances or not.
               "drop_attnd_label": false
          },

          // Filters to apply on categorical columns.
          "filter": {
               
               // Dictionary with keys as column names and value as list of entries to match in each row.
               "in": {
                    "schcat": ["1", "2", "3", "4", "5", "6", "7"]
               },

               // Negation of filters to apply on categorical columns.
               // Dictionary with keys as column names and value as list of entries to un-match in each row.
               "notin": {
                    "schmgt": ["5", "92", "93", "94", "95", "97", "101"]
               }
          },

          // For visualisation purposes.
          "baseline_params": {

               // POAs (periods of absence) to use to generate pseudo-label columns and compute Precision and recall.
               "poas": [10, 15, 20, 30, 40, 50, 60],

               // POCs refers to periods of consideration to apply POAs. Valid POCs are "full", "sem1" and "sem2".
               "pocs": ["full"],

               // Markers to use on Precision vs Recall plots.
               "markers": ["v", "^", "<", ">", "s", "P", "X"]
          }
     },

     // all model aspects
     "model": {

          // Number of trials to tune hyperparameters using Optuna.
          "n_trials": 50,

          // Number of calibration bins to use for calibration.
          "calibration_nbins": 20,

          // all params to use in CatBoost
          "params": {
               
               // Fixed parameters. We don't specify a tuning space for these.
               "fixed": {
                    "loss_function": "Logloss",
                    "random_seed": 42,
                    "task_type": "GPU",
                    "devices": "1",
                    "auto_class_weights": "Balanced"
               },

               // Parameters to be tuned.
               "tune": {

                    // Dictionary to hold independent parameters.
                    "independent": {

                    // Following parameters don't depend on the presence of other parameters, thus are under the "independent" key.
                         "iterations": {
                              "dtype": "int",
                              "tuning_space": {"low": 500, "high": 3000, "step": 100}
                         },
                         "depth": {
                              "dtype": "int",
                              "tuning_space": {"low": 4, "high": 10}
                         },
                         "leaf_estimation_iterations": {
                              "dtype": "int",
                              "tuning_space": {"low": 5, "high": 20}
                         },
                         "learning_rate": {
                              "dtype": "float",
                              "tuning_space": {"low": 0.0001, "high": 0.01, "log": true}
                         },
                         "l2_leaf_reg": {
                              "dtype": "float",
                              "tuning_space": {"low": 1.0, "high": 5.0}
                         },
                         "bootstrap_type": {
                              "dtype": "categorical",
                              "tuning_space": {"choices": ["Bayesian", "Bernoulli", "MVS"]}
                         },
                         "grow_policy": {
                              "dtype": "categorical",
                              "tuning_space": {"choices": ["SymmetricTree"]}
                         },
                         "leaf_estimation_method": {
                              "dtype": "categorical",
                              "tuning_space": {"choices": ["Newton", "Gradient"]}
                         },
                         "boosting_type": {
                              "dtype": "categorical",
                              "tuning_space": {"choices": ["Ordered"]}
                         },
                         "score_function": {
                              "dtype": "categorical",
                              "tuning_space": {"choices": ["Cosine"]}
                         }
                    },
                    
                    // Dictionary to hold dependent parameters.
                    "dependent": {

                    // Following parameters depend on the presence of other parameters, thus are under the "dependent" key.
                         "bagging_temperature": {
                              "dependent_on_param": "bootstrap_type",
                              "dependent_on_value": ["Bayesian"],
                              "dtype": "float",
                              "tuning_space": {"low": 0, "high": 10}
                         },
                         "subsample": {
                              "dependent_on_param": "bootstrap_type",
                              "dependent_on_value": ["Bernoulli"],
                              "dtype": "float",
                              "tuning_space": {"low": 0.1, "high": 1}
                         }
                    }
               }
          }
     }
}
```
</details>

# 🧩 Structure of a dataset file
<details>
<summary>[click to expand]</summary>

**Base dataset**
As highlighted in the Project section of this readme, we receive the following datasets:
  - Enrollment datasets: Information about students collected during enrollment
  - Attendance datasets: Daily attendance data collected throughout the academic year
  - Assessment datasets: Semester-1 Assessment Tests (SAT-1) that capture examination attendance and scores 
These datasets are combined into a single dataset file per grade wherein:
- each row represents a student (identified using a PII ID as index)
- each column is a raw feature coming either from enrollment datasets, attendance datasets or assessment datasets
Such a single combined file per grade is referred to as the base dataset file of that grade. 

**The columns in a base dataset file must be as follows:**
- Unique ID column
- Categorical columns of enrolment data
- Categorical columns of daily attendance data, in the month_day format. For example, "6_15", refers to 15th June.
- The daily attendance columns must be for months from June to April, days 1 to 31.
- Categorical columns of semester-1 SAT data
- Numerical columns of semester-1 SAT data
- Target column [0 for majority class and 1 for minority class]
- A sample data dictionary showing the columns and what they mean is illustrated in ```data/sample_data_dictionary.json```.

**File format:**
- Must be either excel, csv or pickle (preferred).

**Naming conventions:**
- The dataset files must be names with the separator ``` | ```. 
- The name of the dataset is obtained using ```ds_name = filepath.split(' | ')```
- "Train"
  - 
</details>

# 📊 Supporting Data
<details>
<summary>[click to expand]</summary>

The ```data/``` folder contains hardcoded data aspects needed to run training experiments.
**Please note these need to be appropriately modified or updated based on the training, evaluation and production datasets.**

```data/holidays.json```
- This enlists school holidays as per the school academic calendar shared by the state of Gujarat.
- Currently, the academic calendar years are 2022-23, 2023-24 and 2024-25, and are written as '2223', '2324' and '2425'. 
- Months are written as '6' for June, '7' for July and so on.
- For the purposes of this project, academic years commence in June and terminate in April in the next calendar year.
- In this file, for each year and month, non-school working days are listed for exclusion while computing features.
- If new academic years are included, this file needs to be updated accordingly.

```data/cols_to_exclude.json```
- Using ```data/holidays.json``` as explicated in the previous section, the ```cols_to_exclude.json``` [=>columns to exclude] file is generated.
  - This can be done using ```generate_cols_to_exclude(holidays_filepath: <path to holidays json>, output_filepath: str=None)``` in ```utils.py```
- This enlists all daily attendance entry columns to be excluded from computations for all academic years.
- This is important as only those daily attendance columns not in ```cols_to_exclude.json``` are utlised to compute features.
- If new academic years are included, this file needs to be updated accordingly using ```data/holidays.json```.

```data/students_options_v1.json```
- This presents a dictionary wherein keys represent "different student types" and values are lists of student IDs.
- For example, the key "ay2223, grade3, sem2[frac_m]=[40%, 50%)" stands for students in academic year 2022-24, and grade 3, who have fraction of missing attendances in the range of 40% (included) and 50% (excluded) in semester 2.
- It is to be noted that all these computations and lists of students are generated using daily attendances post exclusion of holidays.
- This json file, once fully populated, MUST be saved as a pickle file for efficient storage and retrieval in the scripts
- It is not mandatory to pass this file in the training runs. 
- The file is only used to compute Baseline metrics which are shown as x,y points in precision-recall and recall@k plots. 
- Currently, the file contains dummy data to elucidate the structure. Users can update it to have a deeper evaluation of their trained models.

```data/column_dictionary.json```
- This is the universal dictionary with column names as keys and values as a list of appropriate datatype and description.
- It is preferred that column names aren't modified.
  - Modifying them would require modifying certain hardcoded aspects of scripts and configs as well.

```data/feature_groups.json```
- This is a dictionary that groups features for efficient and combined use.
- If this needs to be modified, it is preferred that the keys of the dictionary remain the same, and the values be modified as per requirement.
  - Changing the keys would impact certain hardcoded aspects of scripts and configs, which would then require corresponding modification.

```data/predictor_groups.json```
- This delineates all the predictor groups formulated to explain predictions and guide interventions.

</details>

# 🎯 Inference
<details>
<summary>[click to expand]</summary>

To obtain predictions on a new dataset using a model from a given experimental directory:
- Load the config from the experimental directory.
- Make any appropriate updates in the config to make it suitable for the new dataset.
  - For example, the name of the new dataset wouldn't be in the list ```config.label_engineering.implement_on```
    - To accomplish this, use ```config.data.label_engineering.implement_on.append(ds_name)```
  - Also, users might not want any filters in case results need to be obtained for all data points in the new dataset. 
    - To accomplish this, use ```config.data.filter = {}```
- Obtain dataset with predictions (probability scores and prediction column)
  - In a notebook cell, use ```dataset = predict_on_raw_data(exp_dir=exp_dir, config=config, file_path=infer_file)```
  - Parameters being
    - ```exp_dir```:- is the full path to the experimental directory
    - ```config```:- is the config dictionary loaded from the config json file
    - ```infer_file```:- is the full path to the new dataset file on which predictions need to be obtained
</details>

# 💡 Generating predictors
<details>
<summary>[click to expand]</summary>

**Methodology**
- We group features into predictor groups for intervention purposes.
  - Currently, predictor groups are as per ```data/predictor_groups.json```. 
- Computing contributions of predictor groups to prediction probability scores:
  - The features with positive SHAP scores contribute to a student dropping out.
  - The features with negative SHAP scores contribute to a student not dropping out.
- Based on the prediction class for a student, we pick features either with positive or negative SHAP scores.
  - The contribution of a predictor group to a student dropping out is the sum of SHAP scores of the features with positive SHAP scores only.
  - The contribution of a predictor group to a student not dropping out is the sum of SHAP scores of the features with negative SHAP scores only.
- Finally, we discern the top driving factor
  - For the non-dropout class, per predictor group, the feature with the maximum magnitude of the negative SHAP score is the top driving factor.
  - For the dropout class, per predictor group, the feature with the maximum positive SHAP score is the top driving factor.

**Pipeline to generate predictor groups and top driving factors**:
- Generate a dictionary of thresholds for required specific recalls (as computed on the evaluation set)
  - Use the ```generate_thresholds_json``` function in ```predict.py```

- Generate the dataframe with SHAP scores (contributions of each feature column to the dropout probability score)
  - Use the ```generate_and_save_shap_values``` function in ```shap_predictor_groups.py```
- Next, append columns showing contributions of the predictor groups
  - Use the ```process_and_save_shap_group_contributions``` function in ```shap_predictor_groups.py```
- Lastly, using a generated prediction column (based on a chosen threshold), discern and append columns of top driving factors per predictor group
  - Use the ```apply_threshold_and_save_selected_predictors``` function in ```shap_predictor_groups.py```

**Summary**
- The SHAP pipeline has been implemented in the function ```run_shap_pipeline``` in ```shap_predictor_groups.py```.
  -  Before calling this wrapper function, please generate the thresholds.json file using ```generate_thresholds_json``` in ```predict.py```.
- Example usage as below:
  -  fn: ```run_shap_pipeline(
               exp_dirs=["/path/to/exp_dir1", "/path/to/exp_dir2"],
               ds_name="prod[ay, fullsat]",
               df_cohorts_dict={"public": df_public, "private": df_private},
               predictor_groups=your_predictor_group_dict,
               target_recall=0.8,
               target_ds_name="val[sem1]",
     )```
</details>
