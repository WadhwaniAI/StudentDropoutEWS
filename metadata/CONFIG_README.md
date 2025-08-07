<details>
<summary><span style="font-size: 20px">📘 CONFIG</span></summary>

**This README explains in detail the [Config Schema](metadata/config_schema.json).**
---

- A new JSON Configuration file is used to define all aspects for training a model.
- An existing JSON configuration file (from a previous experiment) is used to run inference on a new dataset.
- In [Config Schema](metadata/config_schema.json), all *optional* parameters are denoted within `<key>` and the placeholders indicate valid datatypes.
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
          "index": "<unique_id_column>",                             // str: Unique ID column; Eg: "studentid"
          "label": "<target_column>",                                // str: Target label column name; Eg: "target"
          "holidays_calendar_path": "<path_to_holidays_calendar>",   // str: JSON with academic holidays metadata; Eg: "metadata/holidays_calendar.json"
          "column_filters": {                                        // This helps to filter rows based on categorical conditions.
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
          "engineer_features": {                                     // The key set of parameters that define feature engieering.
               "groups_of_months": { "<group>": [<months>] },        // dict[str, list[int]]: Month groupings; Eg: { "full": [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4] }
               "combs_of_chars": [[<max len>, ['a','m','p']]],       // list[list[int, list[str]]]: Max length of permutation, subset of ("a", "m", "p") to use; Eg: [[1, ["m", "p", "a"]]]
               "partitions": [<int>],                                // int: Number of partitions to split each month group; Eg: [3]
               "disc_cols_miss_frxn": <float>,                       // float: Permitted max limit of fraction of missing attendance entries; Eg: 0.9
               "months_for_binary": [<months>],                      // list[int]: Months used for binary features; Eg: [6, 7, 8, 9, 10]
               "absence_thresholds": [<ints>]                        // list[int]: Thresholds (days of continuous absenteeism) to define binary absence. Eg: [10, 15, 30]
          },
          "drop_columns_or_groups": [                                // Features that are counter-productive to modeling and need to be dropped are specified here.
               "<col_or_group1>", "<col_or_group2>"                  // list[str]: Drop any columns or groups; Eg: ["schoolid", "[full][#partns=3][partn_3, frac_p], "exam_attnd_subwise"]
          ]
     },
     "model": {                                                      // The section defines all the modeling-related aspects.
          "n_trials": <int>,                                         // int: Number of hyperparameter tuning trials (gets used only if we tune else is ignored); Eg: 50
          "calibration_nbins": <int>,                                // int: Bins for calibration of probability scores; Eg: 20
          "params": {                                                // Defines all CatBoost training parameters.
               "fixed": {                                            // Fixed parameters (Are not tuned); Mandatory. When "tune" is not present in the Config, all CatBoost parameters need specified here.
                    "loss_function": "Logloss",                      // str: Objective function; Eg: "Logloss"
                    "random_seed": <int>,                            // int: Seed for model reproducibility; Eg: 0
                    "task_type": "<CPU|GPU>",                        // str: Hardware to use; Eg: "CPU"
                    "devices": "<GPU_ids>",                          // str: GPU ID device string (optional); Eg: "0", "0,1"
                    "auto_class_weights": "<a valid value>"          // str: Class imbalance handling; Eg: "Balanced"
               },
               "<tune>": {                                           // [Optional] Specify only for Hyperparameter tuning needs to be done.
                    "independent": {                                 // Independent hyperparameters: Their presence does not depend on the presence or absence of any other CatBoost parameter.
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
                    "dependent": {                                   // Dependent hyperparameters: Their presence depends on another parameter.
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