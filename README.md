<details>
<summary><span style="font-size: 20px">🗂️ OVERVIEW</span></summary>

---

### Problem
- Many students in India drop out of school due to diverse social, economic, and geographical factors.
- Students enrolled in a given academic year (AY) but **failing to re-enroll** in the *next* AY are dropouts.
- *Education gaps* lead to unskilled labour and are linked to poor health, impeding a nation's development.

### Motivation
- **Dropout indicators** are present in social traits, attendance patterns, and performance in assessments.
- This project aims to build an **Early Warning System (EWS)** using *machine learning (ML)* techniques to predict students who are at risk of dropping out of school.
- These predictions could potentially be used to cognize and design interventions to mitigate student dropouts.

### Data Sources
- The project has been developed using the following data that was provided by *Vidya Samiksha Kendra (VSK)—Samagra Shiksha, Department of Education, state of Gujarat, India*.
  - Enrollment data: Delineates a student's regional and socioeconomic factors.
  - Daily attendance data: Delineates a student's daily attendance (present, absent, or missing entry).
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

The [metadata](metadata) directory contains various schemas necessary to use this repository.

[Calendar of holidays](metadata/holidays_calendar.json)
- This is a *mandatory* nested JSON dictionary that holds information about holidays in AYs. 
- Example format: `{"2223": {"6": {"sundays": [5, 12, 19, 26], "vacation": [1, 2]}}}`
  - Stores non-working dates for each AY (e.g., "2223" for AY 2022-23) and month (e.g., "6" for June, and "7" for July). 
  - Dates are integers under categories like "sundays", "festive", "vacation", or others (e.g., "pravesh utsav").
- An example of this file for the AYs from 2022-23 to 2024-25 for the state of Gujarat is [here](metadata/holidays_calendar.json).
- Please edit the dictionary within this file for the AYs of your interest.
- This file could either be manually populated from a PDF or parsed from a CSV of holidays for any AY.

[Attendance Replacement Map](metadata/attendance_replacement_map.json)
- This is a *mandatory* JSON file that defines the semantic mapping for raw attendance values.
- It is used to convert different raw data values (e.g., `"1"`, `"2"`, `"nan"`) into the standardized representations for `"present"`, `"absent"`, and `"missing"`.
- The keys in this file **must** be `"present"`, `"absent"`, and `"missing"`. The validation logic strictly checks for these keys, which correspond to the `constants.Attendance.Status` class in `constants.py`, ensuring consistency.
- Please edit these constants for your dataset appropriately.

[Predictor groups](metadata/predictor_groups.json)
- This is an *optional* JSON dictionary logically categorizing similar features into predictor groups.
  - They are not required for the training or inference pipelines.
  - They are required only for *explainability*.
- Predictor groups are used to explain a model's predictions and guide interventions.
- Features may be manually organized into predictor groups depending on which aspects are most relevant for explanation. 
  - The features in [Predictor groups](metadata/predictor_groups.json) must be a subset of the features used in modeling.
  - We use programmatic inputs and guidance from *VSK* and *UNICEF India* to aggregate features into logical groups.
    - For example, features representing the location are grouped as "geographical_factors", and features representing attendance are grouped as "attendance_factors".
    - **Note:** Different strategies could be adopted to implement this aggregation and modify [Predictor groups](metadata/predictor_groups.json) based on your unique needs.

---

</details>

<details>
<summary><span style="font-size: 20px">📊 DATASET</span></summary>

---

A valid dataset for training and inference must satisfy the following requirements:

#### 1. File Format
- Dataset files must be in the pickle (`.pkl`) format. Other formats are not currently supported.

#### 2. Naming Convention
- The filename (excluding extension) must follow the pattern `ay<academic_year>_grade<grade>`.
- This convention is used to extract metadata like the academic year and grade.
- For example: `dataset/ay2223_grade3.pkl`.

#### 3. Schema Conformance
- The dataset must conform to the structure defined in the [Dataset Schema](metadata/dataset_schema.json).
- This is a *mandatory* JSON dictionary (explained [here](metadata/DATASET_README.md)) that defines the structure of a usable dataset.
- The columns in your dataset must be a subset of those defined in the schema.
- If your dataset uses different column names, you must update the [Dataset Schema](metadata/dataset_schema.json) accordingly.
- An illustrative sample dataset is shown [here](metadata/illustrative_dataset.csv).
  - **Disclaimer:** *This sample dataset contains synthetically generated data for demonstration purposes. Any resemblance to real individuals or entities is purely coincidental.*

---

</details>

<details>
<summary><span style="font-size: 20px">⚙️ CONFIGURATION</span></summary>

---

- A `Config` file is required to run the training or inference pipelines.
- This file must conform to the structure defined in the [Config Schema](metadata/config_schema.json).
- The schema file itself **should not be edited**. Instead, you should create a copy to use for your experiments.
- For a detailed explanation of all the parameters, please see the [Configuration README](metadata/CONFIG_README.md).

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
- Inference generates and saves a dataframe with features and predicted probabilities in `exp_dir`.

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
<summary><span style="font-size: 20px">🙏 ACKNOWLEDGEMENTS</span></summary>

---

We acknowledge with gratitude the collaborative partnership that has made EWS possible. This project is a pioneering initiative between Vidya Samiksha Kendra (VSK)-Gujarat, Wadhwani Institute for Artificial Intelligence (Wadhwani AI), and UNICEF to harness Machine Learning to mitigate school dropouts. We express our sincere appreciation for VSK, specifically the MIS Department, for providing comprehensive student data and program support, without which this transformative project would not have been realized. EWS demonstrates the power of collaborative innovation in education, uniting government institutions, technology leaders, and program partners to create a meaningful impact for Gujarat's children.

---

</details>

<details>
<summary><span style="font-size: 20px">🔔 NOTICE</span></summary>

---

- As things stand now, this repository will **NOT** be maintained.
- However, if you are interested in collaborating, please reach out to us at *education@wadhwaniai.org*. We would be happy to discuss and explore potential opportunities. 


---

</details>

<details>
<summary><span style="font-size: 20px">🛡️ LICENSE</span></summary>

- This project is licensed under the [Apache License 2.0](LICENSE) © 2025 Wadhwani Institute for Artificial Intelligence (Wadhwani AI).

</details>
