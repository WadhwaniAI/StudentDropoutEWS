<details>
<summary><span style="font-size: 20px">🗂️ OVERVIEW</span></summary>

---

### Problem
- Many students in India drop out of schools due to diverse social, economic and geographical factors.
- Students enrolled in a given academic year (AY) but **failing to re-enroll** in the *next* AY are dropouts.
  - This definition does not account for the students who drop out within an AY.
  - *As long as a student re-enrolls, the student is not a dropout.* 
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
- This is a *mandatory* nested JSON dictionary that holds information about holidays in AYs. 
- Example format: `{"2223": {"6": {"sundays": [5, 12, 19, 26], "vacation": [1, 2]}}}`
  - Stores non-working dates for each AY (e.g., "2223" for AY 2022-23) and month (e.g., "6" for June, and "7" for July). 
  - Dates are integers under categories like "sundays", "festive", "vacation", or others (e.g., "pravesh utsav").
- An example of this file for the AYs from 2022-23 to 2024-25 for the state of Gujarat is [here](metadata/holidays_calendar.json).
- Please edit the dictionary within this file for the AYs of your interest.
- This file could either be manually populated from a PDF or parsed from a CSV notified by the administration.

[Dataset Schema](metadata/dataset_schema.json)
- This is a *mandatory* JSON dictionary that defines the structure of a usable dataset.
- Each key is a column name and its corresponding value is a list containing:
  - Datatype: `str` (categorical), `float` (numerical), and `int` (target)
  - Description: A piece of text briefly explaining the column and what it contains, and
  - Grouping: The logical group it belongs to (used in preprocessing)
- Modify [Dataset Schema](metadata/dataset_schema.json) if your dataset has different column names, datatypes, descriptions or groupings.

[Config Schema](metadata/config_schema.json)
- This is a *mandatory* nested JSON dictionary that defines the structure of a `Config` file used for training or inference.
- A new `Config` file needs to be created by the user for every training run, and an existing one is used to infer on a new dataset.
- [Config Schema](metadata/config_schema.json) **should not be deleted or edited**.
  - A copy of this file needs to be made by the user for their own experiments.
- This is explained in detail [here](metadata/CONFIG_README.md).

[Predictor groups](metadata/predictor_groups.json)
- This is an *optional* JSON dictionary logically categorizing similar features into predictor groups.
  - They are not required for the training or inference pipelines.
  - They are required only for *explainability*.
- Predictor groups are used to explain a model's predictions and guide interventions.
- Features are manually organized into predictor groups—currently there is no script to generate them.
  - The features in [Predictor groups](metadata/predictor_groups.json) must be a subset of the features used in modeling.
  - We use programmatic inputs and guidance from *VSK* and *UNICEF India* to aggregate features into logical groups.
    - For example, features representing the location are grouped as "geographical_factors" group, and features representing attendance are grouped as "attendance_factors".
    - **Note:** Different strategies could be adopted to implement this aggregation.
- Please modify [Predictor groups](metadata/predictor_groups.json) for different group explanations.

---

</details>

<details>
<summary><span style="font-size: 20px">📊 DATASET</span></summary>

---

- A valid dataset for training and inference must conform to the [Dataset Schema](metadata/dataset_schema.json). 
  - The columns of your dataset must be a subset of the columns in [Dataset Schema](metadata/dataset_schema.json).
  - If your dataset has different column names, please modify [Dataset Schema](metadata/dataset_schema.json) accordingly before use.
- Dataset files must be in a pickle (`.pkl`) format only (e.g., `dataset/ay2223_grade3.pkl`).
  - Currently, support for other file formats is not provided.
- The stemname of a dataset filepath is used to extract metadata like the "academic year" and "grade".
  - It must follow the pattern: `ay<academic_year>_grade<grade>`. Eg: `dataset/ay2223_grade3.pkl`.

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
<summary><span style="font-size: 20px">🙏 ACKNOWLEDGEMENTS</span></summary>

---

We acknowledge with gratitude the collaborative partnership that has made EWS possible. This project is a pioneering initiative between VSK, Wadhwani Institute for Artificial Intelligence (Wadhwani AI), and UNICEF to harness Machine Learning to mitigate school dropouts. We express our sincere appreciation for VSK, specifically the MIS Department, for providing comprehensive student data and program support, without which this transformative project would not have been realized. EWS demonstrates the power of collaborative innovation in education, uniting government institutions, technology and program partners for Gujarat's children.

---

</details>

<details>
<summary><span style="font-size: 20px">🔔 NOTICE</span></summary>

---

- As things stands now, this repository will not be maintained by Wadhwani AI.
- However, if you would like to collaborate, please reach out to us on *(enter email address)*.
  - We would be happy to discuss and explore opportunities!

Thankyou.

---

</details>