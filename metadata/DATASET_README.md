**This README explains the [Dataset Schema](dataset_schema.json).**

---

The Dataset Schema is a mandatory JSON file that defines the expected structure, data types, and logical groupings of columns in your dataset. The `DataPreprocessor` and `DatasetSchemaValidator` rely on this file to correctly process and validate your data.

### 1. Schema Structure

Each entry in the schema corresponds to a column in your dataset. The key is the column name (which must be lowercased), and the value is a list containing:

1.  **Data Type (`str`)**: The expected data type of the column. This must be one of the values defined in `constants.DtypeCastMap`.
2.  **Description (`str`)**: A brief, human-readable description of what the column represents.
3.  **Column Group (`str`, Optional)**: A logical group name. This is used for validation and applying group-specific logic during preprocessing. This must be one of the values from `constants.ColumnGroups`—This need not be specified for the label and index columns.

**Example:**
```json
"bpl": [
     "string",
     "below poverty line status of a student",
     "socioeconomic"
]
```

---

### 2. Alignment with `constants.py`

The `DatasetSchemaValidator` enforces strict rules based on constants defined in `src/constants.py`. Any modifications to the schema must align with these constants to ensure the pipeline runs smoothly.

#### **Data Types (`constants.DtypeCastMap`)**

The first element of each schema entry must be a valid data type string defined in `constants.DtypeCastMap`.

-   `constants.DtypeCastMap.STR` (value: `"string"`) for categorical features.
-   `constants.DtypeCastMap.FLOAT` (value: `"float"`) for numerical features.
-   `constants.DtypeCastMap.INT` (value: `"int"`) for the target label column.

#### **Label Column (`constants.ColumnNames.LABEL`)**

The schema **must** contain an entry for the target label, and its name must match `constants.ColumnNames.LABEL` (value: `"target"`). Its data type must be `constants.DtypeCastMap.INT`.

```json
"target": [
     "int",
     "ground truth dropout label of a student"
]
```

#### **Pattern-Based Column Groups (`constants.ColumnGroups`)**

For certain column types identified by naming patterns, the `DatasetSchemaValidator` expects a specific data type and column group.

1.  **Daily Attendance Columns**:
    -   **Pattern**: Matches `constants.Attendance.PATTERN` (e.g., `"6_1"`, `"12_31"`).
    -   **Expected Type**: `constants.DtypeCastMap.STR` (`"string"`).
    -   **Expected Group**: `constants.ColumnGroups.ALL_ATTENDANCES` (`"all_attendances"`).
    ```json
    "6_1": [
         "string",
         "daily attendance entry for day-1 of month-06...",
         "all_attendances"
    ]
    ```

2.  **Monthly Aggregate Attendance Columns**:
    -   **Pattern**: Ends with `_agg_attnd` (e.g., `"jun_agg_attnd"`).
    -   **Expected Type**: `constants.DtypeCastMap.FLOAT` (`"float"`).
    -   **Expected Group**: `constants.ColumnGroups.MONTH_AGG_ATTND` (`"month_agg_attnd"`).
    ```json
    "jun_agg_attnd": [
         "float",
         "aggregate attendance entry for jun month...",
         "month_agg_attnd"
    ]
    ```

3.  **Subject-wise Exam Score Columns**:
    -   **Pattern**: Ends with `_score` (e.g., `"maths_score"`).
    -   **Expected Type**: `constants.DtypeCastMap.FLOAT` (`"float"`).
    -   **Expected Group**: `constants.ColumnGroups.EXAM_SCORE_SUBWISE` (`"exam_score_subwise"`).
    ```json
    "maths_score": [
         "float",
         "score achieved in maths SAT...",
         "exam_score_subwise"
    ]
    ```

4.  **Subject-wise Exam Attendance Columns**:
    -   **Pattern**: Ends with `_attnd` (e.g., `"maths_attnd"`).
    -   **Expected Type**: `constants.DtypeCastMap.STR` (`"string"`).
    -   **Expected Group**: `constants.ColumnGroups.EXAM_ATTND_SUBWISE` (`"exam_attnd_subwise"`).
    ```json
    "maths_attnd": [
         "string",
         "attendance entry for maths SAT...",
         "exam_attnd_subwise"
    ]
    ```

---

### 3. How to Modify

-   **Adding New Columns**: If you add a new column to your dataset, add a corresponding entry to `dataset_schema.json`, ensuring the data type and group (if applicable) are correct.
-   **Changing Column Names**: If you rename a column in your data, you must update its key in the schema. If the column name follows a specific pattern (e.g., `_score`), ensure it still adheres to the rules above.
-   **Custom Groups**: You can define your own column groups. However, for the pattern-based rules to work, the groups for those specific columns must match the ones defined in `constants.ColumnGroups`.

By adhering to these guidelines, you ensure that the data preprocessing and validation steps function as intended, leading to a robust and error-free modeling pipeline.