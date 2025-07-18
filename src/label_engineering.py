import numpy as np
import pandas as pd



def attendance_labels(
          df: pd.DataFrame,
          holidays: dict,
          acad_year: str,
          months: list=[12, 1, 2, 3, 4],
          poas: list=[10, 15, 20, 30, 40, 50, 60],
          disc_cols_miss_frxn: np.float32=1.0,
          drop_attnd_label: bool=True,
) -> list:
     
     '''
     Description:
          Generates new labels for each record using an N-day attendance rule and returns list of new targets, features used and 
          period of absence as days (incase input was a fraction).           
          The N-day rule: 
          If a student is absent for N consecutive school working days, and is not present thereafter, the student is tagged as a dropout.
          This is a heuristical label. In our case, we instead use it as a feature. 
          Here, N-days takes values in the poas: which is the list of days being considered for consecutive "periods of absence".
     Args:
          df: input dataframe
          year: relevant academic year of the input dataframe.
          label: name of label column.
          months: which months to use to compute the new labels.
          poas: list of periods of absence to consider.
          disc_cols_miss_frxn: The threshold of missing entries in a column to use to discard them.
          drop_attnd_label: whether to drop the attendance label column/columns after computation/usage.
          holidays_file: file containing list of holidays to exclude, academic year wise.
     Returns:
          List of new targets, features and and period of absence that had been used.
     '''

     # dummy column to hold concatenated attendances
     dummy = "concatenated_attendances"

     # months to be strings
     months = [str(m) for m in months]

     # conditions a column must satisfy to be included in features
     def rules(column):
          conditions = [
               column.split("_")[0] in months,
               column not in holidays[acad_year],
               df[column].value_counts(normalize=True)["m"] < disc_cols_miss_frxn
          ]
          return conditions
     
     # all attendance columns
     all_attendance = [col for col in df.columns if col.split("_")[0].isdigit() and col.split("_")[1].isdigit()]

     # check if all_attendance is not empty
     if not all_attendance:
          raise ValueError("No attendance columns found in the dataframe.")
     
     # relevant columns
     relevant_attendances = [col for col in all_attendance if all(rules(col))]
     
     # convert to string to combine all of them
     df[relevant_attendances] = df[relevant_attendances].astype('str')

     # combine relevant features per student into one string for search
     #df[dummy] = df[relevant_attendances].agg(''.join, axis=1)
     df = pd.concat([df.drop(columns=[dummy], errors='ignore'), df[relevant_attendances].agg(''.join, axis=1).rename(dummy)], axis=1)

     # columns to drop
     columns_to_drop = [*all_attendance, dummy]
     
     # loop over poas (periods of absence)
     cols_to_concat = {}
     for poa in poas:

          # function to compute attendance labels based on the period of absence
          get_attnd_label = get_attnd_label_for_poa(poa)

          # new label column based on attendance, and convert to float to use it as feature if required
          cols_to_concat[f"attnd_label_{poa}"] = df[dummy].apply(get_attnd_label).astype(np.float64)

          # drop attendance label if required
          if drop_attnd_label:
               columns_to_drop.append(f"attnd_label_{poa}")
     
     # add the new label columns to the dataframe
     cols_to_concat_df = pd.DataFrame(cols_to_concat, index=df.index)
     df = pd.concat([df.drop(columns=cols_to_concat_df.columns, errors="ignore"), cols_to_concat_df], axis=1)

     # return the labels
     return df, relevant_attendances, poas, columns_to_drop



def get_attnd_label_for_poa(poa):

     '''
     Description: 
          Returns the function used to compute attendance labels based on a period of absence (poa).
     Args:
          poa: period of absence to consider.
     Returns:
          Function to compute attendance labels based on the period of absence.
     '''

     def get_attnd_label(x):

          '''
          Description:
               Computes attendance labels based on a predefined period of absence.
          Args:
               x: input concatenated attendance string
          Returns:
               0 if student was present for the period of absence, else 1 in case of absence or missing attendance
          '''
          if 'p' in x[-poa:]:
               return 0
          else:
               return 1
     
     # return the function
     return get_attnd_label
