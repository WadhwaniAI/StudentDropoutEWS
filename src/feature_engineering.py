import itertools

from eval import *
from utils import *



def num_of_patterns(astr, pattern):
    
     '''
     Description:
          To compute the number of occurrences of an input pattern in an input string.
          For example, for a given string "papapapapa" and pattern "papa", 4 would be returned.
     Args:
          astr: input string.
          pattern: input pattern.
     Returns:
          Number of occurrences of pattern in input string (int).
     '''

     astr, pattern = astr.strip(), pattern.strip()
     if pattern == '': return 0

     ind, count, start_flag = 0,0,0
     while True:
          try:
               if start_flag == 0:
                    ind = astr.index(pattern)
                    start_flag = 1
               else:
                    ind += 1 + astr[ind+1:].index(pattern)
               count += 1
          except:
               break
     return count



def formulate_attendance_features(combinations_of_characters: list):

     '''
     Description:
          Generates all possible combinations, for given characters, and corresponding maximum lengths. 
          For example, given the characters ["p", "a"] and maximum length 4, all combinations of characters of length 1 to 4 would be generated.
          These would include ['p', 'pa', 'papa', ..., ].
     Args:
          combinations_of_characters: A list of tuples. Each tuple contains a maximum length and a list of characters.
     Returns:
          List of all possible combinations of characters.
     '''

     # placeholder list to cache new features
     new_features = []

     # loop over all combinations of characters
     for max_length, characters in combinations_of_characters:
          for combination_length in range(1, int(max_length)+1):
               combinations = itertools.product(characters, repeat=combination_length)
               for combination in combinations:               
                    new_features.append(''.join(combination))
     
     # sort and return new features
     new_features = list(set(new_features))
     new_features.sort()

     return new_features



def scaling_factor_attendance_features(L, s):
     
     '''
     Description:
          Computes the maximum possible occurrences of a string "s" in a given length value L. 
          For example, for a given string s "papa" and length L=10, the maximum possible occurences of s in L would be 4 ("papapapapa").
     Args:
          L: The length in which s needs to fit. 
          s: The string whose maximum possible occcurences in length L need to be computed.
     Returns:
          Max possible occurrences (int) of s in L.
     '''

     # l is max length from start of string s that occurs again in some part of s. 
     l = 0
     for i in range(1, len(s)):
          if s[0:i] == s[-i:]:
               if i > l:
                    l = i
     
     # counter to keep track of number of occurrences
     ctr = 1
     
     # appending unit in given pattern
     unit_to_append = s[l:]
     
     # continue until we fill L totally
     while (len(s)+len(unit_to_append)) <= L:
          s = s + unit_to_append
          ctr += 1
     
     # final number of occurrences is counter
     return ctr



def generate_attendance_features(     
          df: pd.DataFrame,
          holidays: dict,
          acad_year: str,
          all_attendances: list,
          groups_of_months: dict={"sem1": [6, 7, 8, 9, 10, 11]}, 
          disc_cols_miss_frxn: float=1.0,          
          combs_of_chars: list=[[1, ['a','m','p']]],
          partitions: list=[1],
          last_a_m_p_features: bool=True,
          drop_all_attendances: bool=True,
) -> pd.DataFrame:

     '''
     Generate engineered attendance features from daily attendance records.

     Description:
          This function processes daily student attendance data to generate aggregated features across defined groups of months. 
          It supports generating features based on custom character patterns, partitioning attendance records into multiple segments, and 
          computing the last occurrence positions of attendance types (like 'a', 'm', 'p'). 
          'a' is attendance entries for absence, 'p' is attendance entries for presence and 'm' is attendance entries if missing.
          It also allows dropping original daily attendance columns and tracks columns that were added for cleanup.

     Args:
          df (pd.DataFrame): Input DataFrame containing daily attendance columns.
          holidays (dict): Dictionary mapping academic years to lists of holiday column names.
          acad_year (str): Academic year string used to filter out holidays from attendance.
          all_attendances (list): List of all daily attendance column names.
          groups_of_months (dict, optional): Mapping of labels to lists of month numbers to group attendance columns.
          disc_cols_miss_frxn (float, optional): Maximum fraction of 'm' allowed in an attendance column to include it.
          combs_of_chars (list, optional): List of [length, character_set] combinations to extract attendance patterns.
          partitions (list, optional): List specifying the number of partitions to divide the attendance string into.
          last_a_m_p_features (bool, optional): Whether to compute features for the last occurrence of 'a', 'm', and 'p'.
          drop_all_attendances (bool, optional): Whether to drop original attendance columns after feature generation.

     Returns:
          Tuple[pd.DataFrame, list]: 
               - The DataFrame with new attendance features added.
               - A list of column names that can be safely dropped (e.g., original attendance columns, temporary dummy columns).
     '''

     # list of features to be computed
     new_features = formulate_attendance_features(combinations_of_characters=combs_of_chars)

     # variable to represent a dummy column to hold concatenated attendances
     dummy = "concat_attendances"

     # looping over month groups
     for group, months in groups_of_months.items():

          # convert list of months to string
          months = [str(m) for m in months]
          
          # conditions a column must satisfy to be included in relevant attendance columns
          def rules(column):
               conditions = [
                    column not in holidays[acad_year],
                    column.split("_")[0] in months,
                    "m" not in set(df[column]) or df[column].value_counts(normalize=True)["m"] < disc_cols_miss_frxn
               ]
               return conditions
          
          # relevant attendance columns
          relevant_attendances = [col for col in all_attendances if all(rules(col))]

          # add last occurrence of "p", "a", "m" in current group of months
          if last_a_m_p_features:

               # concatenate relevant attendances as a dummy column
               #df[dummy] = df[relevant_attendances].agg(''.join, axis=1)      
               df = pd.concat([df.drop(columns=[dummy], errors='ignore'), df[relevant_attendances].agg(''.join, axis=1).rename(dummy)], axis=1)
               
               # loop over daily attendance elements
               cols_to_concat = {}
               for el in ["a", "m", "p"]:
                    
                    # function to compute last occurrence of an element in a string
                    def last_occurrence(x):
                         return (x.rfind(el)+1) / (len(x) + 1e-10)
                    
                    # column name
                    col = f"[{group}][last_{el}]"
                    
                    # compute column
                    #df[col] = df[dummy].apply(last_occurrence).astype(np.float64)    
                    cols_to_concat[col] = df[dummy].apply(last_occurrence).astype(np.float64)

               # add the new columns to the dataframe
               cols_to_concat_df = pd.DataFrame(cols_to_concat, index=df.index)
               df = pd.concat([df.drop(columns=cols_to_concat_df.columns, errors="ignore"), cols_to_concat_df], axis=1)

          # loop over different number of partitions per months_group
          for n_p in partitions:
               
               # number of days per partition
               d, rem = divmod(len(relevant_attendances), n_p)
               
               # remainder columns flag to use when using columns in last partition
               rem_col = 0

               # looping over partitions to generate features
               for i in range(0, n_p):

                    # if last partition, include remainder columns if any
                    if i == (n_p-1):
                         rem_col = rem

                    # concatenated attendances column to generate features from
                    #df[dummy] = df[relevant_attendances[i*d : (i+1)*d + rem_col]].agg(''.join, axis=1)
                    df = pd.concat([df.drop(columns=[dummy], errors='ignore'), df[relevant_attendances[i*d : (i+1)*d + rem_col]].agg(''.join, axis=1).rename(dummy)], axis=1)
                    
                    # max length of a dummy attendance string
                    max_len = len(list(df[dummy])[0])
                    
                    # generate features        
                    cols_to_concat = {}       
                    for new_feature in new_features:

                         # function to count occurrences of new_feature in an element x
                         def count(x):
                              return num_of_patterns(x, new_feature)
                         
                         # name of new feature column
                         new_col = f"[{group}][#partns={n_p}][partn_{i+1}, frac_{new_feature}]"
                         
                         # compute new_feature column
                         cols_to_concat[new_col] = df[dummy].apply(count)             
                         
                         # scaling factor
                         scaling_factor = scaling_factor_attendance_features(L=max_len, s=new_feature)
                         
                         # scale the new feature column and convert to float64
                         cols_to_concat[new_col] = (cols_to_concat[new_col] / scaling_factor).astype(np.float64)    

                    # add the new columns to the dataframe
                    cols_to_concat_df = pd.DataFrame(cols_to_concat, index=df.index)
                    df = pd.concat([df.drop(columns=cols_to_concat_df.columns, errors="ignore"), cols_to_concat_df], axis=1)         
     
     # columns to drop
     columns_to_drop = []

     # drop dummy column
     columns_to_drop.append(dummy)

     # drop new daily attendance categorical columns
     if drop_all_attendances: 
          columns_to_drop.extend(all_attendances)

     # set indices from 0 onward
     df.reset_index(inplace=True, drop=True)

     # return updated dataframe
     return df, columns_to_drop
