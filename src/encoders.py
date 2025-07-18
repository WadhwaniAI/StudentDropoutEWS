import category_encoders as ce
import os
import pandas as pd
import pickle



class Encoders():

     def __init__(
               self,
               dataframe: pd.DataFrame,
               cat_features: list,
               label: str,
               exp_dir: str,
               feature_groups: list,
               config_encodings: dict
     ) -> None:
          
          '''
          Description:
               This class defines encoders for an experiment to use to encode categorical columns.
          Args:
               dataframe: The input pandas dataframe.
               cat_features: The list of all categorical features in the input dataframe.
               label: The name of the label column.
               exp_dir: The path to the experimental directory.
               feature_groups: List of feature_groups to address a bunch of features together.
               config_encodings: The dictionary with type of encoder as key and which columns to encode as values.
          Returns:
               None. Defines an encoder object for an experiment. 
          '''

          # copy of dataframe to work with
          self.data = dataframe.copy(deep=True)
          
          # name of label column
          self.label = label
          
          # list of all valid categorical features
          self.cat_features = cat_features
          
          # experimental directory
          self.exp_dir = exp_dir
          
          # list of feature groups as per config
          self.feature_groups = feature_groups
                    
          # folder to save encodings in exp directory
          self.folder = f"{self.exp_dir}/encodings"
          os.makedirs(self.folder, exist_ok=True)
          
          # encodings to implement as per config
          self.encodings = config_encodings
          
          # list of all possible encoders to use
          self.possible_encoders = {
               "binary": ce.BinaryEncoder,
               "count": ce.CountEncoder,
               "hashing": ce.HashingEncoder,
               "helmert": ce.HelmertEncoder,
               "jamesstein": ce.JamesSteinEncoder,
               "onehot": ce.OneHotEncoder,
               "ordinal": ce.OrdinalEncoder,
               "quantile": ce.QuantileEncoder,
               "sum": ce.SumEncoder,
               "target": ce.TargetEncoder
          }
          
          # to validate if features given in config are in the dataset
          for encoder, feature_groups in self.encodings.items():
               assert encoder in self.possible_encoders.keys()
               features = []
               for feature_group in feature_groups:
                    if feature_group in self.feature_groups.keys():
                         features = [*features, *self.feature_groups[feature_group]]
                    elif isinstance(feature_group, str) and feature_group in self.cat_features:
                         features.append(feature_group)
                    else:
                         raise Exception("Invalid feature or feature_group")
               features = list(set(features))
               assert set(features).intersection(set(self.cat_features)) == set(features)
               

     def fit(
               self
     ) -> None:          
          
          '''
          Description:
               This method fits the encoders on the corresponding categorical columns. 
               The fit method gets applied when the input dataframe is the training dataframe.
          '''
          
          # iterate over each encoding option and features as per config
          for encoder, feature_groups in self.encodings.items():
               
               # list to hold all features to encode as per config for each encoder 
               features = []
               for feature_group in feature_groups:
                    if feature_group in self.feature_groups.keys():
                         features = [*features, *self.feature_groups[feature_group]]
                    elif isinstance(feature_group, str) and feature_group in self.data.columns:
                         features.append(feature_group)
               
               # set of all features to encode
               features = list(set(features))
               
               # define encoder as per config and fit to features
               enc = self.possible_encoders[encoder](cols=features)
               enc.fit(self.data, self.data[self.label])
               
               # save pickle file of encoder for further implementation
               with open(f"{self.folder}/{encoder}.pkl","wb") as f:
                    pickle.dump(enc, f)


     def apply(
               self
     )-> pd.DataFrame:     

          '''
          Description:
               This method applies the saved encoders from the experimental directory on the input dataframes.
          Returns:
               Tranformed dataframe with categorical columns encoded.
          ''' 

          # load pickle file from exp directory and apply on input dataframes
          for encoder, _ in self.encodings.items():
               with open(f"{self.folder}/{encoder}.pkl","rb") as f:
                    enc = pickle.load(f)    
               self.data = enc.transform(self.data)
          
          # return encoded dataframe
          return self.data