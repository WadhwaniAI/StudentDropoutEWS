import category_encoders as ce
import json
import math
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from imblearn.combine import SMOTETomek, SMOTEENN

from feature_engineering import *
from label_engineering import *
from preprocess import *
from utils import *



class Dataset():

     def __init__(
               self, 
               exp_dir: str, 
               config: dict, 
               file_path: str, 
     ) -> None:

          '''
          Dataset Class to implement many aspects including the following:
               > clean up data (if initiated)
               > engineer features
               > appropriate transforms
               > alter labels (if initiated)
          Args:
               file_path: path to the input csv file.
               exp_dir: path to the training directory.
               config: dict of the config file
               df: a loaded dataframe corresponding to file_path parameter.
          Returns:
               None. Creates a Dataset object.
          '''
  
          # experimental directory
          self.exp_dir = exp_dir
          os.makedirs(self.exp_dir, exist_ok=True)
          
          # config file
          self.config = config
          
          # path to dataset file
          self.file_path = file_path
          
          # name of dataset
          self.ds_name = self.file_path.split(" | ")[-1][:-4]
          
          # load dataset
          self.df = load_data_file(file_path=self.file_path)

          # grade
          self.grade = self.config.data.grade

          # label column of dataset
          self.label = self.config.data.label
          
          # index column of dataset
          self.index = self.config.data.index

          # list of holidays
          with open(self.config.data.holidays, 'r') as file:
               self.holidays = json.load(file)
               
          # academic year based on dataset
          self.acad_year = get_academic_year(self.ds_name)

          # apply filters
          self.filter(dict((k, v) for k, v in self.config.data.filter.items()))
          
          # basic preprocessing of the dataframe
          self.df, self.feature_groups = basic_preprocess(df=self.df, index=self.index, label=self.label)  
          
          # initialise a sample of the dataset
          self.sample = None

          # calls function to cache features to be dropped as they would not be needed for training / modeling
          self.drop_features()

          # current prevalence
          if self.label in self.df.columns:
               self.curr_p = self.df[self.label].mean()

          # label engineering
          if len(config.data.label_engineering.keys()) > 0:               
               self.label_engineering(**config.data.label_engineering)

          # feature engineering
          if len(config.data.feature_engineering.keys()) > 0:               
               self.feature_engineering(params_from_config=config.data.feature_engineering)

          # update features
          self.get_features()

          # fill categorical missing entries with 'nan'
          self.fill_cat_na()


     def commence(self):

          '''
          Description
               Sample the dataset to ensure given prevalence as per config.
               Thereafter, split the dataset into train and validation.
          '''

          # sample using prevalence
          keys = [k for k in self.config.data.sample.keys() if k != "seed"]
          if isinstance(self.config.data.sample[keys[0]], list):
               self.specific_sampling(
                    sample_ranges={k:v for k,v in self.config.data.sample.items() if k != 'seed'}
               )
          if isinstance(self.config.data.sample[keys[0]], int):
               self.random_sampling(
                    sample_sizes={k:v for k,v in self.config.data.sample.items() if k != 'seed'}
               )
          if isinstance(keys[0], str):
               p = self.config.data.sample[keys[0]]
               if p == 'actual':
                    p = self.curr_p
               if p < self.curr_p:
                    sample_sizes = {
                         0 : int(math.ceil(len(self.df) - sum(self.df[self.label]))), 
                         1 : int(math.ceil((len(self.df) - sum(self.df[self.label])) * p / (1-p)))
                    }
               else:
                    sample_sizes = {
                         0 : int(math.ceil(sum(self.df[self.label]) * ((1 - p) / p))), 
                         1 : int(math.ceil(sum(self.df[self.label])))
                    }
               self.random_sampling(sample_sizes=sample_sizes)

          # split into train, val and test
          self.train_val_split()


     def alter_dtypes(self, cols, dtype) -> None:

          '''  Alter data type of a column in a dataframe  '''
          self.df[cols] = self.df[cols].astype(dtype)


     def drop_cols(self, cols: list) -> None:

          '''  Drop columns from dataframe  '''
          self.df.drop(columns=cols, inplace=True)
          self.get_features()

     
     def drop_features(self) -> None:

          ''' columns to drop as per config as these are not required for training / modeling'''
          self.features_to_drop = []
          for feature_group in self.config.data.processing.drop_feature_groups:
               if feature_group in self.feature_groups.keys():
                    self.features_to_drop = [*self.features_to_drop, *self.feature_groups[feature_group]]
               elif feature_group in self.df.columns and isinstance(feature_group, str):
                    self.features_to_drop.append(feature_group)
          self.features_to_drop.sort()


     def drop_nan_cols(self, cols) -> None:

          '''  drop instances whose given col's value is nan  '''
          self.df.dropna(subset=cols, inplace=True)
          self.get_features()


     def drop_nan_rows(self) -> None:

          '''  drop instances which have atleast one nan  '''
          self.df.dropna(inplace=True)
          self.df.reset_index(inplace=True, drop=True)


     def extract_cols(self, cols) -> None:

          '''  Extract certain columns from dataframe  '''
          self.df = self.df[cols]
          self.get_features()


     def fill_cat_na(self, value='nan') -> None:

          '''  fill nans only in categorical features  '''
          # enlist categorical and numerical features
          self.get_features()

          # fill nans in categorical features     
          self.df[self.cat_features] = self.df[self.cat_features].fillna(value=value)


     def filter(self, filter_cols: dict) -> None:

          ''' Applies filter for columns with discrete values passes a list.'''

          if len(filter_cols.keys()) > 0: 

               assert set(filter_cols.keys()).issubset(set(['in', 'notin'])) # valid filter keys

               # filter if entry in col is in values
               for col, values in filter_cols['in'].items():
                    self.df = self.df[self.df[col].isin(values)]
               
               # filter if entry in col is not in values
               for col, values in filter_cols['notin'].items():
                    self.df = self.df[~self.df[col].isin(values)]
               
               # reset index column 0 onward post filtering
               self.df.reset_index(inplace=True, drop=True)
     

     def get_features(self, input_df=None) -> None:

          '''  set categorical, numerical, label and target features  '''

          if input_df is None:
               input_df = self.df
          
          # categorical features
          self.cat_features = list(input_df.select_dtypes(include=['object']).columns)
          if self.index in self.cat_features:
               self.cat_features.remove(self.index)

          # numerical features
          self.num_features = list(input_df.select_dtypes(include=[np.float64]).columns)
          not_num_features = [col for col in self.num_features if "preds" in col or self.label in col]
          for col in not_num_features:
               self.num_features.remove(col)
          
          # sort features
          self.cat_features.sort()
          self.num_features.sort()     

          # exclude categorical and numerical features which are in features_to_drop
          self.cat_features = [f for f in self.cat_features if f not in self.features_to_drop]
          self.num_features = [f for f in self.num_features if f not in self.features_to_drop]

          # all features in input dataframe
          self.features = self.cat_features + self.num_features

          
     def rename_cols(self, cols: dict) -> None:

          '''  rename columns  '''
          self.df.rename(columns=cols, inplace=True)
          self.get_features()


     def random_sampling(self, sample_sizes: dict) -> None:
          
          indices = []
          for label, num in sample_sizes.items():
               x = self.df.index[self.df[self.label]==int(label)].tolist()
               indices = [*indices, *x[0:num]]

          self.sample = self.df.loc[indices].sample(frac=1, random_state=self.config.data.sample.seed)
          self.sample.reset_index(inplace=True, drop=True)
     

     def specific_sampling(self, sample_ranges: dict) -> None:

          indices = []
          for label, range in sample_ranges.items():
               x = self.df.index[self.df[self.label]==int(label)].tolist()
               indices = [*indices, *x[range[0]:range[1]]]
          
          self.sample = self.df.loc[indices].sample(frac=1)
          self.sample.reset_index(inplace=True, drop=True)

          
     def train_val_split(self):
          
          '''
          Description:
               Splits into train and val. 
          '''
          # dict to store different data-subsets
          self.dataset = {}
          self.dataset['full_dataset'] = self.df

          # dataframe to use 
          if self.sample is None:
               self.sample = self.df
          
          # train and test dicts to store train and test x_train, y_train, x_test and y_test 
          self.dataset['trn'], self.dataset['val'] = {}, {}

          # split data into train val test sets'
          self.dataset['trn']['x'], self.dataset['val']['x'], \
          self.dataset['trn']['y'], self.dataset['val']['y'] = train_test_split(
               self.sample[[f for f in self.sample.columns if f != self.label]], 
               self.sample[self.label], 
               train_size=self.config.data.split.train_size, 
               stratify=self.sample[self.label],
               random_state=self.config.data.split.random_state,
               shuffle=self.config.data.split.shuffle
          )

          # concatenate X and y to form training and validation sets
          for ds_name in ['trn', 'val']:
               
               # concatenate x and y
               self.dataset[ds_name] = pd.concat([self.dataset[ds_name]['x'], self.dataset[ds_name]['y']], axis=1)
               
               # reset index column 0 onward
               self.dataset[ds_name].reset_index(inplace=True, drop=True)

               # reset index column 0 onward
               self.dataset[ds_name].reset_index(inplace=True, drop=True)            

          # update features
          self.get_features(input_df=self.dataset['trn'])


     def label_engineering(
               self,
               implement_on: list=["trn", "val"],
               task: str="altered classification",               
               months=[12, 1, 2, 3, 4],
               poas=[10, 15, 20, 30, 40, 50, 60],
               disc_cols_miss_frxn=1.0,
               drop_attnd_label: bool=True
     ) -> None:
               
          # alter target column to regression targets
          if task == "regression":
               self.df['dummy'] = ''
               def count(x):
                    return num_of_patterns(x, 'p')
               self.df[self.label] = self.df['dummy'].apply(count)
               self.df[self.label] = self.df[self.label].astype("int64")
          
          # apply label engineering if mentioned for given dataset
          if self.ds_name in implement_on:

               # labels using attendance-based rules
               if task == "altered classification":
                    self.df, relevant_attendances, poas, columns_to_drop = attendance_labels(
                         df=self.df,
                         holidays=self.holidays,
                         acad_year=self.acad_year,
                         months=months,
                         poas=poas,
                         disc_cols_miss_frxn=disc_cols_miss_frxn,
                         drop_attnd_label=drop_attnd_label
                    )
                    
                    # update features to drop
                    self.features_to_drop.extend(columns_to_drop)

                    # get features
                    self.get_features()
          

     def feature_engineering(
               self,
               params_from_config
     ) -> None:
          
          self.df, columns_to_drop = generate_attendance_features(
               df=self.df,
               holidays=self.holidays,
               acad_year=self.acad_year,
               all_attendances=self.feature_groups["all_attendances"],
               **params_from_config
          )

          # update features to drop
          self.features_to_drop.extend(columns_to_drop)

          # get features 
          self.get_features()