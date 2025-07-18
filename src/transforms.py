import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA



class Transforms():

     def __init__(
               self,
               dataframe: pd.DataFrame,
               num_features: list,
               label: str,
               exp_dir: str,
               feature_groups: list,
               config_transforms: dict
     ) -> None:
          
          '''
          Description.
               This class transforms numerical features in a dataframe based on config.
               The transformers are saved as pickle files in the experimental directory for later use.
          Args:
               dataframe: a pandas dataframe.
               num_features: list of all numerical features (columns) in the input dataframe.
               label: name of label column.
               exp_dir: path to experimental directory.
               feature_groups: feature groups of the dataset object to deal with features together.
               config_transforms: all transformations and corresponding columns as in config.
          Returns:
               A (numerical) transforms object.
          '''

          # copy of input dataframe
          self.data = dataframe.copy(deep=True)
          
          # list of numerical features (of type np.float64)
          self.num_features = num_features
          
          # path to experimental directory
          self.exp_dir = exp_dir
          
          # name of label olumn
          self.label = label
          
          # list of feature groups for transformations
          self.feature_groups = feature_groups
          
          # transforms from config
          self.transforms = config_transforms
          
          # all valid transforms
          self.possible_transforms = {
               'standard_scaler': StandardScaler,
               'power_transformer': PowerTransformer,
               'pca': PCA,
               'quantile_transformer' : QuantileTransformer
          }

          # check if config has valid transforms
          assert set(self.transforms.keys()).issubset(set(self.possible_transforms.keys()))


     def fit(
               self
     ) -> None:
          
          '''
          Description:
               This method initialises and fits transformers on given numerical features.
               The fitted transfomers are saved in the experimental directory.
          '''

          for name, params in self.transforms.items():
               
               # features to implement transform on
               features = []
               for feature_group in params.features:
                    if feature_group in self.feature_groups.keys():
                         features = [*features, *self.feature_groups[feature_group]]
                    elif feature_group == "all numeric features":
                         features = [*features, *self.num_features]
                    elif isinstance(feature_group, str):
                         features.append(feature_group)
               features = list(set(features))
               features = list(set(features).intersection(set(self.data.columns)))
               features.sort()

               # parameters to initialise the transform object
               init_params = {k: v if v != 'None' else None for k, v in params.init_params.items()}
               
               # initialise transformer
               trnsfrm = self.possible_transforms[name](**init_params)

               # fit the transformer
               trnsfrm.fit(np.array(self.data[features]))
               
               # save the transformer in exp directory
               with open(f"{self.exp_dir}/{name}.pkl",'wb') as f:
                    pickle.dump(trnsfrm, f)

               # apply transform to update dataframe for next transform
               X = trnsfrm.transform(np.array(self.data[features]))
               df = pd.DataFrame.from_dict({f'{features[n]}':X[:,n] for n in range(0, X.shape[1])})
               self.data.drop(columns=features, inplace=True)
               self.data = pd.concat([self.data, df], axis=1)


     def apply(
               self
     )-> pd.DataFrame:

          '''
          Description:
               This method loads the pickle files of transformers from the experimental directory.
               Thereafter, it applies them on the given dataframe and returns the transformed version. 
          '''

          for name, params in self.transforms.items():
               
               # features to implement transform on
               features = []
               for feature_group in params.features:
                    if feature_group in self.feature_groups.keys():
                         features = [*features, *self.feature_groups[feature_group]]
                    elif feature_group == "all numeric features":
                         features = [*features, *self.num_features]
                    elif isinstance(feature_group, str):
                         features.append(feature_group)
               features = list(set(features))
               features = list(set(features).intersection(set(self.data.columns)))
               features.sort()
               
               # load the transformer from exp directory
               with open(f"{self.exp_dir}/{name}.pkl",'rb') as f:
                    trnsfrm = pickle.load(f)

               # apply transform to update dataframe for next transform
               X = trnsfrm.transform(np.array(self.data[features]))
               df = pd.DataFrame.from_dict({f'{features[n]}':X[:,n] for n in range(0, X.shape[1])})
               self.data.drop(columns=features, inplace=True)
               self.data = pd.concat([self.data, df], axis=1)

          # final transformed dataframe
          return self.data