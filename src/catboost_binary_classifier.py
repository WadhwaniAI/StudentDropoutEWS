import json
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pickle

from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

from utils import *
from eval import get_thresholds_for_max_f1_and_lift



class CatBoostBinaryClassifier():

     def __init__(
               self, 
               exp_dir: str, 
               cat_features: list, 
               config: dict
     ) -> None:
          
          '''
          Description:
               Initialises a CatBoost binary classifier based on params in the Config.
          Args:
               exp_dir: Experimental directory model gets saved or loaded from.
               cat_features: List of categorical features in the model setup.
               config: The loaded config file for the experiment.
          Returns:
               None.
          '''

          # experimental directory
          self.exp_dir = exp_dir

          # list of categorical features
          self.cat_features = cat_features
          
          # config file corresponding to the experiment
          self.config = config

          # initialise model
          self.model = CatBoostClassifier(
               train_dir=self.exp_dir, 
               cat_features=self.cat_features,
               **self.config.model.params.fixed
          )

          # initialise best params
          self.best_params = self.config.model.params.fixed

          # initialise dictionary to save parameters for each trial if tuning
          self.trial_params = {}
          
          
     def fit(
               self, 
               x_train: pd.DataFrame, 
               y_train: pd.Series, 
               x_val: pd.DataFrame, 
               y_val: pd.Series,
     ) -> None:

          '''
          Description:
               1. Tunes if tuning in config, and fits the initialised CatBoost model.
               2. Trains an isotonic regression model for calibration. 
               3. Saves both the models in the experimental directory.
          Args:
               x_train: The training dataframe containing features.
               y_train: The training Label column.
               x_val: The validation dataframe containing features.
               y_val: The validation Label column.
          Returns:
               None. Saves the models in the training directory.
          '''

          # tune model if parameters for tuning given in config
          if len(self.config.model.params.keys()) > 1:
               self.tune(x_train, y_train, x_val, y_val)
               
               # save trial parameters dictionary as json in different lines
               with open(f'{self.exp_dir}/trial_params.json', 'w') as f:
                    json.dump(self.trial_params, f, indent=5)

          # fit model with best parameters
          self.model.fit(X=x_train, y=y_train, eval_set=(x_val, y_val), verbose=True)

          # raw prediction confidence scores
          y_val_score = self.model.predict_proba(x_val)[:, 1]

          # save pre_calibration validation scores
          df_val = pd.DataFrame.from_dict({'preds_proba_0': 1 - y_val_score, 'preds_proba_1': y_val_score, 'target': y_val})
          df_val.to_csv(f"{self.exp_dir}/val_precalibration_confidence_scores.csv", index=False)

          # initialise calibration model 
          ir = IsotonicRegression(out_of_bounds="clip")
          
          # get calibration points to train calibration model on val set
          prob_true, prob_pred = calibration_curve(y_true=y_val, y_prob=y_val_score, pos_label=1, n_bins=self.config.model.calibration_nbins)
          
          # train calibration model
          ir.fit(prob_pred, prob_true)
          with open(f'{self.exp_dir}/isotonic_regression.pkl','wb') as f:
               pickle.dump(ir, f)
          
          # obtain calibrated confidence scores to calculate threshold
          y_val_score = ir.transform(y_val_score)

          # save post_calibration validation scores
          df_val = pd.DataFrame.from_dict({'preds_proba_0': 1 - y_val_score, 'preds_proba_1': y_val_score, 'target': y_val})
          df_val.to_csv(f"{self.exp_dir}/val_postcalibration_confidence_scores.csv", index=False)

          # get optimal thresholds for max f1 score and max lift
          threshold_max_f1, threshold_max_lift = get_thresholds_for_max_f1_and_lift(
               df=df_val, 
               preds_proba_1_col="preds_proba_1", 
               preds_proba_0_col="preds_proba_0", 
               target_col="target"
          )
               
          # pass the threshold as model's attribute, only to save it 
          self.model.set_probability_threshold(threshold_max_f1)

          # save optimal threshold as attribute
          self.optimal_threshold = self.model.get_probability_threshold()

          # save model
          self.model.save_model(f'{self.exp_dir}/model.cbm')

          # return best parameters
          return self.best_params, threshold_max_f1, threshold_max_lift


     def predict(
               self, 
               x: pd.DataFrame, 
               features: list
     ) -> pd.DataFrame:
          
          '''
          Description:
               Computes predictions using obtained confidence scores, calibration and threshold.
          Args:
               x: Input dataframe to predict on.
               features: Relevant columns in input dataframe to be used by model for predictions.
          Returns:
               Returns dataframe with added columns showing confidence scores and predictions.
          '''

          # load catboost model
          self.model.load_model(f'{self.exp_dir}/model.cbm')

          # raw confidence scores
          confidence_scores = self.model.predict_proba(x[features])

          # retrieve optimal threshold 
          self.optimal_threshold = self.model.get_probability_threshold()

          # define calibration model outside conditional scope for global use
          ir = IsotonicRegression(out_of_bounds="clip")

          # load calibration model
          with open(f'{self.exp_dir}/isotonic_regression.pkl','rb') as f:
               ir = pickle.load(f)
          
          # obtain calibrated confidence scores
          confidence_scores[:, 1] = ir.transform(confidence_scores[:, 1])
          confidence_scores[:, 0] = 1 - confidence_scores[:, 1]
          
          # obtain predictions
          predictions = np.select(condlist=[(confidence_scores[:, 1] >= self.optimal_threshold)], choicelist=[1], default=0)

          # generate output dataframe
          df_output = pd.DataFrame.from_dict({'preds_proba_0': confidence_scores[:, 0], 'preds_proba_1': confidence_scores[:, 1], 'preds': predictions})

          # concatenate results and return dataframe
          return pd.concat([x, df_output], axis=1)
     

     def tune(
               self, 
               x_train: pd.DataFrame, 
               y_train: pd.Series, 
               x_val: pd.DataFrame, 
               y_val: pd.Series
     ) -> None:
                    
          '''
          Description:
               Tunes the model based on parametric space given in config.
          Args:
               x_train: The training dataframe containing features.
               y_train: The training Label column.
               x_val: The validation dataframe containing features.
               y_val: The validation Label column.
          Returns:
               None. Reinitialises the model with the best obtained parameters.
          '''
          
          def objective(trial):

               parameters = {}
               parameters["train_dir"] = self.exp_dir
               parameters["cat_features"] = self.cat_features

               # iterate over all fixed parameters in the config file
               for key, value in self.config.model.params.fixed.items():
                    parameters[key] = value

               # iterate over all independent parameters to tune
               for key, value in self.config.model.params.tune.independent.items():                    
                    # get suggestions for each parameter
                    parameters[key] = get_optuna_suggestions(
                         trial=trial, 
                         param_name=key, 
                         dtype=value["dtype"], 
                         tuning_space=value["tuning_space"]
                    )

               # iterate over all dependent parameters to tune
               for key, value in self.config.model.params.tune.dependent.items():                    
                    # check for existence of dependent parameter
                    if parameters[value["dependent_on_param"]] in value["dependent_on_value"]:
                         # get suggestions for each parameter
                         parameters[key] = get_optuna_suggestions(
                              trial=trial, 
                              param_name=key, 
                              dtype=value["dtype"], 
                              tuning_space=value["tuning_space"]
                         )

               # save parameters for each trial in a dictionary trial-wise
               self.trial_params[trial.number] = parameters

               # address device compatibility issues
               parameters = address_device_compatibility(parameters)
               
               # initialise model with current set of parameters
               model = CatBoostClassifier(**filter_valid_params(parameters))

               # fit model with current set of parameters
               model.fit(x_train, y_train)

               # confidence scores of minority class 
               preds_proba_1 = model.predict_proba(x_val)[:, 1]

               # points on PR curve
               precision, recall, thresholds = precision_recall_curve(
                    y_true=np.array(y_val), 
                    probas_pred=np.array(preds_proba_1)
               )
               
               # metric to optimise
               ap = metrics.auc(recall, precision)

               # return metric that would guide optimisation
               return ap

          # initialise study 
          study = optuna.create_study(direction="maximize")

          # commence optimisation
          study.optimize(objective, n_trials=self.config.model.n_trials)

          # store best parameters
          self.best_params = study.best_params

          # reinitialise model with best parameters
          self.model = CatBoostClassifier(
               train_dir=self.exp_dir, 
               cat_features=self.cat_features, 
               **study.best_params
          )



def get_optuna_suggestions(
     trial: optuna.trial.Trial,
     param_name: str,
     dtype: str,
     tuning_space: dict, 
):
     
     '''
     Description:
          Suggests parameters for tuning based on the type of parameter.
     Args:
          trial: The optuna trial object.
          param_name: The name of the parameter to tune.
          dtype: The data type of the parameter.
          tuning_space: The tuning space for the parameter.
     Returns:
          Suggested parameter for tuning
     '''

     # alter "none" to None in tuning space due to json / toml constraints
     tuning_space = replace_value_in_nested_dict(
          d=tuning_space, 
          target="None", 
          replacement=None
     )

     # suggest parameters based on dtype
     if dtype == "int":
          return trial.suggest_int(name=param_name, **tuning_space)
     elif dtype == "float":
          return trial.suggest_float(name=param_name, **tuning_space)
     elif dtype == "categorical":
          return trial.suggest_categorical(name=param_name, **tuning_space)
     else:
          raise ValueError(f"Invalid dtype: {dtype}")
     



def address_device_compatibility(parameters: dict) :

     '''
     Description:
          Addresses device compatibility issues for CatBoost.

     Args:
          parameters: All CatBoost parameters that have to be checked for cpu / gpu compatibility.
     Returns:
          "task_type": "gpu" and "device" if gpu compatible, else "cpu".
     '''

     # conditions for CPU usage
     conditions = [
          "random_strength" in parameters and parameters["random_strength"] != 1,          
          "rsm" in parameters and parameters["rsm"] != 1,          
          "diffusion_temperature" in parameters and parameters["diffusion_temperature"] != 10000,
          "sampling_frequency" in parameters and parameters["sampling_frequency"] != "PerTreeLevel",
          "approx_on_full_history" in parameters and parameters["approx_on_full_history"] != False,
          "langevin" in parameters and parameters["langevin"] != False
     ]

     # if any of the conditions are met, then use CPU
     if any(conditions):
          parameters["task_type"] = "CPU"
          parameters.pop("device", None)
     
     # return parameters with updated device type
     return parameters



def loss_curves(train_dir: str):

     '''
     Description:
          Plots the loss curves for training and validation data.
     Args:
          train_dir: The directory where the training data is.
     Returns:
          A matplotlib figure showing the loss curves.
     '''

     # load loss curves
     with open(f"{train_dir}/catboost_training.json") as json_file:
          df_losscurves = pd.DataFrame(json.load(json_file)['iterations'])
     
     # rename columns
     df_losscurves.rename(columns={'learn': 'trn', 'test': 'val'}, inplace=True)

     # get iterations
     iterations = list(df_losscurves['iteration'])

     # initialise figure
     fig, ax = plt.subplots(figsize=(9, 6))
     
     # loop over all training and validation losses
     for ds_name in ['trn', 'val']:

          # plot loss curve
          ax.plot(iterations, [loss[0] for loss in list(df_losscurves[ds_name])], label=ds_name)

     # set labels
     ax.set_xlabel('Iterations')
     ax.set_ylabel('Loss')
     ax.set_title('Loss Curves')
     ax.legend()
     ax.grid(True)

     # save figure
     plt.savefig(f"{train_dir}/loss_curves.png")

     # return figure
     return fig



class auc_pr(object):

     '''
     Description:
          Defines the custom eval metric for CatBoost to use to counter overfitting.
          For more info, read CatBoost documentation on overfitting.
     '''

     def is_max_optimal(self):

          # Returns whether great values of metric are better
          True

     def evaluate(self, approxes, target, weight):

          # approxes is a list of indexed containers
          # (containers with only __len__ and __getitem__ defined),
          # one container per approx dimension.
          # Each container contains floats.
          # weight is a one dimensional indexed container.
          # target is a one dimensional indexed container.

          confidence_scores = approxes[0]
          
          # precision recall curve
          ps, rs, ts = precision_recall_curve(
               y_true=target, 
               probas_pred=confidence_scores
          )

          # area under precision recall curve
          ap = metrics.auc(rs, ps)
          
          weight_sum = 1
          # weight parameter can be None.
          # Returns pair (error, weights sum)

          return (ap, weight_sum)

     def get_final_error(self, error, weight):
          # Returns final value of metric based on error and weight
          return error