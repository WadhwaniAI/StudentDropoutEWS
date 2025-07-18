import argparse
import os
import shutil
import wandb

from catboost_binary_classifier import CatBoostBinaryClassifier, loss_curves
from eval import metric_plots
from drift import *
from dataset import *
from predict import *
from transforms import *
from utils import *



def parse_args():
     
     '''
     Description:
          Function to parse arguments from command line.
     Args:
          None
     Returns:
          args: argparse object
     '''

     # argparse object
     parser = argparse.ArgumentParser()

     # directory holding config file (or files). 
     parser.add_argument('--config_path', type=str)

     # config files must be .toml or .json files. Default is toml. 
     parser.add_argument('--ext', type=str, default='json')

     # path to students_options dictionary, and load students_options pkl file
     parser.add_argument('--students_options', type=str, default=f"None")

     # directory holding config file (or files). 
     parser.add_argument('--n_threads', type=int, default=1)

     # extract arguments
     args = parser.parse_args()

     # check extension of config 
     assert args.ext in ['toml', 'json'], "config file must be either toml or json"
     
     # return arguments
     return args

# parse arguments
args = parse_args()

# This can be used to set the number of threads available for compute
os.environ['OPENBLAS_NUM_THREADS'] = f"{args.n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{args.n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{args.n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{args.n_threads}"

# load pickle file of students_options
students_options_path = resolve_path(args.students_options)
if os.path.exists(students_options_path):
     with open(students_options_path, "rb") as f:
          students_options = pickle.load(f)
else:
    students_options = None

# configs
config_files = get_config_files(config_path=args.config_path, ext=args.ext)

# loop over configs
for config_file in config_files:

     # load config
     config = load_config(config_file)
     
     # experimental directory
     folder = f'{config.exp.title} | {get_timestamp()}'
     exp_dir = f'{config.exp.root_exps}/{folder}'

     # wandb for logging
     wandb.init(project=config.exp.project, config=config, name=folder, notes=config.notes.description, config_exclude_keys=['exp'])

     # instance of dataset class
     dataset = Dataset(exp_dir=exp_dir, config=config, file_path=config.data.trn_file)

     # sample original dataset as per prevalence in config and split into train and validation datasets
     dataset.commence()

     # get numerical and categorical features before transformations
     dataset.get_features(input_df=dataset.dataset['trn'])

     # fit post_split transform on numerical features of training dataset
     transforms = Transforms(
          dataframe=dataset.dataset['trn'],
          num_features=dataset.num_features,
          label=dataset.label,
          feature_groups=dataset.feature_groups,
          exp_dir=exp_dir,
          config_transforms=config.data.post_split_transforms
     )
     transforms.fit()

     # apply post_split transforms on numerical features of trn and val datasets
     for ds_name in ("trn", "val"):
          transforms = Transforms(
               dataframe=dataset.dataset[ds_name],
               num_features=dataset.num_features,
               label=dataset.label,
               feature_groups=dataset.feature_groups,  
               exp_dir=exp_dir,
               config_transforms=config.data.post_split_transforms
          )
          dataset.dataset[ds_name] = transforms.apply()

     # update numerical and categorical features post transformations if any
     dataset.get_features(input_df=dataset.dataset['trn'])

     # cache categorical and numerical features
     cat_features = dataset.cat_features
     num_features = dataset.num_features
     features = cat_features + num_features

     # save categorical and numerical features
     save_model_features(exp_dir=exp_dir, cat_features=cat_features, num_features=num_features)
     
     # log onto wandb
     print(f"shape of training data: {dataset.dataset['trn'][dataset.features].shape}")
     wandb.log(
          {
               "shape of trn data": dataset.dataset['trn'][dataset.features].shape,
               "categorical features" : dataset.cat_features,
               "numerical features" : dataset.num_features
          }
     )

     # initialise catboost binary classifier
     cb_bin_classifier = CatBoostBinaryClassifier(exp_dir=exp_dir, cat_features=dataset.cat_features, config=config)

     # fit model
     best_params, val_threshold_max_f1, val_threshold_max_lift = cb_bin_classifier.fit(
          x_train=dataset.dataset['trn'][dataset.features],
          y_train=dataset.dataset['trn'][dataset.label],
          x_val=dataset.dataset['val'][dataset.features],
          y_val=dataset.dataset['val'][dataset.label]
     )

     # dict to cache datasets for computation inter-dataset drifts
     drift_dfs = {}

     # predict and evaluate on training, validation and full datasets
     for ds_name in ("trn", "val"):
          
          # obtain dataframe with confidence scores and predictions
          dataset.dataset[ds_name] = cb_bin_classifier.predict(x=dataset.dataset[ds_name], features=dataset.features)

          # aggregate metrics and plots on results of all datasets
          perf_metrics = metric_plots(
               df=dataset.dataset[ds_name],
               grade=dataset.grade,
               acad_year=dataset.acad_year,
               save_dir=exp_dir,
               ds_name=ds_name,
               students_options=students_options,
               students_to_discard=None,
               index=dataset.index,
               label=dataset.label,
               baseline_params=config.data.baseline_params,
               preds_proba_0='preds_proba_0',
               preds_proba_1='preds_proba_1',
               preds='preds',               
               thresholds_dictionary={
                    'max_f1 (val)' : val_threshold_max_f1, 
                    'max_lift (val)' : val_threshold_max_lift
               },
          )

          # cache dataframe
          drift_dfs[ds_name] = dataset.dataset[ds_name]

          # save results
          if ds_name in config.data.datasets_to_save:
               dataset.dataset[ds_name].to_pickle(f'{exp_dir}/{ds_name}.pkl')

          # log metrics
          wandb.log(perf_metrics)

     # predict on corresponding test sets
     for infer_file in config.data.tst_file:
          
          # predict on test set, this adds the probability scores and predictions to the dataframe
          dataset = predict_on_raw_data(exp_dir=exp_dir, config=config, file_path=infer_file)

          # derive dataset name from test csv string
          ds_name = infer_file.split(' | ')[-1][:-4]

          # aggregate metrics and plots on results of all datasets
          perf_metrics = metric_plots(
               df=dataset.df,
               grade=dataset.grade,
               acad_year=dataset.acad_year,
               save_dir=exp_dir,
               ds_name=ds_name,
               students_options=students_options,
               students_to_discard=None,
               index=dataset.index,
               label=dataset.label,
               baseline_params=config.data.baseline_params,
               preds_proba_0='preds_proba_0',
               preds_proba_1='preds_proba_1',
               preds='preds',
               thresholds_dictionary={
                    'max_f1 (val)' : val_threshold_max_f1, 
                    'max_lift (val)' : val_threshold_max_lift
               },
          )

          # cache dataset
          drift_dfs[ds_name] = dataset.df

          # save results
          if ds_name in config.data.datasets_to_save:
               dataset.df.to_pickle(f"{exp_dir}/{ds_name}.pkl")

          # log metrics
          wandb.log(perf_metrics)

     # log onto wandb
     wandb.log(
          {
               "train_val_loss_curves" : loss_curves(train_dir=exp_dir),
               "best_params" : best_params,
               "val_threshold_max_f1" : val_threshold_max_f1,
               "val_threshold_max_lift" : val_threshold_max_lift
          }
     )
     
     # store config in exp directory
     shutil.copyfile(config_file, os.path.join(exp_dir, os.path.basename(config_file)))
     
     # list of tuples of ds_names to compare drift between pairs of datasets
     ds_names_for_drift = []
     for file in config.data.tst_file:
          ds_names_for_drift.append(("trn", file.split(' | ')[-1][:-4]))
          ds_names_for_drift.append(("val", file.split(' | ')[-1][:-4]))

     for ref_name, cur_name in ds_names_for_drift:
          get_drift(
               ref_df=drift_dfs[ref_name],
               cur_df=drift_dfs[cur_name],
               ref_name=ref_name,
               cur_name=cur_name,
               features=features,
               save_dir=exp_dir,
          )

     # end current run
     wandb.finish() 