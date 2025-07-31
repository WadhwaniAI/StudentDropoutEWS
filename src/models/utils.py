import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import recall_score, precision_recall_curve
from src.utils import replace_value_in_nested_dict


def get_optuna_suggestions(trial, param_name, dtype, tuning_space):
     """Suggests a parameter using Optuna based on dtype."""
     tuning_space = replace_value_in_nested_dict(tuning_space, "None", None)
     funcs = {
          "int": trial.suggest_int,
          "float": trial.suggest_float,
          "categorical": trial.suggest_categorical,
     }
     if dtype not in funcs:
          raise ValueError(f"Invalid dtype: {dtype}")
     return funcs[dtype](param_name, **tuning_space)
     

def address_device_compatibility(params: dict):
     """Ensures CatBoost uses CPU if GPU-incompatible settings are found."""
     incompatible = {
          "random_strength": 1,
          "rsm": 1,
          "diffusion_temperature": 10000,
          "sampling_frequency": "PerTreeLevel",
          "approx_on_full_history": False,
          "langevin": False,
     }
     if any(k in params and params[k] != v for k, v in incompatible.items()):
          params["task_type"] = "CPU"
          params.pop("device", None)
     return params


def loss_curves(train_dir: str):
     """Plots and saves training/validation loss curves."""
     with open(f"{train_dir}/catboost_training.json") as f:
          df = pd.DataFrame(json.load(f)['iterations']).rename(columns={'learn': 'trn', 'test': 'val'})

     fig, ax = plt.subplots(figsize=(9, 6))
     for ds in ['trn', 'val']:
          ax.plot(df['iteration'], [loss[0] for loss in df[ds]], label=ds)

     ax.set(xlabel='Iterations', ylabel='Loss', title='Loss Curves')
     ax.legend(); ax.grid(True)
     plt.savefig(f"{train_dir}/loss_curves.png")
     return fig


def filter_valid_params(params: dict) -> dict:
     """Keeps only valid CatBoostClassifier init params."""
     valid_keys = set(inspect.signature(CatBoostClassifier.__init__).parameters) - {"self"}
     return {k: v for k, v in params.items() if k in valid_keys}


def get_model_features(dir: str) -> tuple:
     """Loads and returns categorical, numerical, and combined features from a directory."""
     with open(f"{dir}/num_features.pkl", "rb") as f:
          num_features = pickle.load(f)
     with open(f"{dir}/cat_features.pkl", "rb") as f:
          cat_features = pickle.load(f)
     return cat_features, num_features


def save_model_features(exp_dir: str, cat_features: list, num_features: list) -> None:
     """Saves categorical and numerical features as pickle files."""
     with open(f"{exp_dir}/cat_features.pkl", "wb") as f:
          pickle.dump(cat_features, f)
     with open(f"{exp_dir}/num_features.pkl", "wb") as f:
          pickle.dump(num_features, f)


def k_recall_curve(
          data: pd.DataFrame, label_col: str = "target", preds_proba_1_col: str = "preds_proba_1", num_points: int=1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
     """Compute recall and lift over top-k fractions."""
     df = data.sort_values(by=preds_proba_1_col, ascending=False).copy()
     total = len(df)
     ks = np.linspace(0, 1, num_points)
     recalls, lifts = [], []

     for k in ks:
          n = int(total * k)
          preds_k = np.zeros(total)
          preds_k[:n] = 1
          r = recall_score(df[label_col], preds_k)
          recalls.append(r)
          lifts.append(r - k)

     return np.array(lifts), np.array(recalls), ks


def get_optimal_thresholds(
          df: pd.DataFrame, proba_1_col: str, target_col: str, num_points: int=1000
) -> tuple[float, float | None]:
     """Return thresholds for max F1 and max Lift."""
     y_score = df[proba_1_col].values
     y_true = df[target_col].values

     p, r, t = precision_recall_curve(y_true, y_score)
     f1 = 2 * p * r / (p + r + 1e-10)
     thresh_f1 = t[np.argmax(f1)]

     lifts, rs_k, _ = k_recall_curve(
          df,
          label_col=target_col,
          preds_proba_1_col=proba_1_col,
          num_points=num_points
     )
     r_lift = rs_k[np.argmax(lifts)]

     try:
          thresh_lift = t[np.argmin(np.abs(r[:-1] - r_lift))]
     except:
          thresh_lift = None

     return thresh_f1, thresh_lift