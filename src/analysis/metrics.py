import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
     precision_recall_curve,
     roc_curve,
     confusion_matrix,
     precision_score,
     recall_score,
     auc,
     average_precision_score
)
from src import constants


class BinaryModelEvaluator:
     """Summary metrics and plotting for binary classifiers."""

     def __init__(
          self,
          df,
          label_col,
          proba_1_col,
          ds_name="dataset",
          save_dir=None,
          manual_thresholds: dict = None,
     ):
          """
          :param df (pd.DataFrame): DataFrame containing predictions and optionally labels.
          :param label_col (str): Name of the label column.
          :param proba_1_col (str): Name of the column containing probabilities for class 1.
          :param ds_name (str): Name of the dataset for labeling plots.
          :param save_dir (str): Directory to save plots. If None, plots will be shown instead.
          :param manual_thresholds (dict): Optional dictionary of manual thresholds to use in addition to computed ones.
          """
          self.df = df.copy()
          self.label_col = label_col
          self.proba_1_col = proba_1_col
          self.save_dir = save_dir
          self.ds_name = ds_name
          self.manual_thresholds = manual_thresholds or {}

          if save_dir:
               os.makedirs(save_dir, exist_ok=True)

          self.y_score = self.df[self.proba_1_col].astype(float).values

          # Check if the label column exists in the DataFrame
          if self.label_col in self.df.columns:
               self.y_true = self.df[self.label_col].astype(int).values
               assert set(np.unique(self.y_true)).issubset({0, 1}), f"Label column {label_col} must be binary (0 or 1)"
          else:
               # If not present, set y_true to None
               self.y_true = None

          self.thresholds_info = {}
          self.metrics = {}

     def compute_thresholds(self):
          """Compute thresholds for max F1 and max lift."""
          # Guard clause: only run if labels are available
          if self.y_true is None:
               return
          ps, rs, ts = precision_recall_curve(self.y_true, self.y_score)
          f1s = 2 * ps * rs / (ps + rs + 1e-10)
          self.thresholds_info["max_f1"] = ts[np.argmax(f1s)]

          lifts, recalls, _ = self._k_recall_curve()
          recall_max_lift = recalls[np.argmax(lifts)]
          try:
               self.thresholds_info["max_lift"] = ts[np.argmin(abs(recall_max_lift - rs[:-1]))]
          except:
               print(f"[WARN] Could not compute max_lift threshold on {self.ds_name} due to misalignment.")

          self.thresholds_info.update(self.manual_thresholds)

     def _k_recall_curve(
          self,
          num_points=constants.PlotConfig.RECALL_K_POINTS
     ):
          """Compute lift and recall curves for varying k."""
          # Guard clause: only run if labels are available
          if self.y_true is None:
               return np.array([]), np.array([]), np.array([])
          df_sorted = self.df.sort_values(self.proba_1_col, ascending=False)
          total = len(df_sorted)
          ks, recalls, lifts = [], [], []
          num_points = min(total, num_points)
          for n in range(num_points):
               k = n / (num_points - 1)
               cut = int(total * k)
               preds_k = np.zeros(total, dtype=int)
               preds_k[:cut] = 1
               recall = recall_score(df_sorted[self.label_col], preds_k)
               ks.append(k)
               recalls.append(recall)
               lifts.append(recall - k)
          return np.array(lifts), np.array(recalls), np.array(ks)

     def _finalize_plot(self):
          """Standardize plot formatting and force square shape."""
          ax = plt.gca()
          if ax.get_legend_handles_labels()[0]:
               plt.legend(loc="best")
          plt.tight_layout(pad=1.5)
          plt.grid()

     def _save_show_fig(self, key):
          """Save or show the figure."""
          if self.save_dir:
               plt.savefig(
                    os.path.join(self.save_dir, f"{self.ds_name}_{key}.png"),
                    dpi=constants.PlotConfig.DPI,
                    bbox_inches="tight"
               )
          else:
               plt.show()

     def plot_precision_recall(self):
          """Plot precision vs recall."""
          if self.y_true is None:
               return
          ps, rs, ts = precision_recall_curve(self.y_true, self.y_score)
          ap = round(auc(rs, ps), 3)
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          plt.plot(rs, ps, label=f"Model [AP: {ap}]")
          for i, (name, thresh) in enumerate(self.thresholds_info.items()):
               pr = precision_score(self.y_true, self.y_score >= thresh)
               rc = recall_score(self.y_true, self.y_score >= thresh)
               plt.scatter(
                    rc,
                    pr,
                    c=constants.PlotConfig.COLORS[i % len(constants.PlotConfig.COLORS)],
                    s=100,
                    edgecolor="k",
                    label=f"{name}: {round(thresh, 2)}",
               )
          plt.xlabel("Recall")
          plt.ylabel("Precision")
          plt.title(f"{self.ds_name}: Precision-Recall curve")
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.PRECISION_RECALL)
          plt.close()

     def plot_roc(self):
          """Plot ROC curve."""
          if self.y_true is None:
               return
          fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
          auc_val = auc(fpr, tpr)
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          plt.plot(fpr, tpr, label=f"Model [AUC: {auc_val:.3f}]")
          plt.plot([0, 1], [0, 1], "k--")
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          plt.xlabel("False Positive Rate")
          plt.ylabel("True Positive Rate")
          plt.title(f"{self.ds_name}: ROC curve")
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.ROC_CURVE)
          plt.close()

     def plot_calibration(self):
          """Plot calibration curve."""
          if self.y_true is None:
               return
          prob_true, prob_pred = calibration_curve(
               self.y_true,
               self.y_score,
               n_bins=constants.PlotConfig.CALIBRATION_BINS,
               strategy="uniform"
          )
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          plt.plot(prob_pred, prob_true, marker="o", label="Model")
          plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect")
          plt.xlabel("Mean predicted probability")
          plt.ylabel("True positive rate")
          plt.title(f"{self.ds_name}: Calibration curve")
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.CALIBRATION)
          plt.close()

     def plot_proba_distribution(self, with_labels=True):
          """Plot probability score histogram."""
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          # Only use hue if labels are available and requested
          if with_labels and self.y_true is not None:
               sns.histplot(
                    self.df,
                    x=self.proba_1_col,
                    hue=self.label_col,
                    bins=50,
                    legend=True
               )
               plt.title(f"{self.ds_name}: Proba dist with labels")
          else:
               sns.histplot(self.df, x=self.proba_1_col, bins=50)
               plt.title(f"{self.ds_name}: Proba dist (no labels)")
          plt.xlabel("Probability Score")
          plt.ylabel("Count")
          plt.yscale("log")
          self._finalize_plot()
          savefig_key = constants.PlotConfig.Suffixes.PROBA_DIST
          if with_labels and self.y_true is not None:
               savefig_key = constants.PlotConfig.Suffixes.PROBA_DIST_WITH_LABELS
          self._save_show_fig(savefig_key)
          plt.close()

     def plot_error_distribution(self):
          """Plot error scores (proba - label)."""
          if self.y_true is None:
               return
          self.df["error"] = self._compute_error_scores()
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          sns.histplot(
               self.df,
               x="error",
               hue=self.label_col,
               bins=50,
               legend=True
          )
          plt.xlabel("Error Score")
          plt.ylabel("Count")
          plt.yscale("log")
          plt.title(f"{self.ds_name}: Error Score (proba - label)")
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.ERROR_DIST)
          plt.close()

     def plot_ppv_npv_vs_threshold(self):
          """Plot PPV and NPV vs threshold."""
          if self.y_true is None:
               return
          thresholds = np.linspace(0, 1, 100)
          ppvs, npvs = [], []
          for t in thresholds:
               preds = (self.y_score >= t).astype(int)
               tn, fp, fn, tp = confusion_matrix(
                    self.y_true,
                    preds,
                    labels=[0, 1]
               ).ravel()
               ppvs.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
               npvs.append(tn / (tn + fn) if (tn + fn) > 0 else np.nan)
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          plt.plot(thresholds, ppvs, label="PPV")
          plt.plot(thresholds, npvs, label="NPV")
          plt.xlabel("Threshold")
          plt.ylabel("Value")
          plt.title(f"{self.ds_name}: PPV & NPV vs Threshold")
          plt.ylim(0.0, 1.0)
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.PPV_NPV)
          plt.close()

     def plot_dropout_rate_vs_threshold(self):
          """Plot dropout rate vs threshold."""
          thresholds = np.linspace(0, 1, 100)
          rates = [(self.y_score >= t).mean() for t in thresholds]
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          plt.plot(thresholds, rates, label="Model")
          plt.xlabel("Threshold")
          plt.ylabel("Dropout rate")
          plt.yscale("log")
          plt.title(f"{self.ds_name}: Dropout rate vs threshold")
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.DROPOUT_RATE)
          plt.close()

     def plot_recall_at_k(
          self,
          num_points=constants.PlotConfig.RECALL_K_POINTS
     ):
          """Plot recall and lift vs proportion of population (k)."""
          if self.y_true is None:
               return
          lifts, recalls, ks = self._k_recall_curve(num_points=num_points)
          plt.figure(figsize=constants.PlotConfig.FIGURE_SIZE)
          plt.plot(ks, recalls, label="Recall@k")
          plt.plot(ks, lifts, label="Lift@k")
          plt.xlabel("Proportion of population (k)")
          plt.ylabel("Value")
          plt.title(f"{self.ds_name}: Recall@k and Lift@k")
          self._finalize_plot()
          self._save_show_fig(constants.PlotConfig.Suffixes.RECALL_AT_K)
          plt.close()

     def _compute_error_scores(self):
          """Compute error scores from probabilities and labels."""
          if self.y_true is None:
               return None
          return np.where(self.y_true == 1, self.y_score, (1 - self.y_score)) - self.y_true

     def summary_metrics(self):
          """Return summary classification metrics."""
          if self.y_true is None:
               return {}  # Return empty dict if no labels
          if 'max_f1' not in self.thresholds_info:
               self.compute_thresholds()
          thresh = self.thresholds_info.get('max_f1', 0.5)
          preds = (self.y_score >= thresh).astype(int)
          fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
          self.metrics = {
               f'{self.ds_name}_prevalence': self.y_true.mean(),
               f'{self.ds_name}_threshold': thresh,
               f'{self.ds_name}_precision': precision_score(self.y_true, preds),
               f'{self.ds_name}_recall': recall_score(self.y_true, preds),
               f'{self.ds_name}_auc': auc(fpr, tpr),
               f'{self.ds_name}_ap': average_precision_score(self.y_true, self.y_score)
          }
          return self.metrics

     def plot_all(self):
          """Run all plots with threshold computation."""
          # Check for labels before running label-dependent plots
          if self.y_true is not None:
               self.compute_thresholds()
               self.plot_precision_recall()
               self.plot_roc()
               self.plot_calibration()
               self.plot_error_distribution()
               self.plot_ppv_npv_vs_threshold()
               self.plot_recall_at_k()
               # This one can be run with or without labels
               self.plot_proba_distribution(with_labels=True)

          # These plots do not require labels and can always run
          self.plot_proba_distribution(with_labels=False)
          self.plot_dropout_rate_vs_threshold()