import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, precision_score, recall_score, confusion_matrix

from utils import *



def metric_plots(
          df: pd.DataFrame,
          grade: int,
          acad_year: str,
          save_dir: str,
          ds_name: str,
          students_options: dict=None,
          students_to_discard: set=None,
          index: str="aadhaaruid",
          label: str="target",
          baseline_params: dict=None,
          preds_proba_0: str="preds_proba_0",
          preds_proba_1: str="preds_proba_1",
          preds: str="preds",
          thresholds_dictionary: dict=None,
          dpi=300,
          pre_title=""
) -> dict:
     
     '''
     Description:
          Generates evaluation metrics and plots of a model's probability scores on a given dataset.
          Plots include
               1. Precision vs Recall
               2. Recall @ K
               3. ROC
               4. Calibration
               5. Probability Score Distribution with labels
               6. Error Score Distribution
               7. PPV & NPV vs Threshold
               8. Dropout Rate vs Threshold
               9. Probability Score Distribution
     Args:
          df: dataframe containing the probability scores, predictions and targets.
          grade: grade of the dataset.
          acad_year: academic year of the dataset.
          save_dir: directory to store the evaluation plots.
          ds_name: name of the dataset.
          students_options: dictionary containing various descriptions as keys and lists of student IDs as values.
          students_to_discard: set of students (their student IDs) to discard from the dataset for partial evaluations.
          index: name of the index column.
          label: name of the label column.
          baseline_params: dictionary containing parameters for baselines, such as periods of absence and consideration.
          preds_proba_0: name of the column containing the probability score for class 0.
          preds_proba_1: name of the column containing the probability score for class 1.
          preds: name of the column containing the predictions.
          thresholds_dictionary: dictionary containing thresholds as values and their names as keys.
          dpi: resolution of the plots.
          pre_title: prefix to add to the title of the plots.
     Returns:
          perf_metrics: dictionary containing the evaluation metrics.
     '''

     # initialise thresholds_dictionary
     if thresholds_dictionary is None:
          thresholds_dictionary = {}

     # initialise metrics
     perf_metrics = {}

     # set pre_title
     if pre_title:
          pre_title = f"{pre_title}\n"

     # make save_dir if it does not exist
     os.makedirs(save_dir, exist_ok=True)

     # discard students
     if students_to_discard:
          df = df[~df[index].isin(students_to_discard)]

     # correct datatypes
     df[preds_proba_1] = df[preds_proba_1].astype(np.float64)
     df[preds_proba_0] = df[preds_proba_0].astype(np.float64)
     if label in df.columns:
          df[label] = df[label].astype("int64")
     if preds in df.columns:
          df[preds] = df[preds].astype("int64")

     # common plot aspects
     plt.figure(figsize=(7,6))
     threshold_marker_size = 30

     # plot only if label is present
     if label in df.columns:
               
          # points for precision recall curve, and threshold vs recall curve
          ps, rs, ts = precision_recall_curve(
               y_true=df[label], 
               probas_pred=df[preds_proba_1]
          )
          
          # area under precision recall curve
          ap = round(metrics.auc(rs, ps),3)

          # threshold for max f1 score on current dataset
          f1_scores = 2 * ps * rs / (ps + rs + 1e-10)
          max_f1 = np.max(f1_scores)
          max_f1_threshold = ts[np.argmax(f1_scores)]
          max_f1_threshold_name = f"max_f1: {round(max_f1,3)}"
          thresholds_dictionary[max_f1_threshold_name] = max_f1_threshold

          # points for recall @ k curve
          lifts, recalls, ks = k_recall_curve(
               data=df, 
               label_col=label, 
               num_points=101
          )

          # threshold for max lift on current dataset
          try:
               recall_max_lift = recalls[np.argmax(lifts)]
               max_lift = np.max(lifts)
               max_lift_threshold = ts[np.argmin(abs(recall_max_lift-rs[:-1]))]
               max_lift_threshold_name = f"max_lift: {round(max_lift,3)}"
               thresholds_dictionary[max_lift_threshold_name] = max_lift_threshold
          except:
               dummy = None

          # points for roc curve
          fpr, tpr, _ = roc_curve(
               y_true=df[label], 
               y_score=df[preds_proba_1]
          )

          # points for calibration curve
          prob_true, prob_pred = calibration_curve(
               y_true=df[label], 
               y_prob=df[preds_proba_1], 
               n_bins=20, 
               pos_label=1,
               strategy='uniform'
          )

          # plot precision recall curve
          plt.scatter(rs, ps, s=5, label=f"model [ap: {ap}]")
          plt.xlabel('Recall')
          plt.ylabel('Precision')
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          title = f"{ds_name}: Precision vs Recall (label=1)"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        

          # add thresholds_dictionary to precision recall curve
          for threshold_name, threshold in thresholds_dictionary.items():
               x = recall_score(df[label], df[preds_proba_1] >= threshold)
               y = precision_score(df[label], df[preds_proba_1] >= threshold)
               plt.scatter(x, y, s=threshold_marker_size, label=threshold_name)
          
          # add baselines to precision recall curve if given
          if baseline_params and students_options:
               baselines = compute_baseline(
                    df=df.copy(deep=True), 
                    grade=grade, 
                    acad_year=acad_year, 
                    students_options=students_options, 
                    index=index,
                    label=label,
                    baseline_params=baseline_params
               )
               markers, marker_index = baseline_params["markers"], 0
               for key, baseline in baselines.items():
                    plt.scatter(baseline['recall'], baseline['precision'], s=threshold_marker_size, label=f"B [{key}]", c='black', marker=markers[marker_index])
                    marker_index += 1
          
          # legend, grid, save, clear
          plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
          plt.tight_layout()
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # plot threshold vs recall curve
          plt.scatter(rs[:-1], ts, s=5, label=f"model")
          plt.xlabel('Recall')
          plt.ylabel('Threshold')
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          title = f"{ds_name}: Threshold vs Recall (label=1)"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        

          # legend, grid, save, clear
          plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
          plt.tight_layout()
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # plot recall @ k curve
          plt.scatter(ks, recalls, s=5, label=f"model [K_ml: {round(ks[np.argmax(lifts)], 2)}]")
          plt.plot([0,1], [0,1], linestyle='--', color='black', label="random")
          plt.xlabel('K (fraction of students targeted)')
          plt.ylabel('Recall')
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          title = f"{ds_name}: Recall @ K (label=1)"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        

          # add thresholds_dictionary to recall @ k curve
          for threshold_name, threshold in thresholds_dictionary.items():
               r = recall_score(df[label], df[preds_proba_1] >= threshold)
               x = ks[np.argmin(abs(r-recalls))]
               y = recalls[np.argmin(abs(r-recalls))]
               plt.scatter(x, y, s=threshold_marker_size, label=threshold_name)

          # legend, grid, save, clear
          plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
          plt.tight_layout()
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # plot roc curve
          plt.scatter(fpr, tpr, s=5, label=f"model [auc: {round(metrics.auc(fpr, tpr),3)}]")
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          title = f"{ds_name}: ROC"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        
          plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
          plt.tight_layout()
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # plot calibration
          plt.plot(prob_pred, prob_true, marker='o', label='model calibration')
          plt.plot([0,1], [0,1], linestyle='--', color='black', label="perfect calibration")
          plt.xlabel('mean predicted probability')
          plt.ylabel('True positive rate')
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          title = f"{ds_name}: Calibration"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        
          plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
          plt.tight_layout()
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # histogram of confidence scores for both labels using hue in seaborn
          sns.histplot(data=df, x=preds_proba_1, hue=label, bins=50)
          plt.xlabel('Probability Score')
          plt.ylabel('Count')
          plt.xlim(-0.05, 1.05)
          plt.yscale('log')
          plt.ylim(1, 1e6)
          title = f"{ds_name}: Probability Score Distribution with labels"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # histogram of error scores for both labels using hue in seaborn
          df["error"] = compute_error_scores(df=df, label=label, preds_proba_1=preds_proba_1, preds_proba_0=preds_proba_0)
          sns.histplot(data=df, x="error", hue=label, bins=50)
          plt.xlabel('Error Score')
          plt.ylabel('Count')
          plt.xlim(-1.05, 1.05)
          plt.yscale('log')
          plt.ylim(1, 1e6)
          title = f"{ds_name}: Error Score Distribution"
          plt.title(f"{pre_title}{title}\n(Probability Score - Label)\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # points of positive predictive value and negative predictive value vs threshold
          ppvs = []
          npvs = []
          thresholds = np.linspace(0,1,100)

          for threshold in thresholds:

               # add predictions based on threshold
               df = add_preds_threshold(df=df, preds_proba_col=preds_proba_1, threshold=threshold, preds_col='preds_threshold')

               # calculate confusion matrix
               cm = confusion_matrix(np.array(df[label]), np.array(df['preds_threshold']))               
               tp, tn, fp, fn = cm[1,1], cm[0,0], cm[0,1], cm[1,0]

               # calculate ppv and npv
               ppvs.append(round(tp/(tp+fp+1e-10),2))
               npvs.append(round(tn/(tn+fn+1e-10),2))

          # plot ppv and npv vs threshold
          plt.scatter(thresholds, ppvs, s=5, label='ppv')
          plt.scatter(thresholds, npvs, s=5, label='npv')
          plt.xlabel('Threshold')
          plt.ylabel('PPV & NPV')
          plt.xlim(-0.05, 1.05)
          plt.ylim(-0.05, 1.05)
          title = f"{ds_name}: PPV & NPV vs Threshold"
          plt.title(f"{pre_title}{title}\n[# students: {len(df)}, # dropouts: {df[label].sum()}]")        
          plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
          plt.tight_layout()
          plt.grid()
          plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
          plt.clf()

          # populate perf_metrics
          perf_metrics[f"{ds_name} : ap"] = ap
          perf_metrics[f'{ds_name} : prevalence'] = f"{round(100*df[label].mean(),2)}%"
          perf_metrics[f'{ds_name} : actual not dropout'] = df[label].count() - df[label].sum()
          perf_metrics[f'{ds_name} : actual dropout'] = df[label].sum()

     # points for dropout rate vs threshold
     dropout_rates = []
     thresholds = np.linspace(0,1,100)

     for threshold in thresholds:
          
          # add predictions based on threshold
          df = add_preds_threshold(df=df, preds_proba_col=preds_proba_1, threshold=threshold)
          
          # calculate dropout rate
          dropout_rates.append(df['preds_threshold'].mean())
     
     # plot dropout rate vs threshold
     plt.scatter(thresholds, dropout_rates, s=5, label=f"model")
     plt.xlabel('Threshold')
     plt.ylabel('Dropout Rate')
     plt.xlim(-0.05, 1.05)
     plt.ylim(1e-6 - 6*(1e-7 - 1e-8), 2)
     plt.yscale('log')
     title = f"{ds_name}: Dropout Rate vs Threshold"
     plt.title(f"{pre_title}{title}\n[# students: {len(df)}]")        
     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
     plt.tight_layout()
     plt.grid()
     plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
     plt.clf()

     # histogram of confidence scores without label information
     sns.histplot(data=df, x=preds_proba_1, bins=50)
     plt.xlabel('Probability Score')
     plt.ylabel('Count')
     plt.xlim(-0.05, 1.05)
     plt.yscale('log')
     plt.ylim(1, 1e6)
     title = f"{ds_name}: Probability Score Distribution without labels"
     plt.title(f"{pre_title}{title}\n[# students: {len(df)}]")        
     plt.grid()
     plt.savefig(f"{save_dir}/{title}.png", dpi=dpi, bbox_inches="tight")
     plt.clf()

     # close all active plots
     plt.close('all')
          
     # return perf_metrics
     return perf_metrics



def k_recall_curve(
          data: pd.DataFrame, 
          label_col: str='target', 
          num_points: int=1000, 
          preds_proba_1_col: str='preds_proba_1'
):
     
     '''
     Description:
          Constructs Recall @ k curves for input dataframes with confidence score columns.
     Args:
          data: input pandas dataframe.
          label_col: name of label column.
          num_points: number of k values to compute the recall for and plot.
          preds_proba_1_col: name of column with confidence scores for class 1 (minority class)
     Returns:
          Tuple (of lifts, recalls and ks)
     '''

     # copy of input dataframe
     df = data.copy(deep=True)

     # sort in descending order of confidence scores
     df.sort_values(by=preds_proba_1_col, ascending=False, inplace=True)
     
     # number of records in dataframe
     total = len(df)
     
     # least fraction to increment 
     d = (1-0)/(num_points-1)
     
     # all values of k to compute recall on and plot
     ks = [n*d for n in range(0, num_points)]
     
     # placeholder lists to cache recalls and lifts
     recalls = []
     lifts = []

     # loop over ks to compute corresponding recalls
     for k in ks:
          
          # number of records for a given k
          num_records = int(total * k)
          
          # generating a new (and dummy) prediction column
          df['preds_k'] = [*[1 for i in range(0, num_records)], *[0 for i in range(0, total-num_records)]]
          
          # compute recall score using new dummy prediction column
          recall = recall_score(
               y_true=np.array(df[label_col]),
               y_pred=np.array(df['preds_k']),
               pos_label=1,
               average='binary'
          )          
          
          # append computed recall and k to corresponding lists
          recalls.append(recall)
          lifts.append(recall-k)
     
     return np.array(lifts), np.array(recalls), np.array(ks)



def compute_confmat(df, label: str, pred: str):

     '''
     Description:
          Compute confusion matrix from a dataframe.
     Args:
          df: input dataframe.
          label: name of the column with true labels.
          pred: name of the column with predictions.
     Returns:
          Indices and counts of true positives, true negatives, false positives and false negatives.
     '''

     # compute confusion matrix
     cf = confusion_matrix(np.array(df[label]), np.array(df[pred]))

     # compute count of true positives, true negatives, false positives and false negatives
     tp_count, tn_count, fp_count, fn_count = cf[1,1], cf[0,0], cf[0,1], cf[1,0]

     # labels in dataframe
     labels = list(df[label].unique())
     labels.sort()

     # true positives indices in dataframe
     tp_indices = list(df.index[(df[label] == labels[1]) & (df[pred] == labels[1])])

     # true negatives indices in dataframe
     tn_indices = list(df.index[(df[label] == labels[0]) & (df[pred] == labels[0])])

     # false positive indices in dataframe
     fp_indices = list(df.index[(df[label] == labels[0]) & (df[pred] == labels[1])])

     # false negatives indices in dataframe
     fn_indices = list(df.index[(df[label] == labels[1]) & (df[pred] == labels[0])])

     # validate length of indices and counts
     assert len(tp_indices) == tp_count

     return {
          "tp_count": int(tp_count), "tn_count": int(tn_count), "fp_count": int(fp_count), "fn_count": int(fn_count), 
          "tp_indices": tp_indices, "tn_indices": tn_indices, "fp_indices": fp_indices, "fn_indices": fn_indices,
          "precision": round(tp_count/(tp_count+fp_count+1e-10),2), "recall": round(tp_count/(tp_count+fn_count+1e-10),2)
     }



def compute_baseline(
          df: pd.DataFrame, 
          grade: int,
          acad_year: str,
          students_options: dict,
          index: str="aadhaaruid",
          label: str="target",
          baseline_params: dict={},
) -> dict:
     
     '''
     Description:
          Compute baseline metrics for a given dataset.
          In general, label 0 refers to students who are not dropouts, and label 1 refers to students who are dropouts. 
          The same convention is also followed in the variable names and keys used in this function.
     Args:
          df: input dataset.
          grade: grade of the input dataset.
          acad_year: academic year of the input dataset.
          students_options: dictionary containing various options for students.
          index: name of the column with unique student identifiers.
          label: name of the column with true labels.
          baseline_params: dictionary containing periods of absence and consideration to compute baselines.
     Returns:          
     '''

     # validate baseline_params dictionary
     if baseline_params:
          assert "poas" in baseline_params # poas is periods of absenteeism
          assert "pocs" in baseline_params # poas is periods of consideration

     # initialise baselines dictionary
     baselines = {}

     # for the given df, grade, acad_year, poas, and pocs, get list of students with attendance labels 0 and 1 for each combination.
     for poc in baseline_params["pocs"]:
          for poa in baseline_params["poas"]:
               attnd_0_students = f"ay{acad_year}, grade{grade}, attnd_label[{poc}, poa:{poa}]=0"
               attnd_1_students = f"ay{acad_year}, grade{grade}, attnd_label[{poc}, poa:{poa}]=1"

               # do a number of checks
               assert attnd_0_students in students_options
               assert attnd_1_students in students_options
               assert len(students_options[attnd_0_students].intersection(students_options[attnd_1_students])) == 0
               assert set(df[index]).issubset(students_options[attnd_0_students].union(students_options[attnd_1_students]))

               # append the attendance label column to the dataframe to compute metrics using compute_confmat
               df["attnd_label"] = np.where(df[index].isin(students_options[attnd_0_students]), 0, 1)

               # append the baseline to baselines dictionary
               key = f"{poc}, poa:{poa}"
               assert key not in baselines
               baselines[key] = compute_confmat(df, label=label, pred="attnd_label")

     # return baselines dictionary
     return baselines



def compute_error_scores(
          df: pd.DataFrame,
          label: str="target",
          preds_proba_1: str="preds_proba_1",
          preds_proba_0: str="preds_proba_0"
) -> np.array:
     
     '''
     Description:
          Compute error scores for a given dataset.
     Args:
          df: input dataset.
          label: name of the column with true labels.
          preds_proba_1: name of the column with probability scores for class 1.
          preds_proba_0: name of the column with probability scores for class 0.
     Returns:
          numpy array of error scores.
     '''

     # arrays of true labels and confidence scores
     labels = np.array(df[label])
     preds_proba_1 = np.array(df[preds_proba_1])
     preds_proba_0 = np.array(df[preds_proba_0])

     # error scores are confidence score of a class - true label using np.where
     error_scores = np.where(labels == 1, preds_proba_1, preds_proba_0) - labels

     # return error scores
     return error_scores



def compute_specificity(
          precision: float, 
          recall: float, 
          n_dropouts: int, 
          n_students: int
):
     
     '''
     Description:
          Compute specificity for a given dataset.
     Args;
          precision: precision score.
          recall: recall score.
          n_dropouts: number of dropouts in the dataset.
          n_students: total number of students in the dataset.
     Returns:
          specificity: specificity score.
     '''

     # true positives, false negatives, false positives and true negatives
     tp = round(n_dropouts * recall)
     fn = n_dropouts - tp
     fp = round(tp * (1 - precision) / precision + 1e-10)
     tn = n_students - tp - fp - fn
     
     # compute specificity
     specificity = round(tn / (tn + fp),2)

     # return specificity
     return specificity



def get_thresholds_for_max_f1_and_lift(
          df,
          preds_proba_1_col: str,
          preds_proba_0_col: str,
          target_col: str
):
     """
     Compute thresholds for max F1 and max Lift given a DataFrame and column names.

     Parameters
     ----------
     df : pd.DataFrame
          DataFrame containing prediction probabilities and true labels.
     preds_proba_1_col : str
          Column name for predicted probabilities for class 1.
     preds_proba_0_col : str
          Column name for predicted probabilities for class 0. (Not used, but kept for interface completeness.)
     target_col : str
          Column name for true binary labels.

     Returns
     -------
     dict
          {
               'threshold_max_f1': float,
               'threshold_max_lift': float or None
          }
     """
     y_val_score = df[preds_proba_1_col].values
     y_val = df[target_col].values

     # Precision-recall curve and F1 computation
     ps, rs, ts = precision_recall_curve(y_true=y_val, probas_pred=y_val_score)
     f1s = 2 * rs * ps / (rs + ps + 1e-10)
     threshold_max_f1 = ts[np.argmax(f1s)]

     # Recall@k curve
     lifts, recalls, ks = k_recall_curve(
          data=df,
          label_col=target_col,
          num_points=1000,
          preds_proba_1_col=preds_proba_1_col
     )
     recall_max_lift = recalls[np.argmax(lifts)]

     try:
          threshold_max_lift = ts[np.argmin(abs(recall_max_lift - rs[:-1]))]
     except:
          threshold_max_lift = None

     return threshold_max_f1, threshold_max_lift
