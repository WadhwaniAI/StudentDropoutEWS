import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


class IsotonicCalibrator:
     """Wraps isotonic regression for calibration."""

     def __init__(self, path: str):
          """Initialize calibrator with path to save/load model.          
          :param path (str): File path to save or load the calibration model.
          """
          self.path = path
          self.ir = IsotonicRegression(out_of_bounds="clip")

     def fit_transform(self, y_true, y_pred, n_bins: int):
          """Fits isotonic regression, saves the model and returns calibrated scores.          
          :param y_true (array-like): True binary labels.
          :param y_pred (array-like): Uncalibrated predicted probabilities.
          :param n_bins (int): Number of bins for calibration curve.
          :returns: Calibrated probabilities after isotonic regression.
          """
          prob_true, prob_pred = calibration_curve(
               y_true=y_true, y_prob=y_pred, pos_label=1, n_bins=n_bins
          )
          self.ir.fit(prob_pred, prob_true)
          with open(self.path, 'wb') as f:
               pickle.dump(self.ir, f)
          return self.ir.transform(y_pred)

     def transform(self, y_pred):
          """Loads the saved model (during inferene) and applies it on raw probabilities.
          :param y_pred (array-like): Uncalibrated predicted probabilities.
          Returns: (array-like) Calibrated probabilities after applying isotonic regression.
          """
          if not hasattr(self.ir, 'X_'):
               with open(self.path, 'rb') as f:
                    self.ir = pickle.load(f)
          return self.ir.transform(y_pred)