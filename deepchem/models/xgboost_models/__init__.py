"""
Use Scikit-learn wrapper interface of xgboost
"""

import xgboost as xgb
import numpy as np
from deepchem.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.save import load_from_disk
from deepchem.utils.save import save_to_disk
from sklearn.cross_validation import train_test_split

class XGBoostModel(SklearnModel):
  """
  Abstract base class for different ML models.
  """

  def fit(self, dataset, **kwargs):
    """
    Fits XGBoost model to data.
    """
    X = dataset.X
    y = np.squeeze(dataset.y)
    w = np.squeeze(dataset.w)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state=0,
                                                        stratify=y)
    #TODO: allow user passing early_stopping_rounds
    self.model_instance.fit(X_train, y_train, early_stopping_rounds=50,
                            eval_metric="auc",eval_set=[(X_test, y_test)],
                            verbose=False)
    # Since test size is 20%, when retrain model to whole data, expect
    # n_estimator increased to 1/0.8 = 1.25 time.
    estimated_best_round = np.round(self.model_instance.best_ntree_limit * 1.25)
    self.model_instance.n_estimators = np.int64(estimated_best_round)
    self.model_instance.fit(X_train, y_train, eval_metric="auc", verbose=False)
