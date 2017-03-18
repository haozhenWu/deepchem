"""
Scikit-learn wrapper interface of xgboost
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
  Abstract base class for XGBoost model.
  """
  def __init__(self, model_instance=None, model_dir=None,
               verbose=True, **kwargs):
    """Abstract class for XGBoost models.
    Parameters:
    -----------
    model_instance: object
      Scikit-learn wrapper interface of xgboost
    model_dir: str
      Path to directory where model will be stored.
    """
    if model_dir is not None:
      if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
      model_dir = tempfile.mkdtemp()
    self.model_dir = model_dir
    self.model_instance = model_instance
    self.model_class = model_instance.__class__

    self.verbose = verbose
    if 'early_stopping_rounds' in kwargs:
       	self.early_stopping_rounds = kwargs['early_stopping_rounds']
    else:
	self.early_stopping_rounds = 50
    

  def fit(self, dataset, **kwargs):
    """
    Fits XGBoost model to data.
    """
    X = dataset.X
    y = np.squeeze(dataset.y)
    w = np.squeeze(dataset.w)
    seed = self.model_instance.seed
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state=seed,
                                                        stratify=y)
    if isinstance(self.model_instance,xgb.XGBClassifier):
	metric = "auc"
    else isinstance(self.model_instance,xgb.XGBRegressor):
	metric = "mae"
    self.model_instance.fit(X_train, y_train, 
			    early_stopping_rounds=self.early_stopping_rounds,
                            eval_metric=metric,eval_set=[(X_test, y_test)],
                            verbose=self.verbose)
    # Since test size is 20%, when retrain model to whole data, expect
    # n_estimator increased to 1/0.8 = 1.25 time.
    estimated_best_round = np.round(self.model_instance.best_ntree_limit * 1.25)
    self.model_instance.n_estimators = np.int64(estimated_best_round)
    self.model_instance.fit(X_train, y_train, eval_metric=metric, 
			    verbose=self.verbose)
