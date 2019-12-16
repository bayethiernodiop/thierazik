import json
import os
from pathlib import Path
from thierazik import const
from thierazik.ensemble_glm import *
from thierazik.joiner import perform_join
from thierazik import util
from thierazik.util import *
from thierazik.perturb_importance import calculate_importance_perturb

from thierazik.loader import *

from thierazik.train_xgboost import TrainXGBoost
from thierazik.train_keras import TrainKeras
from thierazik.train_sklearn import TrainSKLearn
from thierazik.train_lightgbm import TrainLightGBM
from thierazik.train_catboost import TrainCatBoost

from thierazik.HyperOptimizer import *
from thierazik.model_evaluation import *
from thierazik.explorer import *
config = {}

def load_config(profile,filename = None):
  global config
  if not filename:
    home = str(Path.home())
    filename = os.path.join(home,"thierazikConfig.json") # use a default config
    if not os.path.isfile(filename):
      raise Exception(f"If no 'filename' paramater specifed, assume '.thierazikConfig.json' exists at HOME: {home}")

  with open(filename) as f:
      data = json.load(f)
      if profile not in data:
        raise Exception(f"Undefined profile '{profile}' in file '{filename}'")
      config = data[profile]