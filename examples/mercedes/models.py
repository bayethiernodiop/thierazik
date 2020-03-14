import config
import thierazik
import thierazik.train_xgboost
import thierazik.train_keras
import thierazik.train_sklearn
import thierazik.train_lightgbm
import thierazik.train_catboost
from thierazik.ensemble_glm import ensemble
import time
import sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from thierazik.HyperOptimizer import HyperOptimizer
from thierazik.model_evaluation import rmse
# Modify the code in this function to build your own XGBoost trainers
# It will br executed only when you run this file directly, and not when
# you import this file from another Python script.s
def run_xgboost():
    COMMON = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    params = {'base_score': 100.669318128, 'learning_rate': 0.005, 'scale_pos_weight': 1, 'colsample_bytree': 0.7, 'min_child_weight': 9, 'subsample': 0.6, 'max_depth': 2, 'silent': 1, 'gamma': 0.0, 'seed': 42, 'reg_alpha': 0.01}
    params = {'scale_pos_weight': 1, 'seed': 42, 'learning_rate': 0.005, 'base_score': 100.669318128, 'colsample_bytree': 0.6, 'max_depth': 2, 'gamma': 0.0, 'reg_alpha': 1, 'silent': 1, 'subsample': 0.7, 'min_child_weight': 9}
    params = {'learning_rate':0.0045,'base_score': 100.669318128,'seed':4242}


    params = {'max_depth': 2, 'subsample': 0.9, 'reg_alpha': 100, 'gamma': 0.0, 'min_child_weight': 7, 'seed': 4242, 'colsample_bytree': 0.9, 'silent': 1, 'base_score': 100.669318128, 'learning_rate': 0.0045}
    params = {'reg_alpha': 1e-05}
    params = {'silent': 1, 'learning_rate': 0.0045, 'max_depth': 3, 'min_child_weight': 1, 'gamma': 0.0, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_alpha': 1e-05}


    params = {**params, **COMMON}
    start_time = time.time()
    train = thierazik.train_xgboost.TrainXGBoost("1",params=params,run_single_fold=False)
    train.run()

    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(thierazik.util.hms_string(elapsed_time)))

def run_keras():
  def define_neural_network(input_shape,output_shape):
        model = Sequential()
        model.add(Dense(20, input_dim=input_shape[1], activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        return model

  start_time = time.time()
  train = thierazik.train_keras.TrainKeras("1",define_neural_network,run_single_fold=False)
  train.zscore = False
  train.run()

  elapsed_time = time.time() - start_time
  print("Elapsed time: {}".format(thierazik.util.hms_string(elapsed_time)))

def run_sklearn():
    # Here you can train a sklearn model including Bagging and voting
  import pickle
  knn_best_params = pickle.load(open(os.path.join(thierazik.config["PATH"],"hp_trials/mercedes_model_best_params.pickle"),"rb"))
  alg_list = [
        #['lreg',LinearRegression()],
        #['bagging_xgb',BaggingRegressor(XGBRegressor(n_jobs=-1))],
        #['extree',ExtraTreesRegressor(n_estimators = 1000,max_depth=2)],
        #['adaboost',AdaBoostRegressor(base_estimator=None, n_estimators=600, learning_rate=1.0)],
        ['knn', sklearn.neighbors.KNeighborsRegressor(**knn_best_params)]
    ]

  start_time = time.time()
  for name,alg in alg_list:
      train = thierazik.train_sklearn.TrainSKLearn("1",name,alg,run_single_fold=False)
      train.run()
      train = None
  elapsed_time = time.time() - start_time
  print("Elapsed time: {}".format(thierazik.util.hms_string(elapsed_time)))

def run_lgb():
  params = {
    'metric':'rmse', 'num_threads': -1, 'objective': 'regression', 'verbosity': 0
  }
  start_time = time.time()

  train = thierazik.train_lightgbm.TrainLightGBM("1",params,run_single_fold=False)
  train.early_stop = 50
  train.run()

  elapsed_time = time.time() - start_time
  print("Elapsed time: {}".format(thierazik.util.hms_string(elapsed_time)))
def run_catboost():
  params = {}
  train = thierazik.train_catboost.TrainCatBoost("1",params=params,run_single_fold=False)
  train.early_stop = 50
  train.run()

def run_ensemble():
  models=open(os.path.join(thierazik.config["PATH"],thierazik.config["MODEL_FILE_FOR_ENSEMBLE"])).read()
  MODELS = models.split("\n")[:-1]
  print("*************",MODELS)
  ensemble(MODELS,LinearRegression())

from hyperopt import hp
from thierazik.util import load_pandas
from thierazik.joiner import join_oos_for_meta_model_finetuning
def finetune_knn():
  target = thierazik.config["TARGET_NAME"]
  train_id = thierazik.config["TRAIN_ID"]
  data_source = "1"
  x_train = load_pandas("train-joined-{}.pkl".format(data_source))
  y_train = x_train[target]
  x_train.drop(train_id, axis=1, inplace=True)
  x_train.drop('fold', axis=1, inplace=True)
  x_train.drop(target, axis=1, inplace=True)
  models_spaces = {}
  space = {}
  models_spaces['knn'] = { 
    'model':        sklearn.neighbors.KNeighborsRegressor,
    'n_neighbors':    hp.choice('n_neighbors', range(1,20)),
    'weights': hp.choice('weights', ["distance"]),
    'p': hp.choice('p', range(1,3)),
    'n_jobs':    hp.choice('n_jobs', [-1]),
}
  space['models_spaces'] = hp.choice(
        'models_spaces',
        [ models_spaces[key] for key in models_spaces ] 
    )
  rmse_scorer = sklearn.metrics.make_scorer(rmse)  
  optimizer = HyperOptimizer(search_space=space,X=x_train,y=y_train,scoring=rmse_scorer,
            trial_folder=os.path.join(thierazik.config["PATH"],"hp_trials"),seed = thierazik.config["SEED"],
            trial_file="mercedes.hyperopt",trial_step=1, trial_initial_step=1,
     debug=False, n_split=3, score_multiplier=1)
  optimizer.optimize()
  #print(space)
def fintune_ensemble():
  models=open(os.path.join(thierazik.config["PATH"],thierazik.config["MODEL_FILE_FOR_ENSEMBLE"])).read()
  MODELS = models.split("\n")[:-1] 
  df_oos, y = join_oos_for_meta_model_finetuning(MODELS)
  print(df_oos.head())
  print(df_oos.shape, y.shape)
if __name__ == "__main__":
    #run_lgb()
    #run_catboost()
    #run_xgboost()
    #run_keras()
    #run_sklearn()
    #run_ensemble()
    #finetune_knn()
    fintune_ensemble()