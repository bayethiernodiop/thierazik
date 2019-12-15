import pandas as pd
import numpy as np
import xgboost as xgb
import time
import os
import zipfile
import operator
from sklearn.metrics import log_loss
import scipy
import thierazik.util
from thierazik.TrainModel import TrainModel
from xgboost import XGBClassifier, XGBRegressor

class TrainXGBoost(TrainModel):
    def __init__(self, data_source, params, run_single_fold=True):
        super().__init__(data_source, run_single_fold)
        self.name="xgboost"
        self.params = params
        self.rounds = 10000
        self.early_stop = 50

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Will train XGB for {} rounds, RandomSeed: {}".format(self.rounds, thierazik.config["SEED"]))
        if(thierazik.config["FIT_TYPE"] == thierazik.const.FIT_TYPE_REGRESSION):
            model = XGBRegressor(**self.params,random_state=thierazik.config["SEED"],
            n_estimators=self.rounds,
            verbosity=0,n_jobs=-1)
        
        elif(thierazik.config["FIT_TYPE"] == thierazik.const.FIT_TYPE_BINARY_CLASSIFICATION):
            model = XGBClassifier(**self.params,random_state=thierazik.config["SEED"],
            n_estimators=self.rounds,
            verbosity=0,n_jobs=-1)

        if y_val is None:
            model.fit(x_train, y_train)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            model.fit(x_train, y_train,
            eval_set=[(x_val,y_val)],verbose=False, early_stopping_rounds=early_stop)

            self.steps = model.best_iteration

        return model

    def predict_model(self, model, X_test):
        return model.predict(X_test)

    def feature_rank(self,output):
        rank = self.model.get_booster().get_fscore()
        rank_sort = sorted(rank.items(), key=operator.itemgetter(1))
        rank_sort.reverse()
        for f in rank_sort:
            output += str(f) + "\n"

        return output