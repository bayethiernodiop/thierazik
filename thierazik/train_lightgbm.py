import pandas as pd
import numpy as np
import time
import os
import operator
from sklearn.metrics import log_loss
import lightgbm as lgb
import scipy
import thierazik.util
import json
from lightgbm import LGBMClassifier, LGBMRegressor
from thierazik.TrainModel import TrainModel

class TrainLightGBM(TrainModel):
    def __init__(self, data_source, params, run_single_fold=True):
        super().__init__(data_source, run_single_fold)
        self.name = "lgb"
        self.params = params
        self.rounds = 25000 
        self.early_stop = 50

    def train_model(self, x_train, y_train, x_val, y_val):
        # use categorical_feature params with labeled encoded categorical feature
        print("Will train LightGB for {} rounds".format(self.rounds))
        if(thierazik.config["FIT_TYPE"] == thierazik.const.FIT_TYPE_REGRESSION):
            model = LGBMRegressor(**self.params,
            n_estimators=self.rounds,random_state=thierazik.config["SEED"],
            silent=True,n_jobs=-1)
        
        elif(thierazik.config["FIT_TYPE"] == thierazik.const.FIT_TYPE_BINARY_CLASSIFICATION):
            model = LGBMClassifier(**self.params,
            n_estimators=self.rounds,random_state=thierazik.config["SEED"],
            silent=True,n_jobs=-1)

        if y_val is None:
            model.fit(x_train, y_train)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            model.fit(x_train, y_train,
            eval_set=[(x_val,y_val)],verbose=False, early_stopping_rounds=early_stop)

            self.steps = model.best_iteration_

        return model

    def predict_model(self, model, X_test):
        return model.predict(X_test)

    def feature_rank(self,output):
        importance = self.model.booster_.feature_importance()
        top_importance = max(importance)
        importance = [x/top_importance for x in importance]
        importance = sorted(zip(self.x_train.columns, importance), key=lambda x: x[1])
        importance = sorted(importance, key=lambda tup: -tup[1])
        
        for row in importance:
            output += f"{row}\n"
        return output
    
    
        