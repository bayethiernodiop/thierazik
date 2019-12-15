import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import time
import os
import operator
import scipy
import thierazik.util
import thierazik
from thierazik.TrainModel import TrainModel
from catboost import EFstrType
import json


class TrainCatBoost(TrainModel):
    def __init__(self, data_source, params, run_single_fold=True):
        super().__init__(data_source, run_single_fold)
        self.name="catboost"
        self.params = params
        self.rounds = 10000
        self.early_stop = 50
    def train_model(self, x_train, y_train, x_val, y_val):
        cat_columns = list(x_train.select_dtypes(exclude=["number"]).columns)
        if(thierazik.config["FIT_TYPE"] == thierazik.const.FIT_TYPE_REGRESSION):
            model = CatBoostRegressor(**self.params,
            iterations=self.rounds,random_state=thierazik.config["SEED"],
            silent=True,thread_count=-1)
        
        elif(thierazik.config["FIT_TYPE"] == thierazik.const.FIT_TYPE_BINARY_CLASSIFICATION):
            model = CatBoostClassifier(**self.params, iterations=self.rounds,
            random_state=thierazik.config["SEED"],
            silent=True,thread_count=-1)

        print(f"Will train catboost for {self.rounds} rounds, RandomSeed: {thierazik.config['SEED']}")
        

        if y_val is None:
            model.fit(x_train, y_train,cat_features=cat_columns)
        else:
            early_stop = self.rounds if self.early_stop == 0 else self.early_stop
            model.fit(x_train, y_train, cat_features=cat_columns,
            eval_set=(x_val,y_val), early_stopping_rounds=early_stop)

            self.steps = model.get_best_iteration()
        return model

    def predict_model(self, model, X_test):
        return model.predict(X_test)

    def feature_rank(self,output):
        rank_df = self.model.get_feature_importance(data=None,
                       type=EFstrType.FeatureImportance,
                       prettified=True,
                       thread_count=-1,
                       verbose=False)
        output+="\n"+rank_df.to_string()
        return(output)

    def save_model(self, path, name):
        print("Saving Model")
        self.model.save_model(os.path.join(path, name + ".cbm"),
           format="cbm",
           export_parameters=None,
           pool=None)

        meta = {
            'name': 'TrainCatboost',
            'data_source': self.data_source,
            'params': self.params
        }
        
        with open(os.path.join(path,"meta.json"), 'w') as outfile:
            json.dump(meta, outfile)

    @classmethod
    def load_model(cls,path,name):
        root = thierazik.config['PATH']
        model_path = os.path.join(root,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = cls(meta['data_source'],None,False)
        if(thierazik.config["FIT_TYPE"] == 'reg'):
            model = CatBoostRegressor()
        else:
            model = CatBoostClassifier()
        model.load_model(os.path.join(model_path,name+".cbm"))
        result.model = model
        return result