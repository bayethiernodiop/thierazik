import thierazik.util
import scipy.stats
import numpy as np
import time
import sklearn
import os
import json
from sklearn.ensemble import RandomForestRegressor
import pickle
from thierazik.TrainModel import TrainModel

class TrainSKLearn(TrainModel):
    def __init__(self, data_set, name, alg, run_single_fold=True):
        super().__init__(data_set, run_single_fold)
        self.name=name
        self.alg=alg
        self.early_stop = 50
        self.rounds = 10000
        self.params = str(alg)

    def train_model(self, x_train, y_train, x_val, y_val):
        print("Training SKLearn model: {}".format(self.alg))

        x_train = x_train.values.astype(np.float32)
        y_train = y_train.values.astype(np.int32)

        self.alg.fit(x_train, y_train)

        self.steps = 0
        
        return self.alg

    def predict_model(self, model, x):
        fit_type = thierazik.config['FIT_TYPE']

        if fit_type == thierazik.const.FIT_TYPE_REGRESSION:
            return model.predict(x)
        else:
            pred = model.predict_proba(x)
            pred = np.array([v[1] for v in pred])
            return pred

    @classmethod
    def load_model(cls,path,name):
        root = thierazik.config['PATH']
        model_path = os.path.join(root,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = cls(meta['data_source'],meta['params'],None,False)
        with open(os.path.join(model_path, name + ".pkl"), 'rb') as fp:  
            result.model = pickle.load(fp)
        return result
