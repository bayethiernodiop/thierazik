import thierazik.util
import tensorflow as tf
import scipy.stats
import numpy as np
import time
import os
import json
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
#tf.random.set_seed(thierazik.config["SEED"])

from thierazik.TrainModel import TrainModel

class TrainKeras(TrainModel):
    def __init__(self, data_source,model_creator,run_single_fold=True,loss=None,optimizer=None,epochs=1000):
        super().__init__(data_source, run_single_fold)
        fit_type = thierazik.config['FIT_TYPE']
        self.name="keras"
        self.params = []
        self.early_stop = 50
        self.model_creator = model_creator
        self.epochs = epochs
        self.loss = loss if loss else "mean_squared_error" if fit_type == thierazik.const.FIT_TYPE_REGRESSION else "categorical_crossentropy"
        self.optimizer = optimizer if optimizer else "adam"
    def train_model(self, x_train, y_train, x_val, y_val):
        fit_type = thierazik.config['FIT_TYPE']

        if type(x_train) is not np.ndarray:
            x_train = x_train.values.astype(np.float32)
        if type(y_train) is not np.ndarray:
            y_train = y_train.values.astype(np.int32)

        if x_val is not None:
            if type(x_val) is not np.ndarray:
                x_val = x_val.values.astype(np.float32)
            if type(y_val) is not np.ndarray:
                y_val = y_val.values.astype(np.int32)

        #if fit_type == thierazik.const.FIT_TYPE_REGRESSION:
        if fit_type in [thierazik.const.FIT_TYPE_BINARY_CLASSIFICATION,
        thierazik.const.FIT_TYPE_MULTI_CLASSIFICATION]:
            y_train = pd.get_dummies(y_train).values.astype(np.float32)
            if x_val is not None:
                y_val = pd.get_dummies(y_val).values.astype(np.float32)
        
        model = self.model_creator(x_train.shape,y_train.shape)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # save best model

        if x_val is not None:
            # Early stopping
            monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
            checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=0, save_best_only=True) # sav

            # Fit/train neural network
            model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=512,callbacks=[monitor,checkpointer],verbose=0,epochs=1000,)
            model.load_weights('best_weights.hdf5') # load weights from best model
        else:
            model.fit(x_train,y_train,verbose=0,epochs=self.epochs)

        #self.classifier = clr.best_iteration
        return model

    def predict_model(self, model, x):
        fit_type = thierazik.config['FIT_TYPE']

        if type(x) is not np.ndarray:
            x = x.values.astype(np.float32)

        if fit_type == thierazik.const.FIT_TYPE_REGRESSION:
            pred = model.predict(x)
        else:
            pred = model.predict(x)
            pred = np.array([v[1] for v in pred])
        return pred.flatten()

    def save_model(self, path, name):
        print("Saving Model")

        self.model.save(os.path.join(path, name + ".h5"))

        meta = {
            'name': 'TrainKeras',
            'data_source': self.data_source,
            'params': self.params
        }
        
        with open(os.path.join(path,"meta.json"), 'w') as outfile:
            json.dump(meta, outfile)

    @classmethod
    def load_model(cls,path,name):
        root_path = thierazik.config['PATH']
        model_path = os.path.join(root_path,path)
        meta_filename = os.path.join(model_path,"meta.json")
        with open(meta_filename, 'r') as fp:
            meta = json.load(fp)

        result = TrainKeras(meta['data_source'],False,model_creator=None)
        result.model = load_model(os.path.join(model_path,name + ".h5"))
        return result



        

