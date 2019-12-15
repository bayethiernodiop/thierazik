import abc
from typing import List
import os
import time
from thierazik.util import load_pandas,create_submit_package,hms_string
from thierazik.model_evaluation import model_score
import scipy
import thierazik
import pickle
import json
import numpy as np
import pandas as pd
class TrainModel(abc.ABC):
    def __init__(self, data_source, run_single_fold,name=None,params=None):
        self.data_source = data_source
        self.run_single_fold = run_single_fold
        self.num_folds = None
        self.zscore = False
        self.steps = None # How many steps to the best model
        self.cv_steps = [] # How many steps at each CV fold
        self.rounds = None # How many rounds are desired (if supported by model)
        self.pred_denom = 1
        self.name=name
        self.params=params
        self.model = None
        self.train_id = thierazik.config["TRAIN_ID"]
        self.test_id = thierazik.config["TEST_ID"]
        self.target = thierazik.config["TARGET_NAME"]
    @abc.abstractmethod
    def train_model(self,x_train, y_train, x_valid=None, y_valid=None):pass

    @abc.abstractmethod
    def predict_model(self,model, x):pass
    
    def _run_startup(self):
        self.start_time = time.time()
        self.x_train = load_pandas("train-joined-{}.pkl".format(self.data_source))
        self.x_submit = load_pandas("test-joined-{}.pkl".format(self.data_source))

        self.input_columns = list(self.x_train.columns.values)

        # Grab what columns we need, but are not used for training
        self.train_ids = self.x_train[self.train_id]
        self.y_train = self.x_train[self.target]
        self.submit_ids = self.x_submit[self.test_id]
        self.folds = self.x_train['fold']
        self.num_folds = self.folds.nunique()
        print("Found {} folds in dataset.".format(self.num_folds))

        # Drop what is not used for training
        self.x_train.drop(self.train_id, axis=1, inplace=True)
        self.x_train.drop('fold', axis=1, inplace=True)
        self.x_train.drop(self.target, axis=1, inplace=True)
        self.x_submit.drop(self.test_id, axis=1, inplace=True)

        self.input_columns2 = list(self.x_train.columns.values)
        self.final_preds_train = np.zeros(self.x_train.shape[0])
        self.final_preds_submit = np.zeros(self.x_submit.shape[0])

        for i in range(len(self.x_train.dtypes)):
            dt = self.x_train.dtypes[i]
            name = self.x_train.columns[i]

            if dt not in [np.float64, np.float32, np.int32, np.int64]:
                print("Bad type: {}:{}".format(name,name.dtype))

            elif self.x_train[name].isnull().any():
                print("Null values: {}".format(name))

        if self.zscore:
            self.x_train = scipy.stats.zscore(self.x_train)
            self.x_submit = scipy.stats.zscore(self.x_submit)

    def _run_cv(self):
        folds2run = self.num_folds if not self.run_single_fold else 1

        for fold_idx in range(folds2run):
            fold_no = fold_idx + 1
            mask_train = np.array(self.folds != fold_no)
            mask_test = np.array(self.folds == fold_no)
            fold_x_train = self.x_train[mask_train]
            fold_x_valid = self.x_train[mask_test]
            fold_y_train = self.y_train[mask_train]
            fold_y_valid = self.y_train[mask_test]

            print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(fold_x_train.shape, len(fold_y_train),
                                                                               self.x_submit.shape))
            self.model = self.train_model(fold_x_train, fold_y_train, fold_x_valid, fold_y_valid)
            preds_valid = self.predict_model(self.model, fold_x_valid)

            score = model_score(preds_valid,fold_y_valid)

            preds_submit = self.predict_model(self.model, self.x_submit)

            self.final_preds_train[mask_test] = preds_valid
            self.final_preds_submit += preds_submit
            self.denom += 1
            self.pred_denom +=1

            if self.steps is not None:
                self.cv_steps.append(self.steps)

            self.scores.append(score)
            print("Fold score: {}".format(score))

            if fold_no==1:
                self.model_fold1 = self.model
        self.score = np.mean(self.scores)

        if len(self.cv_steps)>0:
            self.rounds = max(self.cv_steps) # Choose how many rounds to use after all CV steps

    def _run_single(self):
        print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(self.x_train.shape, len(self.y_train),
                                                                               self.x_submit.shape))
        self.model = self.train_model(self.x_train, self.y_train, None, None)

        self.final_preds_submit = self.predict_model(self.model, self.x_submit)
        self.pred_denom = 1

    def _run_assemble(self):
        target_name = thierazik.config['TARGET_NAME']
        test_id = thierazik.config['TEST_ID']
        train_id = thierazik.config['TRAIN_ID']
        print("Training done, generating submission file")

        if len(self.scores)==0:
            self.denom = 1
            self.scores.append(-1)
            self.score = -1
            print("Warning, could not produce a validation score.")


        # create model output folder
        path, score_str, time_str,folder_name = create_submit_package(self.name, self.score)
        filename = "submit-" + score_str + "_" + time_str
        filename_log_train = "log_train-" + score_str + "_" + time_str
        filename_csv = os.path.join(path, filename) + ".csv"
        filename_txt = os.path.join(path, filename_log_train) + ".txt"

        sub = pd.DataFrame()
        sub[test_id] = self.submit_ids
        sub[target_name] = self.final_preds_submit / self.pred_denom
        sub.to_csv(filename_csv, index=False)
        output = ""
        print("Generate training OOS file")
        if not self.run_single_fold:
            filename = "oos-" + score_str + "_" + time_str + ".csv"
            filename = os.path.join(path, filename)
            sub = pd.DataFrame()
            sub[train_id] = self.train_ids
            sub['expected'] = self.y_train
            sub['predicted'] = self.final_preds_train
            sub.to_csv(filename, index=False)
            output+="OOS Score: {}\n".format(model_score(self.final_preds_train,self.y_train))
            self.save_model(path, 'model-submit')
            if self.model_fold1:
                t = self.model
                self.model = self.model_fold1
                self.save_model(path, 'model-fold1')
                self.model = t

        print("Generated: {}".format(path))
        elapsed_time = time.time() - self.start_time

        output += "Elapsed time: {}\n".format(hms_string(elapsed_time))
        output += "Mean score: {}\n".format(self.score)
        output += "Fold scores: {}\n".format(self.scores)
        output += "Params: {}\n".format(self.params)
        output += "Columns: {}\n".format(self.input_columns)
        output += "Columns Used: {}\n".format(self.input_columns2)
        output += "Steps: {}\n".format(self.steps)

        output += "*** Model Specific Feature Importance ***\n"
        output = self.feature_rank(output)

        with open(filename_txt, "w") as text_file:
            text_file.write(output)
        with open(os.path.join(thierazik.config['PATH'], "models_names_for_ensemble.txt"), "a") as f:
            f.write(f"{folder_name}\n")
    def feature_rank(self,output):
        return output

    def run(self):
        self.denom = 0
        self.scores = []

        self._run_startup()
        self._run_cv()
        print("Fitting single model for entire training set.")
        self._run_single()
        self._run_assemble()

    def save_model(self, path, name):
        print("Saving Model")
        with open(os.path.join(path, name + ".pkl"), 'wb') as fp:  
            pickle.dump(self.model, fp)

        meta = {
            'name': self.__class__.__name__,
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

        result = cls(meta['data_source'],meta['params'])
        with open(os.path.join(model_path, name + ".pkl"), 'rb') as fp:  
            result.model = pickle.load(fp)
        return result