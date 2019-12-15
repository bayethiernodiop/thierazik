from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
import pickle
from sklearn.model_selection import KFold
import threading as th
import keyboard
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
import pickle
import os
import thierazik
import copy
import json
""" One can add change values for an existing key after the trial is saved, 
this help to not optimize over some hyperparams after some trials by giving for example 
a fixed value , However you can't add new keys to a saved trial, a solution would be to add
 the key but giving it constant value then change it after to many values
"""
class HyperOptimizer(object):
    def __init__(self, search_space,X,y,scoring,trial_folder, trial_file,seed,trial_step=1, trial_initial_step=1,
     debug=False, n_split=3, score_multiplier=1, preprocess_steps = None):
        assert isinstance(search_space,dict), "searc_space need to be a dict"
        assert "models_spaces" in search_space, "model or models need to be associated with the key models_spaces"
        self.search_space = search_space
        self.keep_going = True
        self.scoring = scoring
        self.trial_step = trial_step# how many additional trials to do after loading saved trials. 1 = save after iteration
        self.trial_initial_step = trial_initial_step  # initial max_trials. put something small to not have to wait
        self.preprocess_steps = preprocess_steps
        self.score_multiplier = score_multiplier
        self.debug = debug
        self.trial_folder = trial_folder
       # self.trial_folder = "/home/thierno/Downloads/hp_trials"
        self.trial_file = trial_file
        self.trial_file_path = os.path.join(self.trial_folder,self.trial_file)
        self.X = X
        self.y=y
        self.cv_inner = KFold(
            n_splits=n_split, 
            shuffle=True, 
            random_state=seed, 
            #random_state=54, 
        )
        self.best_params=None
    def get_acc_status(self,model, X_, y):
    
        # Proceed to the cross-validation
        # cv_result is a dict : test_score, train_score, fit_time, score_time, estimator
        cv_results = cross_validate(
            model,
            X_,
            y,
            cv=self.cv_inner,
            scoring=self.scoring,
            n_jobs=-1
        )
        
        return {
            'loss': self.score_multiplier * cv_results['test_score'].mean(),
            'loss_std': cv_results['test_score'].std(),
            'status': STATUS_OK,
        }

    def obj_fnc(self,params):   
        """
        The function that return the value to be minimzed by FMIN wrt hyperparams space
        """ 
        X_train_ = self.X
        # proceed to preprocessing
        if(self.preprocess_steps):
            X_train_ = self.preprocess_steps(params, self.X[:])
        
        # get all parameters, except the model
        parameters = params['models_spaces'].copy()
        del parameters['model']
        
        # instantiation of the classifier model with parameters
        model = params['models_spaces']['model'](**parameters)
        
        # return loss and status
        return(self.get_acc_status(model, X_train_, self.y))

    def run_trials(self):
        os.makedirs(self.trial_folder, exist_ok=True)
        
        try:  # try to load an already saved trials object, and increase the max
            # use data path for this project
            hypopt_trials = pickle.load(open(self.trial_file_path, "rb"))
            print("Found saved Trials! Loading...")
            max_evals = len(hypopt_trials.trials) + self.trial_step
            print("Rerunning from {} trials.".format(len(hypopt_trials.trials)))
            
        except:  # create a new trials object and start searching
            print("Unable to load previous trials...")
            hypopt_trials = Trials()
            max_evals = self.trial_initial_step

        # Optimization accross the search space
        self.best_params = fmin(
            self.obj_fnc,
            space=self.search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=hypopt_trials
        )

        # save the trials object
        with open(self.trial_file_path, mode="wb") as f:
            pickle.dump(hypopt_trials, f)
            
        # get the best_params
        self.best_params = space_eval(self.search_space, self.best_params)
        
        # print the main results
        if(self.debug):
            print(
                "\n----------------------",
                "\nAlgo:", self.best_params['models_spaces']['model'],
                "\nLoss:", hypopt_trials.best_trial['result']['loss'],
                "\nPreprocessing:", self.best_params['preprocessing_steps'],
                "\nModel params:", self.best_params['models_spaces'],
            )
    def save_best_params(self):
        o=self.best_params
        with open(self.trial_file_path.split(".")[0]+"_best_params.txt", mode="w") as f:
            o = copy.deepcopy(self.best_params)
            print(o)
            print("************")
            o["models_spaces"]["model"] = str(o["models_spaces"]["model"]).split("'")[1]
            print(o)
            f.write(json.dumps(o))
       
       
       
       
        with open(self.trial_file_path.split(".")[0]+"_model_best_params.pickle", mode="wb") as f:
            o = copy.deepcopy(self.best_params)
            o = o["models_spaces"]
            o.pop('model', None)
            pickle.dump(o,f)
    def key_capture_thread(self):
        # Blocks until you press 'ESC'.
        keyboard.wait('esc')
        self.keep_going = False
        print('\nInterrupting… Please wait until shut down and the saving of the current trial state.')
    
    def optimize(self):
        self.keep_going = True
        """
        Call this method to run the trials and press ESC to stop the optimization
        """
        th.Thread(target=self.key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
        while self.keep_going:
            print("\nExecuting... Press 'ESC' key to interrupt.")
            self.run_trials()
            if(not self.keep_going):
                self.save_best_params()
            
        print('\nSuccessfully interrupted! The optimization can be restarted with the same state using the saved file')
