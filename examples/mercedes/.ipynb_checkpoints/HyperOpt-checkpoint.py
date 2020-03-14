# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"Collapsed": "false"}
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
class HyperOptimizer(object):
    def __init__(self, search_space,X,y,scoring,trial_file,trial_step=1, trial_initial_step=1,
     debug=False, n_split=3, neg_pos_scoring=1, preprocess_steps = None):
        self.search_space = search_space
        self.keep_going = True
        self.scoring = scoring
        self.trial_step = trial_step# how many additional trials to do after loading saved trials. 1 = save after iteration
        self.trial_initial_step = trial_initial_step  # initial max_trials. put something small to not have to wait
        self.preprocess_steps = preprocess_steps
        self.neg_pos_scoring = neg_pos_scoring
        self.debug = debug        
        self.trial_folder = "/home/thierno/Downloads/hp_trials"
        self.trial_file = trial_file
        self.trial_file_path = os.path.join(self.trial_folder,self.trial_file)
        self.X = X
        self.y=y
        self.cv_inner = KFold(
            n_splits=n_split, 
            shuffle=True, 
            random_state=400, 
        )
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
            'loss': self.neg_pos_scoring * cv_results['test_score'].mean(),
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
        global best_params
        best_params = fmin(
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
        best_params = space_eval(self.search_space, best_params)
        
        # print the main results
        if(self.debug):
            print(
                "\n----------------------",
                "\nAlgo:", best_params['models_spaces']['model'],
                "\nLoss:", hypopt_trials.best_trial['result']['loss'],
                "\nPreprocessing:", best_params['preprocessing_steps'],
                "\nModel params:", best_params['models_spaces'],
            )
    def key_capture_thread(self):
        # Blocks until you press 'ESC'.
        keyboard.wait('esc')
        self.keep_going = False
        print('\nInterrupting… Please wait until shut down and the saving of the current trial state.')

    def optimize(self):
        """
        Call this method to run the trials and press ESC to stop the optimization
        """
        th.Thread(target=self.key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
        while self.keep_going:
            print("\nExecuting... Press 'ESC' key to interrupt.")
            self.run_trials()

        print('\nSuccessfully interrupted! The optimization can be restarted with the same state using the saved file')


# + {"Collapsed": "false"}
X, y = datasets.make_classification(n_samples=1000, n_features=10,
                                    n_informative=5, n_redundant=2, random_state=0)

# + {"Collapsed": "false"}
space = {}
models_spaces = {}

# Initializing the search space for preprocessing steps
space['preprocessing_steps'] = {
    'scale':       hp.choice('scale', [True, False]),
    'normalize':       hp.choice('normalize', [True, False]),
}

models_spaces['rf'] = { 
    'model':        RandomForestClassifier,
    'max_depth':    hp.choice('rf_max_depth', range(1,20)),
    'max_features': hp.choice('rf_max_features', range(1,3)),
    'n_estimators': hp.choice('rf_n_estimators', range(10,50)),
    'criterion':    hp.choice('rf_criterion', ["gini", "entropy"]),
}

### LOGISTIC REGRESSION
models_spaces['logit'] = { 
    'model':          LogisticRegression,
    'warm_start' :    hp.choice('logit_warm_start', [True, False]),
    'fit_intercept' : hp.choice('logit_fit_intercept', [True, False]),
    'tol' :           hp.uniform('logit_tol', 0.00001, 0.0001),
    'C' :             hp.uniform('logit_C', 0.05, 3),
    'solver' :        hp.choice('logit_solver', ['newton-cg', 'lbfgs', 'liblinear']),
    'max_iter' :      hp.choice('logit_max_iter', range(100,1000)),
    'multi_class' :   'auto',
    'class_weight' :  'balanced',
}
space['models_spaces'] = hp.choice(
        'models_spaces',
        [ models_spaces[key] for key in models_spaces ] # add key to know the model
    )


# + {"Collapsed": "false"}
def preprocess_steps(params, X_):
    from sklearn.preprocessing import Normalizer
    from sklearn.preprocessing import StandardScaler
    
    # print(params)
    
    if 'normalize' in params['preprocessing_steps']:
        if params['preprocessing_steps']['normalize'] == True:
            X_ = Normalizer().fit_transform(X_)
        
    if 'scale' in params['preprocessing_steps']:
        if params['preprocessing_steps']['scale'] == True:
            X_ = StandardScaler().fit_transform(X_)

    return X_


# + {"Collapsed": "false"}
optimizer = HyperOptimizer(search_space=space,X=X,y=y,scoring="accuracy",trial_file="first_test.hyperopt",trial_step=1, trial_initial_step=1,
     debug=False, n_split=3, neg_pos_scoring=-1, preprocess_steps = None)

# + {"Collapsed": "false"}
optimizer.optimize()

# + {"Collapsed": "false"}

