import thierazik.const
import math
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import shutil
import os
import time
from tqdm import tqdm
import scipy
import json
from sklearn.utils import check_array
from thierazik.const import NA_VALUES

def save_pandas(df,filename):
    path = thierazik.config['PATH']
    if filename.endswith(".csv"):
        df.to_csv(os.path.join(path, filename),index=False)
    elif filename.endswith(".pkl"):
        df.to_pickle(os.path.join(path, filename))

def load_pandas(filename):
    path = thierazik.config['PATH']
    if filename.endswith(".csv"):
        return pd.read_csv(os.path.join(path, filename), na_values=NA_VALUES)
    elif filename.endswith(".pkl"):
        return pd.read_pickle(os.path.join(path, filename))


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.int32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)



# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
def get_model_dir(name, erase):
    path = thierazik.config['PATH']
    model_dir = os.path.join(path, name)
    os.makedirs(model_dir, exist_ok=True)
    if erase and len(model_dir) > 4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)  # be careful, this deletes everything below the specified path
    return model_dir


def create_submit_package(name, score):
    score_str = str(round(float(score), 6)).replace('.', 'p')
    time_str = time.strftime("%Y%m%d-%H%M%S")
    filename = name + "-" + score_str + "_" + time_str
    path = os.path.join(thierazik.config['PATH'], filename)
    if not os.path.exists(path):
        os.makedirs(path)
    return path, score_str, time_str,filename


def load_model_preds(model_name): 
    train_id = thierazik.config['TRAIN_ID']
    test_id = thierazik.config['TEST_ID']
    
    dash_idx = model_name.find('-')
    suffix = model_name[dash_idx:]
    
    path = os.path.join(thierazik.config['PATH'], model_name)
    filename_oos = "oos" + suffix + ".csv"
    filename_submit = "submit" + suffix + ".csv"
    path_oos = os.path.join(path,filename_oos)
    path_submit = os.path.join(path,filename_submit)

    df_oos = pd.read_csv(path_oos)
    df_oos.sort_values(by=train_id, ascending=True, inplace=True)
    df_submit = pd.read_csv(path_submit)
    df_submit.sort_values(by=test_id, ascending=True, inplace=True)

    return df_oos, df_submit

def save_importance_report(model,imp):
  root_path = thierazik.config['PATH']
  model_path = os.path.join(root_path,model)
  imp.to_csv(os.path.join(model_path,'peturb.csv'),index=False)

