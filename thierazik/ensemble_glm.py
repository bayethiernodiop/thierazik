import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
import numpy as np
import os
import time
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso
import thierazik.util
from thierazik.model_evaluation import model_score

# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

def fit_ensemble(x,y,blender):
    blender.fit(x, y)
    return blender

def predict_ensemble(blender,x):
    fit_type = thierazik.config['FIT_TYPE']

    if fit_type == thierazik.const.FIT_TYPE_BINARY_CLASSIFICATION:

        pred = blender.predict_proba(x)
        pred = pred[:, 1]
    else:
        pred = blender.predict(x)

    return pred

def ensemble(models,blender):
    test_id = thierazik.config['TEST_ID']
    train_id = thierazik.config['TRAIN_ID']
    target_name = thierazik.config['TARGET_NAME']
    ensemble_oos_df = []
    ensemble_submit_df = []
    
    print("Loading models...")
    for model in models:
        print("Loading: {}".format(model))
        
        df_oos,df_submit = thierazik.util.load_model_preds(model)
        ensemble_oos_df.append( df_oos )
        ensemble_submit_df.append( df_submit )


    ens_y = np.array(ensemble_oos_df[0]['expected'],dtype=np.int)
    ens_x = np.zeros((ensemble_oos_df[0].shape[0],len(models)))
    pred_x = np.zeros((ensemble_submit_df[0].shape[0],len(models)))

    for i, df in enumerate(ensemble_oos_df):
        ens_x[:,i] = df['predicted']

    for i, df in enumerate(ensemble_submit_df):
        pred_x[:,i] = df[target_name]

    print("Cross validating and generating OOS predictions...")

    start_time = time.time()

    x_train = thierazik.util.load_pandas("train-joined-1.pkl")
    folds = x_train['fold']
    num_folds = folds.nunique()
    print("Found {} folds in dataset.".format(num_folds))

    y_train = x_train[target_name]
    train_ids = x_train[train_id]

    final_preds_train = np.zeros(x_train.shape[0])
    scores = []
    for fold_idx in range(num_folds):
        fold_no = fold_idx + 1
        print("*** Fold #{} ***".format(fold_no))

        mask_train = np.array(folds != fold_no)
        mask_test = np.array(folds == fold_no)
        fold_x_train = ens_x[mask_train]
        fold_x_valid = ens_x[mask_test]
        fold_y_train = y_train[mask_train]
        fold_y_valid = y_train[mask_test]

        print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(fold_x_train.shape, len(fold_y_train),
                                                                        fold_x_valid.shape))

        fold_blender = fit_ensemble(fold_x_train,fold_y_train,blender)
        fold_pred = predict_ensemble(fold_blender,fold_x_valid)
        score = model_score(fold_pred,fold_y_valid)
        final_preds_train[mask_test] = fold_pred
        print("Fold score: {}".format(score))
        scores.append(score)

    score = np.mean(scores)
    print("Mean score: {}".format(score))
    print("OOS Score: {}".format(model_score(final_preds_train,y_train)))




    print("Blending on entire dataset...")


    blender = fit_ensemble(ens_x,ens_y,blender)

    pred = predict_ensemble(blender,pred_x)

    sub = pd.DataFrame()
    sub[test_id] = ensemble_submit_df[0][test_id]
    sub[target_name] = pred
    #stretch(sub)

    print("Writing submit file")



    path, score_str, time_str,folder_name = thierazik.util.create_submit_package(
        f"blend_{blender.__class__.__name__}", score)
    filename = "submit-" + score_str + "_" + time_str
    filename_csv = os.path.join(path, filename) + ".csv"
    filename_txt = os.path.join(path, filename) + ".txt"
    sub.to_csv(filename_csv,index=False)

    filename = "oos-" + score_str + "_" + time_str + ".csv"
    filename = os.path.join(path, filename)
    sub = pd.DataFrame()
    sub[test_id] = train_ids
    sub['expected'] = y_train
    sub['predicted'] = final_preds_train
    sub.to_csv(filename, index=False)

    output = ""

    elapsed_time = time.time() - start_time

    output += "Elapsed time: {}\n".format(thierazik.util.hms_string(elapsed_time))
    output += "OOS score: {}\n".format(score)
    output += "-----Blend Results-------\n"
    if(hasattr(blender, 'coef_')):
        z = abs(blender.coef_)
        z = z / z.sum()
        for name, d in zip(models, z):
            output += "{} : {}\n".format(d, name)

    print(output)

    with open(filename_txt, "w") as text_file:
        text_file.write(output)
    with open(os.path.join(thierazik.config['PATH'], "models_names_for_ensemble.txt"), "a") as f:
            f.write(f"{folder_name}\n")