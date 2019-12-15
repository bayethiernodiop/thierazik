import thierazik
from sklearn.metrics import log_loss, auc, roc_curve, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import seaborn as sns
sns.set()
import pandas as pd

def model_score(y_pred,y_valid):
    final_eval = thierazik.config['FINAL_EVAL']
    if final_eval == thierazik.const.EVAL_R2:
        return r2_score(y_valid, y_pred)
    elif final_eval == thierazik.const.EVAL_LOGLOSS:
        return log_loss(y_valid, y_pred)
    elif final_eval == thierazik.const.EVAL_AUC:
        fpr, tpr, _ = roc_curve(y_valid, y_pred, pos_label=1)
        return auc(fpr, tpr)
    else:
        raise Exception(f"Unknown FINAL_EVAL: {final_eval}")

def rmse(y_true,y_preds):
    return np.sqrt(mean_squared_error(y_true,y_preds))

def importance_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

def plot_feature_importance(imp_df, title,ax=None):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue',ax=ax) \
       .set_title(title, fontsize = 15)

def cv_score(model, train, y_train, scoring, n_folds=3,score_multiplicator=1):
    kf = KFold(n_folds, shuffle=True, random_state=thierazik.config["SEED"]).get_n_splits(train.values)
    score= np.sqrt(score_multiplicator * cross_val_score(model, train.values, y_train, scoring=scoring, cv = kf))
    return(np.mean(score))

def compare_models_predictions(models,features,labels,eval_score,score_colmn_name="score"):
    """
    models : a dictionary => e.g model_name:model
    data : the data to be tested
    eval_score : a function to compute metric => (y_true,y_predictions) return
    score_colmn_name : name to be used for the score column in the returned dataframe
    
    return df with column score,model_name1,...,model_name_n and in row indexes : model_name1,...,model_name_n with df[model_name_i,model_name_j] = mean difference between their predictions
    """
    assert type(models) == dict, "models need to be in a dictionary"
    assert callable(eval_score) == True, "eval score need to be a function"
    
    rows=[]
    scores=[]
    all_predictions = []
    for model_name, model in models.items():
        predictions = model.predict(features)
        all_predictions.append(predictions)
        score=eval_score(labels,predictions)
        scores.append(score)
    for i,(model_name, model) in enumerate(models.items()):
        model_row_values = [scores[i]] + list([np.abs(np.array(all_predictions[i])-np.array(all_predictions[j])).sum()/len(predictions) for j in range(len(models)) ])
        rows.append(model_row_values)
    df_result = pd.DataFrame(
        rows,
        columns=[score_colmn_name] + list(models.keys()),
        index=list(models.keys())
    )
    return df_result