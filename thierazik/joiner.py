import pandas as pd
import os
import os.path
import thierazik
import thierazik.util
from sklearn.model_selection import KFold,StratifiedKFold
from thierazik.const import NA_VALUES
import numpy as np
def perform_join(profile_name):
    train_id = thierazik.config['TRAIN_ID']
    test_id = thierazik.config['TEST_ID']
    path = thierazik.config['PATH']
    fit_type = thierazik.config['FIT_TYPE']
    target_name = thierazik.config['TARGET_NAME']
    
    df_train_joined = None
    df_test_joined = None
    data_columns = []

    if profile_name not in thierazik.config['JOIN_PROFILES']:
        raise Exception(f"Undefined join profile: {profile_name}")
    profile = thierazik.config['JOIN_PROFILES'][profile_name]
    folds = thierazik.config['FOLDS']
    seed = thierazik.config['SEED']

    df_train_orig = pd.read_csv(os.path.join(path, "train.csv"), na_values=NA_VALUES)
    df_test_orig = pd.read_csv(os.path.join(path, "test.csv"), na_values=NA_VALUES)
    train_length, test_length = len(df_test_orig), len(df_test_orig)

    for source in profile['SOURCES']:
        print("Processing: {}".format(source))
        filename_train = "data-{}-train.pkl".format(source)
        filename_test = "data-{}-test.pkl".format(source)

        if not os.path.exists(os.path.join(path, filename_train)):
            filename_train = "data-{}-train.csv".format(source)
            filename_test = "data-{}-test.csv".format(source)
            
        df_train = thierazik.util.load_pandas(filename_train)
        df_test = thierazik.util.load_pandas(filename_test)

        assert len(df_train) == train_length and len(df_test) == test_length, f"DATASET {source} Miss match between length of train or/and test"
        df_train.sort_values(by=train_id, ascending=True, inplace=True)
        df_test.sort_values(by=test_id, ascending=True, inplace=True)


        if df_train_joined is None:
            df_train_joined = pd.DataFrame()
            df_test_joined = pd.DataFrame()

            df_train_joined[train_id] = df_train[train_id]
            df_test_joined[test_id] = df_test[test_id]

        # Copy columns

        feature_names = list(df_train.columns.values)

        for name in feature_names:
            col_name = f"{source}:{name}"
            assert col_name not in data_columns, f"{col_name} already in columns, check your config file for duplicates" 
            if name == train_id or name == 'fold' or name == target_name or col_name in profile['IGNORE']:
                continue
            data_columns.append(col_name)
            df_train_joined[col_name] = df_train[name]
            df_test_joined[col_name]  = df_test[name]

    # Eliminate any missing values
    print("Checking and Handling Missing Values and making sure column name are in the same order")
    for name in data_columns:
        med = df_train_joined[name].median()
        df_train_joined[name] = df_train_joined[name].fillna(med)
        df_test_joined[name] = df_test_joined[name].fillna(med)

    # Add in any requested orig fields
    print("Adding Selected Original fields")
    for name in profile['ORIG_FIELDS']:
        col_name = "{}:{}".format('orig', name)
        df_train_joined[name] = df_train_orig[name]
        df_test_joined[name] = df_test_orig[name]

    # Add target
    print("Adding the Target")
    df_train_joined[target_name] = df_train[target_name] # get the target from the last file joined (targets SHOULD be all the same)

    # Balance
    if profile['BALANCE']:  pass

    # Designate folds
    print("Folding")
    df_train_joined.insert(1, 'fold', 0)

    fold = 1
    if fit_type == thierazik.const.FIT_TYPE_REGRESSION:
        kf = KFold(folds, shuffle=True, random_state=seed)
        for train, test in kf.split(df_train_joined):
            df_train_joined.ix[test, 'fold'] = fold
            fold += 1
    else:
        targets = df_train_joined[target_name]
        kf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for train, test in kf.split(df_train_joined, targets):
            df_train_joined.ix[test, 'fold'] = fold
            fold += 1
    #check for duplicate column or rows 

    corr = df_train_joined.corr()
    number_of_duplicated_columns  = 0
    for idx, row in corr.iterrows():
        row_with_perfect_correlation = row[row==1]
        duplicated_columns = list(row_with_perfect_correlation.index)
        if idx in duplicated_columns : duplicated_columns.remove(idx)
        if(len(duplicated_columns)>0):
            print(f"Duplicated columns to {idx} => {duplicated_columns}")
            number_of_duplicated_columns+=len(duplicated_columns)
    print(f"There is {number_of_duplicated_columns/2} duplicated columns")
    print(f"There is {df_train_joined.duplicated().sum()} duplicated rows")

    # Write joined files
    print("Writing output...")
    thierazik.util.save_pandas(df_train_joined,f"train-joined-{profile_name}.pkl")
    thierazik.util.save_pandas(df_test_joined,f"test-joined-{profile_name}.pkl")
    thierazik.util.save_pandas(df_train_joined,f"train-joined-{profile_name}.csv")
    thierazik.util.save_pandas(df_test_joined,f"test-joined-{profile_name}.csv")

def join_oos_for_meta_model_finetuning(models):
    ensemble_oos_df = []
    columns_name = []
    print("Loading models...")
    for model in models:
        dash_idx = model.find('-')
        prefix = model[:dash_idx]
        columns_name.append(f"{prefix}_preds")
        print("Loading: {}".format(model))
        df_oos,_ = thierazik.util.load_model_preds(model)
        ensemble_oos_df.append( df_oos )


    ens_y = np.array(ensemble_oos_df[0]['expected'])
    ens_x = np.zeros((ensemble_oos_df[0].shape[0],len(models)))

    for i, df in enumerate(ensemble_oos_df):
        ens_x[:,i] = df['predicted']
    ens_df = pd.DataFrame(ens_x,columns=columns_name)
    return ens_df,ens_y