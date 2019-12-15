from pandas.api.types import is_numeric_dtype
import numpy as np
# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
               * (normalized_high - normalized_low) + normalized_low

def stretch(y):
    return (y - y.min()) / (y.max() - y.min())

def change_types(df, int_to_float_columns):
    for column in df.columns:
        if(column in int_to_float_columns):
            df[column] = df[column].astype("float")
        elif(not is_numeric_dtype(df[column])):
            df[column] = df[column].astype("category")
        elif(np.issubdtype(df[column], np.int64) and column not in int_to_float_columns):
            df[column] = df[column].astype("category")
    return df