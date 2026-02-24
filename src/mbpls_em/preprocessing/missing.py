import pandas as pd

## missing values ##

def remove_missing(df, r_zero = False):
    """"remove missing data
        (optional) remove 0's values
    """
    df_no_missing = df.dropna(axis= 1)
    if r_zero:
        col_with_zero = (df_no_missing == 0).any()
        col_to_drop = col_with_zero[col_with_zero].index.tolist()
        df_no_missing_zero = df_no_missing.drop(columns=col_to_drop)
        return  df_no_missing_zero
    return df_no_missing
