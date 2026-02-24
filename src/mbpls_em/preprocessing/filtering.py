import pandas as pd
from mbpls_em.preprocessing import remove_missing
from mbpls_em.preprocessing import symbol_names

def cleaning_data(X,Y, dropna=True, get_symbol=False):
    """
    previous name: Prepare_data
    Aling the rows of X and Y. Drop the missng values and 0s
    change the columns names of X by the gene's names
    """
    # align index of Y #
    if not X.index.equals(Y.index):
        Y = Y.reindex(X.index)
    if dropna and get_symbol:
        X_dropna = remove_missing(X,r_zero=True)
        X_new = symbol_names(X_dropna)
    elif dropna:
        X_new = remove_missing(X,r_zero=True)
    elif get_symbol:
        X_new = symbol_names(X)
    else:
        X_new = X

    return dict(X=X_new,Y=Y)

def input_data_mbpls_em(*datasets):
    """
    Previous name: set_input_data
    Keep only common columns across datasets['X'] (preserving the first dataset's column order),
    and ensure each dataset's Y aligns to its own X rows. Row indices can differ across datasets.
    """
    if len(datasets) == 0:
        raise ValueError("No datasets provided")
    # validated and collect columns
    for i, d in enumerate(datasets):
        if "X" not in d or "Y" not in d:
            raise KeyError(f"Dataset {i} must have keys 'X' and 'Y'.")
        if not isinstance(d["X"], pd.DataFrame):
            raise TypeError(f"Dataset {i}['X'] must be a pandas DataFrame.")

    # Common columns across all X's; keep order of the FIRST dataset
    common_cols =  datasets[0]["X"].columns
    for d in datasets[1:]:
        common_cols = common_cols.intersection(d["X"].columns)

    if len(common_cols) == 0:
        raise ValueError("No common columns across datasets.")

    out = []
    for i, d in enumerate(datasets):
        X_sub = d["X"].loc[:,common_cols].copy()
        Y = d["Y"]
        Y = Y.reindex(X_sub.index)
        out.append({"X":X_sub.to_numpy(), "Y":Y.to_numpy()})

    return out, common_cols
