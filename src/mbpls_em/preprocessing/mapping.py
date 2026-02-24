import pandas as pd
import mygene

def symbol_names(df):
    """ change the col names by the genes.
        The genes with the max var are selected if repetition
    """

    # --- 1) Clean Ensembl IDs (drop version suffix) ---
    ens_cols_orig = pd.Index(df.columns)
    ens_cols_clean = ens_cols_orig.str.replace(r"\.\d+$", "", regex=True)
    df.columns = ens_cols_clean
    # --- 2) Map Ensembl -> symbol, but fall back to Ensembl if no symbol ---
    mg = mygene.MyGeneInfo()
    res = mg.querymany(df.columns.unique().tolist(),
                       scopes="ensembl.gene", fields="symbol", species="human")

    symbol_map = {}
    for r in res:
        q = r["query"]
        sym = r.get("symbol")
        # fallback to Ensembl ID if no symbol (avoid NaN column names!)
        symbol_map[q] = sym if isinstance(sym, str) and sym.strip() else q

    df_symbols = df.rename(columns=symbol_map)
    # --- 3) For duplicated names, keep the column with the highest variance ---
    # compute variance for each original column
    vars_ = df_symbols.var(axis=0, numeric_only=True)

    # pick the index (column label) with max variance per gene name
    idx_keep = vars_.groupby(level=0).idxmax()

    # subset columns to those winners (this preserves unique names)
    df_unique = df_symbols.loc[:, idx_keep]
    # If any duplicates could remain (edge cases), ensure uniqueness:
    df_unique = df_unique.loc[:, ~df_unique.columns.duplicated()]

    is_symbol = df_unique.columns.str.match(r"^[A-Za-z0-9\-_.]+$") & ~df_unique.columns.str.startswith("ENSG")
    df_unique_symbols_only = df_unique.loc[:, is_symbol]

    return df_unique_symbols_only