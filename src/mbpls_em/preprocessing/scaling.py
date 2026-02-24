import pandas as pd
from sklearn.preprocessing import StandardScaler

def center_scale(data_list, key="X", with_mean=True, with_std=True):
    out = []
    for d in data_list:
        X = d[key]
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

        if isinstance(X, pd.DataFrame):
            X_scaled = scaler.fit_transform(X.values)
            X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        else:
            # assume ndarray / array-like
            X_scaled = scaler.fit_transform(X)

        d2 = dict(d)           # avoid mutating the original dict (safer)
        d2[key] = X_scaled
        out.append(d2)

    return out