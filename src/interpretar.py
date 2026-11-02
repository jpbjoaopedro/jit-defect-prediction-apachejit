import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

def shap_global(modelo, X, n=2000, seed=42):
    amostra = X.sample(min(n, len(X)), random_state=seed)
    vals = shap.TreeExplainer(modelo).shap_values(amostra)
    if isinstance(vals, list):          # alguns modelos devolvem uma matriz por classe
        vals = vals[1]
    elif getattr(vals, "ndim", 2) == 3:
        vals = vals[:, :, 1]
    imp = pd.Series(np.abs(vals).mean(0), index=amostra.columns).sort_values(ascending=False)
    return imp, vals, amostra

def permutacao(modelo, X, y, seed=42):
    r = permutation_importance(modelo, X, y, scoring="average_precision",
                               n_repeats=10, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)