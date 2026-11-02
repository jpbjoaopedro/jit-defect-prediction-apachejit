import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

def shap_global(modelo, X, n=2000, seed=42):
    """Importância SHAP média (|valor|) numa amostra de X; devolve (importância, valores, amostra)."""
    amostra = X.sample(min(n, len(X)), random_state=seed)
    vals = shap.TreeExplainer(modelo).shap_values(amostra)
    # Conforme a versão do shap e o modelo, vem uma matriz por classe; ficamos com a classe 1 (indutor).
    if isinstance(vals, list):
        vals = vals[1]
    elif getattr(vals, "ndim", 2) == 3:
        vals = vals[:, :, 1]
    imp = pd.Series(np.abs(vals).mean(0), index=amostra.columns).sort_values(ascending=False)
    return imp, vals, amostra

def permutacao(modelo, X, y, seed=42):
    """Queda média da AUC-PR ao embaralhar cada atributo; serve para qualquer modelo, inclusive a RL."""
    r = permutation_importance(modelo, X, y, scoring="average_precision",
                               n_repeats=10, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)