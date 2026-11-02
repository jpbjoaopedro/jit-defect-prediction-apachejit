import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

def modelos():
    return {
        "RL": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     n_jobs=-1, random_state=SEED),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               random_state=SEED, verbose=-1),
    }

def escore_tamanho(teste):
    """Linha de base não supervisionada: menor mudança primeiro (Yang et al., 2016)."""
    return 1.0 / (teste["la"] + teste["ld"] + 1)

def lapredict():
    """Linha de base LApredict (Zeng et al., 2021): RL só com log(1 + la)."""
    return LogisticRegression(max_iter=2000)