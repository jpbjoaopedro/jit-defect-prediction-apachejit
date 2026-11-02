import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

def limiar_por_mcc(y_val, p_val):
    grade = np.linspace(0.05, 0.95, 91)
    return max(grade, key=lambda t: matthews_corrcoef(y_val, p_val >= t))

def estatisticas(y, p, limiar):
    yhat = (p >= limiar).astype(int)
    return {
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),   # sem interpolação linear
        "precisao": precision_score(y, yhat, zero_division=0),
        "revocacao": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "brier": brier_score_loss(y, p),
    }

def _curva(y, esforco):
    x = np.concatenate([[0], np.cumsum(esforco) / esforco.sum()])
    v = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    return np.trapezoid(v, x)           # NumPy 2.x; em versões antigas, np.trapz

def esforco(y, escore, la, ld, corte=0.20):
    y = np.asarray(y); e = np.asarray(la + ld, dtype=float) + 1
    ordem = np.argsort(-np.asarray(escore), kind="stable")
    ys, es = y[ordem], e[ordem]
    acum = np.cumsum(es) / es.sum()
    insp = acum <= corte
    primeiro = np.argmax(ys == 1) if ys.any() else len(ys)
    ot = np.lexsort((e, -y))            # indutores primeiro, menores antes
    pi = np.lexsort((-e, y))            # limpos primeiro, maiores antes
    a_m, a_ot, a_pi = _curva(ys, es), _curva(y[ot], e[ot]), _curva(y[pi], e[pi])
    return {
        "recall20": ys[insp].sum() / max(ys.sum(), 1),
        "pci20": insp.mean(),
        "ifa": int(primeiro),           # falsos alarmes antes do 1º indutor
        "popt": 1 - (a_ot - a_m) / (a_ot - a_pi) if a_ot != a_pi else np.nan,
    }

def delta_cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    maior = sum((x > b).sum() for x in a); menor = sum((x < b).sum() for x in a)
    return (maior - menor) / (len(a) * len(b))

def comparar(res_a, res_b):
    """res_a e res_b: mesma métrica, mesmas janelas, na mesma ordem."""
    return {"p_wilcoxon": wilcoxon(res_a, res_b).pvalue,
            "delta_cliff": delta_cliff(res_a, res_b)}