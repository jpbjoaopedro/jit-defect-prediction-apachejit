import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

def limiar_por_mcc(y_val, p_val):
    """Limiar de decisão que maximiza o MCC; o 0,5 padrão não serve com classes desbalanceadas."""
    grade = np.linspace(0.05, 0.95, 91)
    return max(grade, key=lambda t: matthews_corrcoef(y_val, p_val >= t))

def estatisticas(y, p, limiar):
    yhat = (p >= limiar).astype(int)
    return {
        "auc_roc": roc_auc_score(y, p),
        # average precision em vez da regra do trapézio: interpolar a curva PR em linha reta superestima a área
        "auc_pr": average_precision_score(y, p),
        "precisao": precision_score(y, yhat, zero_division=0),
        "revocacao": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "mcc": matthews_corrcoef(y, yhat),
        "brier": brier_score_loss(y, p),
    }

def _area_sob_curva_de_esforco(y, esforco):
    """Área da curva (fração do esforço gasto, fração dos indutores achados)."""
    x = np.concatenate([[0], np.cumsum(esforco) / esforco.sum()])
    v = np.concatenate([[0], np.cumsum(y) / max(y.sum(), 1)])
    return np.trapezoid(v, x)  # np.trapezoid só existe a partir do NumPy 2.0 (antes era np.trapz)

def esforco(y, escore, la, ld, corte=0.20):
    """Métricas sensíveis ao esforço, com esforço = linhas modificadas (la + ld + 1).

    Os commits são revisados do maior para o menor escore. Recall@20% e PCI@20% olham os
    primeiros 20% do esforço total; IFA conta os alarmes falsos até o primeiro indutor; Popt
    compara a curva do modelo com a da ordem ótima (1) e a da pior ordem possível (0).
    """
    y = np.asarray(y)
    linhas = np.asarray(la + ld, dtype=float) + 1  # +1: commit sem linhas ainda custa uma olhada
    ordem = np.argsort(-np.asarray(escore), kind="stable")
    y_ord, linhas_ord = y[ordem], linhas[ordem]
    revisado = np.cumsum(linhas_ord) / linhas_ord.sum() <= corte
    alarmes_antes_do_primeiro = np.argmax(y_ord == 1) if y_ord.any() else len(y_ord)
    # np.lexsort ordena pela última chave primeiro
    ordem_otima = np.lexsort((linhas, -y))  # indutores primeiro e, entre eles, os menores
    ordem_pior = np.lexsort((-linhas, y))   # limpos primeiro e, entre eles, os maiores
    area_modelo = _area_sob_curva_de_esforco(y_ord, linhas_ord)
    area_otima = _area_sob_curva_de_esforco(y[ordem_otima], linhas[ordem_otima])
    area_pior = _area_sob_curva_de_esforco(y[ordem_pior], linhas[ordem_pior])
    return {
        "recall20": y_ord[revisado].sum() / max(y_ord.sum(), 1),
        "pci20": revisado.mean(),
        "ifa": int(alarmes_antes_do_primeiro),
        "popt": (1 - (area_otima - area_modelo) / (area_otima - area_pior)
                 if area_otima != area_pior else np.nan),
    }

def delta_cliff(a, b):
    """Tamanho de efeito de -1 a 1: P(a > b) - P(a < b) sobre todos os pares."""
    a, b = np.asarray(a), np.asarray(b)
    maior = sum((x > b).sum() for x in a)
    menor = sum((x < b).sum() for x in a)
    return (maior - menor) / (len(a) * len(b))

def comparar(res_a, res_b):
    """Wilcoxon pareado e delta de Cliff.

    res_a e res_b precisam ser a mesma métrica, nas mesmas janelas e na mesma ordem;
    se a ordem não bater, o pareamento compara janelas diferentes e o teste perde o sentido.
    """
    return {"p_wilcoxon": wilcoxon(res_a, res_b).pvalue,
            "delta_cliff": delta_cliff(res_a, res_b)}