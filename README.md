# jit-defect-prediction-apachejit

Código do TCC "Predição de commits indutores de defeito com aprendizado de máquina: modelo preditivo,
análise dos fatores de risco e ferramenta de apoio à revisão de código" (Engenharia de Computação, UVA, 2026).

Treina e avalia regressão logística, Random Forest e LightGBM no [ApacheJIT](https://doi.org/10.5281/zenodo.5907847)
com validação temporal (walk-forward) que respeita o tempo de descoberta dos defeitos, compara com duas
linhas de base (tamanho da mudança e LApredict), explica os modelos com SHAP e permutação e entrega uma
ferramenta de linha de comando que estima o risco de um commit.

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Os dados não estão no repositório. Baixe o pacote do ApacheJIT no Zenodo e copie:

- `dataset/apachejit_total.csv` para `data/apachejit_total.csv`;
- `data/commit_links_*.csv` para `data/links/`.

## Reproduzir os resultados

Rode da raiz do repositório, nesta ordem. Os resultados vão para `resultados/`.

Os arquivos em `resultados/` e o `modelos/modelo_rl.joblib` foram gerados com Python 3.10.12,
scikit-learn 1.7.2, pandas 2.3.3, NumPy 2.2.6, LightGBM 4.7.0, SHAP 0.49.1 e PyDriller 2.12 (semente 42).
O `requirements.txt` fixa versões mais recentes; com elas, as últimas casas decimais podem mudar.

| Etapa | Comando | Saída principal |
|---|---|---|
| 1. Conferência e recorte | `python src/etapa1.py` | `etapa1_*.csv` |
| 2. Exploração e seleção dos projetos | `python src/etapa2.py` | `etapa2_projetos.csv`, figuras de latência |
| 3. Correlação entre atributos | `python src/etapa3.py` | `etapa3_*.csv`, mapa de calor |
| 4 a 6. Walk-forward, modelos e métricas | `python src/experimento.py <projeto> [w_dias]` para cada projeto | `metricas/<projeto>_w<w>.csv` |
| 6. Consolidação e testes estatísticos | `python src/etapa6.py` e `python src/etapa6_curva.py` | `etapa6_*.csv`, boxplots, curva de esforço |
| 7. Validação aleatória e sensibilidade de W | `python src/etapa7_cv.py <projeto>` para cada projeto, depois `python src/etapa7.py` | `etapa7_*.csv` |
| 8. Interpretação | `python src/etapa8.py <projeto>` para cada projeto, depois `python src/etapa8.py --resumo` | `etapa8_*.csv`, figuras SHAP |
| 9. Ferramenta | `python src/modelo_final.py`, `python src/avaliar_zookeeper.py` e `python src/validar_ferramenta.py <clone do zookeeper>` | `modelos/modelo_rl.joblib`, `etapa9_*.csv` |

O `experimento.py` grava janela a janela e, se for interrompido, continua de onde parou.
Para a sensibilidade de W, rode `python src/experimento.py <projeto> 30 RL` (e 180, 365).

Projetos usados: activemq, camel, cassandra, flink, groovy, hadoop, hbase, hive, ignite, kafka, spark e zeppelin.

## Ferramenta

```powershell
python src/jit.py --repo C:\caminho\do\repositorio --commit <hash completo> [--grafico explicacao.png]
```

Saída de exemplo:

```text
Commit 344a30792b
Risco estimado de introduzir defeito: 29% (abaixo do limiar de revisão, 53%)
Fatores que mais pesaram (comparado a um commit médio do treino):
  - linhas adicionadas = 262: aumenta o risco (+0.96 log-odds)
  - commit de correção = 0: reduz o risco (-0.32 log-odds)
  - mudanças anteriores nos arquivos = 2.3: reduz o risco (-0.22 log-odds)
```

É uma prova de conceito. Ela percorre o histórico até o commit, então demora mais em repositórios grandes.
O atributo `fix` é estimado pela mensagem do commit (no ApacheJIT ele vem do Jira), e `ndev`, `nuc` e
`asexp` não reproduzem exatamente os valores do ApacheJIT; veja `resultados/etapa9_validacao.csv`.

Se o scikit-learn instalado for de outra versão, a ferramenta avisa; rode `python src/modelo_final.py`
para treinar o modelo de novo.

Testes da ferramenta: `python tests/teste_ferramenta.py`.
