# Resumo dos resultados (etapas 1 a 9)

Gerado em 02/11/2026 e refeito depois da revisão de código (ordenação
determinística dos commits com a mesma data). Execução: Python 3.10, scikit-learn 1.7.2, pandas 2.3.3,
LightGBM 4.7.0, SHAP 0.49.1, semente 42. Rodar de novo no ambiente do
`requirements.txt` (Python 3.14) pode mudar as casas decimais, mas não as
conclusões.

## 1. Dados e recorte

- 106.674 commits e 28.239 indutores conferem com Keshavarz e Nagappan (2022).
- Recorte do estudo: commits a partir de 2010, sem as 2 latências negativas. Ficam
  92.872 commits e 25.625 indutores (27,6%). Janelas de teste vão até 2018.
- 12 de 13 projetos selecionados. O ZooKeeper sai por ter só 590 commits depois de
  2010. Ao todo são 150 janelas de teste de 6 meses (com W = 90), com em média
  3.481 commits de treino e 439 de teste por janela.
- Figura: `fig_taxa_latencia_por_ano.png`. Tabelas: `etapa2_projetos.csv`,
  `etapa2_por_ano.csv`.

## 2. Latência de descoberta (achado original)

- Commits de 2010 a 2017: mediana de 141 dias até a correção, e só 40,4% dos
  defeitos descobertos em até 90 dias.
- Por projeto, a mediana vai de 62,7 dias (Kafka) a 313,1 dias (Camel). Só Kafka e
  Spark ficam abaixo de 90 dias. Em Cabral et al. (2019) eram 8 de 10 projetos.
- Figura: `fig_latencia_por_projeto.png`.

## 3. Atributos

- Pares com |rho| de Spearman ≥ 0,7: nf × ent (0,96), aexp × arexp (0,99), nf × nd
  (0,85), nd × ent (0,81), ndev × nuc (0,77), la × nf (0,72).
- A remoção de correlacionados tira arexp e nf em todos os projetos.
- Figura: `fig_correlacao_spearman.png`. Tabela: `etapa3_atributos_por_projeto.csv`.

## 4. Desempenho dos modelos (walk-forward, W = 90, 150 janelas)

| Modelo | AUC-ROC | AUC-PR | MCC | Brier |
|---|---|---|---|---|
| Regressão logística | **0,823** | **0,717** | **0,417** | **0,162** |
| Random Forest | 0,807 | 0,703 | 0,407 | 0,166 |
| LightGBM | 0,780 | 0,670 | 0,368 | 0,189 |
| LApredict (só la) | 0,776 | 0,633 | 0,343 | 0,185 |

- A RL supera RF, LightGBM e LApredict em AUC-PR com p < 0,001 (Wilcoxon pareado),
  mas os efeitos são pequenos: delta de Cliff de 0,05 contra RF e 0,15 contra
  LightGBM. Em MCC, RL e RF empatam (p = 0,56).
- Com a correção de Holm sobre as 40 comparações (`p_holm` em `etapa6_testes.csv`), todas as
  diferenças significativas continuam significativas (RL x RF em AUC-PR: p ajustado 0,005).
- A AUC-PR da RL fica de 1,35 a 2,82 vezes acima da prevalência do teste e nunca
  abaixo dela, em nenhuma das 150 janelas. O Brier da RL (0,162) é melhor que o de
  um preditor ingênuo (0,198).
- Tabelas: `etapa6_resumo_modelos.csv`, `etapa6_testes.csv`,
  `etapa6_auc_pr_por_projeto.csv`. Figura: `fig_modelos_aucpr_recall20.png`.

## 5. Avaliação sensível ao esforço

- Ordenados pela probabilidade, os modelos supervisionados vão mal: Recall@20%
  entre 0,03 e 0,15. Eles põem no topo as mudanças grandes, que consomem o
  orçamento de esforço.
- Ordenados por densidade de risco (probabilidade dividida pelo esforço), RL e RF
  empatam com a linha de base por tamanho: Recall@20% de 0,61 e 0,60 contra 0,59, e
  Popt de 0,87 contra 0,86.
- A diferença está nos falsos alarmes iniciais: a linha de base por tamanho dá, em
  média, 18 alarmes falsos antes do primeiro indutor; RL, 4,0; LightGBM, 2,7 (delta
  de Cliff de -0,61 a -0,74). Ela também exige inspecionar mais commits: PCI@20% de
  0,77 contra 0,72 da RL.
- Isso confirma, em outro conjunto de dados, o padrão relatado por Yang et al.
  (2016) e Huang, Xia e Lo (2017).

## 6. Validação aleatória superestima, e mais nos modelos complexos

| Modelo | AUC-ROC 10-fold | AUC-ROC walk-forward | Diferença | Diferença na AUC-PR |
|---|---|---|---|---|
| RL | 0,839 | 0,825 | +0,014 | +0,010 |
| RF | 0,857 | 0,807 | +0,049 | +0,067 |
| LightGBM | 0,853 | 0,781 | +0,072 | +0,093 |

- O ranking dos modelos se inverte. Na validação cruzada, RF ou LightGBM vence em
  todos os 12 projetos (a RL nunca fica em primeiro); no walk-forward, a RL vence
  em 10 de 12.
- A diferença de AUC-ROC por projeto e modelo vai de -0,007 a +0,12, dentro da faixa
  de Falessi et al. (2020).
- Figura: `fig_cv_vs_walkforward.png`. Tabelas: `etapa7_cv_vs_walkforward.csv`,
  `etapa7_melhor_modelo.csv`.

## 7. Sensibilidade da janela de latência W (regressão logística)

| W (dias) | AUC-PR | MCC | Brier | % de indutores visíveis no treino |
|---|---|---|---|---|
| 30 | 0,719 | 0,391 | 0,173 | 31,6 |
| 90 | 0,717 | 0,417 | 0,162 | 33,8 |
| 180 | 0,713 | 0,418 | 0,155 | 37,0 |
| 365 | 0,717 | 0,325 | 0,161 | 42,6 |

- O ranking (AUC) praticamente não muda com W.
- A calibração (Brier) e o MCC, que dependem do limiar, mudam: W de 90 ou 180 dias
  dá o melhor equilíbrio. Uma hipótese para a queda do MCC com W = 365 é o treino
  ficar mais distante no tempo, de modo que o limiar escolhido na validação se
  ajusta pior ao teste; isso não foi testado.
- Figura: `fig_sensibilidade_w.png`. Tabela: `etapa7_sensibilidade_w.csv`.

## 8. Fatores de risco (última janela de cada projeto)

- `la` (linhas adicionadas) está entre os 3 atributos mais importantes em todos os
  12 projetos na RL e no SHAP do RF e do LightGBM. `fix` aparece em 8 a 11 dos
  12, conforme o modelo e o método. `ld` (linhas removidas) vem em terceiro.
- Direção do efeito pelo SHAP: mais linhas adicionadas, mais linhas removidas e o
  commit ser uma correção (`fix`) aumentam o risco em todos os projetos, no RF e no
  LightGBM. Entropia e número de diretórios também elevam o risco, em 75% a 100%
  dos projetos.
- Os atributos de experiência do autor (aexp, asexp) têm efeito pequeno e de sinal
  inconsistente entre projetos.
- Conclusão: o risco é dominado pelo tamanho e pela difusão da mudança, de forma
  estável entre projetos, o que conversa com Zeng et al. (2021), em que o LApredict
  usa só `la`.
- Figuras: `fig_ranking_atributos_rl.png`, `shap/fig_shap_hadoop_rf.png` (há um por
  projeto em `figuras/shap/`). Tabelas: `etapa8_estabilidade_top3.csv`,
  `etapa8_direcao_shap.csv`.

## 9. Ferramenta e validação funcional

- Curva de esforço de exemplo: `fig_curva_esforco_hadoop.png`, última janela do
  Hadoop (abril a setembro de 2018, 1.033 commits, só 27 indutores). Nessa janela o
  Tamanho (Recall@20% = 0,78) supera a RL por densidade (0,52); a RL por
  probabilidade fica em 0,22, perto do aleatório. Na média das 150 janelas a RL por
  densidade fica à frente (0,608 contra 0,589), então a figura mostra a variação entre
  janelas e não o caso típico. A taxa de 2,6% de indutores ilustra a censura à direita
  nas janelas de 2018.
- Modelo da ferramenta (`modelos/modelo_rl.joblib`): RL treinada uma vez com os 12
  projetos juntos, regra de latência com corte em 01/01/2019 e W = 90. São 80.267
  commits, 23.070 indutores, C = 10 e limiar 0,53 escolhidos nos 20% mais recentes.
  Ficam 9 atributos depois da remoção de redundâncias (saem nf, nd, ndev e arexp). Os
  maiores coeficientes, já padronizados, são la (0,93), fix (0,85) e ld (0,35).
- A explicação de cada alerta é coeficiente x valor padronizado (log-odds em relação
  a um commit médio). Num modelo linear isso coincide com o SHAP; o teste
  `tests/teste_ferramenta.py` confere que as contribuições somam ao logit da
  probabilidade.
- Validação funcional em 50 commits do ZooKeeper (semente 42), projeto que ficou fora
  do treino (`etapa9_validacao.csv`). As definições de nd e age foram acertadas antes,
  em outros 20 commits do ZooKeeper.

| Atributo | Igual ao ApacheJIT | Até 1% de diferença |
|---|---|---|
| la, ld, nf | 100% | 100% |
| nd, ns | 98% | 98% |
| ent | 94% | 94% |
| age | 90% | 90% |
| aexp | 86% | 86% |
| arexp | 82% | 86% |
| fix | 76% | 76% |
| nuc | 14% | 46% |
| asexp | 26% | 26% |
| ndev | 10% | 10% |

- Por que divergem: `fix` no ApacheJIT vem da ligação com o Jira, e a ferramenta só
  tem a mensagem do commit (5 falsos positivos e 7 falsos negativos em 50). `ndev`,
  `nuc` e `asexp` dependem de como o ApacheJIT conta autores e mudanças anteriores,
  e o código que gerou essas métricas não está no pacote de replicação; os valores
  de `asexp` no CSV (por exemplo 8,9e-11) indicam uma definição diferente da de Kamei.
  Desses, só `nuc` e `asexp` entram no modelo, com coeficientes pequenos (0,20 e -0,04).
- Na amostra, 23 dos 50 commits são indutores. O risco médio foi 0,47 nos indutores e
  0,28 nos limpos (AUC-ROC 0,73, só indicativo com n = 50).
- Exemplos de explicação: `fig_waterfall_indutor.png` (risco 96%) e
  `fig_waterfall_alarme_falso.png` (risco 81%). Nos dois a ferramenta marcou `fix` = 1
  e o ApacheJIT tem 0. Como `fix` é o segundo atributo de maior peso, o erro de
  aproximação dele é a principal limitação prática da ferramenta.

- Modelo da ferramenta no ZooKeeper inteiro (2010 a 2018, 437 commits, 50,6% indutores),
  com os atributos do ApacheJIT (`etapa9_zookeeper.csv`): AUC-ROC 0,899, AUC-PR 0,910,
  MCC 0,631, Brier 0,135, Recall@20% 0,64 com IFA 0, contra Recall@20% 0,63 e IFA 1 do
  Tamanho. É avaliação entre projetos, não temporal: os outros 12 projetos cobrem o
  mesmo período. A queda para AUC 0,73 nos 50 commits com atributos extraídos pela
  ferramenta vem da extração, principalmente de `fix`.

## Limitações a discutir

- Os rótulos vêm do SZZ e, por projeto do ApacheJIT, privilegiam falsos negativos.
- As distorções da janela de coleta foram contornadas pelo recorte, mas não
  eliminadas.
- A interpretação usa só a última janela de cada projeto.
- A validação cruzada foi feita sem ajuste de hiperparâmetros (parâmetros padrão).
  A comparação continua justa para medir o otimismo, mas isso precisa ser dito no
  texto.
- Os resultados vêm só de projetos Java da Apache.
- A ferramenta é prova de conceito: o modelo é agrupado (os experimentos são por
  projeto), `fix` é aproximado pela mensagem e não houve avaliação com usuários.
