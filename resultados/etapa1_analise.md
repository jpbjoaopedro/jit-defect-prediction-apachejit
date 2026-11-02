# Etapa 1: verificação dos dados e latência de descoberta

Análise conferida em 01/11/2026 sobre `data/apachejit_total.csv` e `data/links/`.

## Integridade (bate com Keshavarz e Nagappan, 2022)

- 106.674 commits, 28.239 indutores (26,47%), nenhum valor ausente.
- Período de 11/09/2003 a 26/12/2019.
- Depois da unificação do Hadoop: 13 projetos, Hadoop com 16.192 commits e 18,9%.

## Latência de descoberta

Latência é o tempo entre o commit indutor e o primeiro commit que o corrigiu
(`fix_date` em `commit_links_*.csv`).

- Todos os 28.239 indutores têm data de descoberta. A regra completa de Cabral et
  al. (2019) é aplicável.
- Há 2 latências negativas: Hadoop `ab934e8...` (-0,1 dia, provável diferença entre
  data do autor e do commit) e Hive `38c3f1a...` (-183 dias, inconsistência). Os
  dois devem ser excluídos.
- Em 27% dos indutores, o `bug_date` dos links difere do `author_date` do conjunto
  (4.204 casos com mais de 1 dia). O mais provável é que um use a data do autor e o
  outro a do commit (rebase, cherry-pick). Isso não afeta a latência calculada, que
  usa `author_date`.

### Medianas por projeto (todos os anos)

| Projeto | Indutores | Mediana (dias) | % descobertos em até 90 dias |
|---|---|---|---|
| Kafka | 1.115 | 67,1 | 57,0 |
| Spark | 632 | 69,6 | 54,9 |
| Flink | 2.811 | 95,9 | 48,2 |
| Hive | 4.223 | 97,2 | 48,4 |
| Zeppelin | 622 | 98,3 | 47,6 |
| Ignite | 2.439 | 128,8 | 42,0 |
| HBase | 3.782 | 137,0 | 41,2 |
| Cassandra | 3.117 | 142,3 | 40,4 |
| Hadoop | 3.060 | 163,8 | 37,3 |
| ZooKeeper | 342 | 401,6 | 21,3 |
| Camel | 3.078 | 406,3 | 20,4 |
| Groovy | 1.614 | 561,2 | 19,4 |
| ActiveMQ | 1.404 | 568,3 | 17,4 |
| **Geral** | 28.239 | **155,0** | **38,7** |

## Distorções causadas pela janela de coleta do ApacheJIT

O ApacheJIT só usou relatos de defeito abertos entre 2010 e 2019, o que gera dois
efeitos visíveis nos dados.

Indutores por ano de autoria:

| Ano | 2003 | 2005 | 2007 | 2009 | 2010 | 2012 | 2014 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Taxa de indutores | 5,8% | 10,2% | 14,9% | 26,2% | 40,1% | 33,0% | 27,3% | 28,9% | 26,6% | 21,8% | 10,0% |
| Latência mediana (dias) | 3.868 | 2.332 | 1.427 | 462 | 162 | 197 | 159 | 138 | 98 | 82 | 48 |

1. **Corte à esquerda (antes de 2010).** Um defeito introduzido em 2005 só entra
   no conjunto se foi relatado depois de 2010. Por isso os commits antigos têm taxa
   de indutores artificialmente baixa e latências de anos. As medianas altas de
   Groovy, ActiveMQ, Camel e ZooKeeper vêm em parte daí, porque esses projetos
   começam entre 2003 e 2007.
2. **Corte à direita (2018 e 2019).** Commits recentes só aparecem como indutores
   se o defeito foi corrigido antes do fim de 2019. A taxa cai para 21,8% em 2018 e
   10,0% em 2019, e a latência mediana encolhe. Muitos desses commits "limpos" são
   indutores ainda não descobertos.

Restringindo aos indutores de 2010 a 2017, a mediana geral é 141 dias e só 40,4%
dos defeitos são descobertos em até 90 dias. As medianas por projeto vão de 62,7
(Kafka) a 313,1 dias (Camel).

## Consequências para o experimento

1. **Período de estudo.** Usar commits a partir de 01/01/2010 e não testar em
   janelas que comecem depois de 31/12/2018. Os commits de 2019 podem entrar no
   treino das janelas finais, se fizer sentido, mas não no teste. O motivo é que
   2019 não teve tempo de revelar seus defeitos.
2. **Janela de latência.** Os 90 dias de Cabral et al. (2019) não se transferem ao
   ApacheJIT: lá, 8 de 10 projetos tinham mediana de até cerca de 90 dias; aqui são
   2 de 13 (Kafka e Spark). A análise de sensibilidade deve incluir 180 e 365 dias,
   além de 30 e 90.
3. **Achado para a discussão.** A latência de descoberta no ApacheJIT é medida aqui
   pela primeira vez e é bem maior que nos projetos de Cabral et al. Isso tem
   consequência prática: um modelo retreinado com frequência aprende com muitos
   indutores ainda rotulados como limpos.
