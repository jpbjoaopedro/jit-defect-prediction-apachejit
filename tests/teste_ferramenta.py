"""Testes da ferramenta com um repositório Git pequeno, criado na hora.

Uso: python tests/teste_ferramenta.py   (sai com erro se algum assert falhar)
"""
import math
import os
import subprocess
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from extrair import extrair  # noqa: E402
from jit import carregar_modelo, explicar  # noqa: E402


def commit(repo, autor, data, mensagem, arquivos):
    for caminho, conteudo in arquivos.items():
        os.makedirs(os.path.join(repo, os.path.dirname(caminho)), exist_ok=True)
        with open(os.path.join(repo, caminho), "w") as f:
            f.write(conteudo)
    env = {**os.environ, "GIT_AUTHOR_NAME": autor, "GIT_AUTHOR_EMAIL": f"{autor}@x",
           "GIT_COMMITTER_NAME": autor, "GIT_COMMITTER_EMAIL": f"{autor}@x",
           "GIT_AUTHOR_DATE": data, "GIT_COMMITTER_DATE": data}
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", mensagem], cwd=repo, check=True, env=env)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True, env=env).stdout.strip()


def teste_metricas():
    with tempfile.TemporaryDirectory() as repo:
        subprocess.run(["git", "init", "-q", repo], check=True)
        commit(repo, "ana", "2018-01-01T10:00:00", "primeira versão",
               {"a/x.py": "1\n2\n3\n", "leia.txt": "oi\n"})
        alvo = commit(repo, "bia", "2018-01-11T10:00:00", "fix: corrige x",
                      {"a/x.py": "1\n2\nmudou\n", "c/y.py": "1\n2\n"})
        m = extrair(repo, [alvo])[alvo]
    assert (m["la"], m["ld"], m["nf"], m["nd"], m["ns"]) == (3, 1, 2, 2, 2), m
    assert math.isclose(m["ent"], 1.0), m["ent"]       # 2 linhas em cada arquivo
    assert m["ndev"] == 0.5 and m["nuc"] == 0.5, m      # só a/x.py tem histórico (1 autor, 1 mudança)
    assert m["age"] == 5.0, m["age"]                    # 10 dias em a/x.py, média com 0 de c/y.py
    assert (m["aexp"], m["asexp"], m["fix"]) == (0, 0, 1), m


def teste_explicacao_soma_ao_logit():
    artefato = carregar_modelo(os.path.join(RAIZ, "modelos", "modelo_rl.joblib"))
    metricas = {"fix": 1, "la": 120, "ld": 30, "nf": 4, "nd": 2, "ns": 1, "ent": 1.5, "ndev": 3,
                "age": 20, "nuc": 15, "aexp": 50, "arexp": 30, "asexp": 20}
    p, contribuicoes, _ = explicar(artefato, metricas)
    logit = artefato["modelo"][-1].intercept_[0] + contribuicoes.sum()
    assert math.isclose(p, 1 / (1 + math.exp(-logit)), rel_tol=1e-9), (p, logit)


if __name__ == "__main__":
    teste_metricas()
    teste_explicacao_soma_ao_logit()
    print("ok: 2 testes")
