"""Etapa 9: calcula as 13 métricas de mudança de um commit a partir do repositório Git (PyDriller).

As definições seguem Kamei et al. (2013) e o formato do ApacheJIT: ndev, age e nuc são médias
por arquivo modificado, e a experiência do autor é contada nos commits anteriores do repositório.
"""
import math
import os
import re
from collections import defaultdict
from pydriller import Repository

DIA = 86400
ANO = 365.25 * DIA
# Aproximação: no ApacheJIT, fix vem da ligação com o Jira; aqui só há a mensagem do commit.
PADRAO_FIX = re.compile(r"\b(fix(e[sd])?|bug|defect|patch|issue|error)\b", re.IGNORECASE)


def subsistema(caminho):
    partes = caminho.replace("\\", "/").split("/")
    return partes[0] if len(partes) > 1 else ""


def diretorio(caminho):
    # O ApacheJIT conta diretórios pelo nome final (".../server" e ".../test/server" contam uma vez).
    return os.path.basename(os.path.dirname(caminho.replace("\\", "/")))


def caminho_de(arquivo):
    return arquivo.new_path or arquivo.old_path


class Historico:
    """Acumula, commit a commit, o que é preciso para as métricas de histórico e experiência."""

    def __init__(self):
        self.autores_do_arquivo = defaultdict(set)
        self.ultima_mudanca = {}
        self.mudancas_do_arquivo = defaultdict(int)
        self.datas_do_autor = defaultdict(list)
        self.mudancas_do_autor_no_subsistema = defaultdict(int)

    def metricas(self, commit):
        """Métricas do commit usando só o que veio antes dele; não altera o histórico."""
        arquivos = [a for a in commit.modified_files if caminho_de(a)]
        caminhos = [caminho_de(a) for a in arquivos]
        autor = commit.author.email
        data = commit.author_date.timestamp()
        linhas = [a.added_lines + a.deleted_lines for a in arquivos]
        total = sum(linhas)
        ent = -sum((n / total) * math.log2(n / total) for n in linhas if n) if total else 0.0
        n = max(len(caminhos), 1)
        anteriores = self.datas_do_autor[autor]
        return {
            "fix": int(bool(PADRAO_FIX.search(commit.msg))),
            "la": sum(a.added_lines for a in arquivos),
            "ld": sum(a.deleted_lines for a in arquivos),
            "nf": len(caminhos),
            "nd": len({diretorio(c) for c in caminhos}),
            "ns": len({subsistema(c) for c in caminhos}),
            "ent": ent,
            "ndev": sum(len(self.autores_do_arquivo[c]) for c in caminhos) / n,
            "age": sum((data - self.ultima_mudanca[c]) // DIA  # dias inteiros, como no ApacheJIT
                       for c in caminhos if c in self.ultima_mudanca) / n,
            "nuc": sum(self.mudancas_do_arquivo[c] for c in caminhos) / n,
            "aexp": len(anteriores),
            # experiência recente: cada commit anterior pesa 1 / (anos atrás + 1)
            "arexp": sum(1 / (int((data - d) // ANO) + 1) for d in anteriores),
            "asexp": sum(self.mudancas_do_autor_no_subsistema[(autor, s)]
                         for s in {subsistema(c) for c in caminhos}),
        }

    def registrar(self, commit):
        arquivos = [caminho_de(a) for a in commit.modified_files if caminho_de(a)]
        autor = commit.author.email
        data = commit.author_date.timestamp()
        for c in arquivos:
            self.autores_do_arquivo[c].add(autor)
            self.ultima_mudanca[c] = data
            self.mudancas_do_arquivo[c] += 1
        for s in {subsistema(c) for c in arquivos}:
            self.mudancas_do_autor_no_subsistema[(autor, s)] += 1
        self.datas_do_autor[autor].append(data)


def extrair(repo, alvos):
    """Percorre o histórico uma vez e devolve {hash: métricas} para cada commit em alvos."""
    alvos = set(alvos)
    historico, saida = Historico(), {}
    for commit in Repository(repo).traverse_commits():
        if commit.hash in alvos:
            saida[commit.hash] = historico.metricas(commit)
            if len(saida) == len(alvos):
                break
        if not commit.merge:
            historico.registrar(commit)
    return saida
