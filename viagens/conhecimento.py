"""Base de conhecimento do assistente: documentos -> chunks -> índice FAISS.

O índice é construído uma vez (Fase 2) e salvo em data/faiss_index/<modo>/.
As fases seguintes apenas CARREGAM o índice existente (requisito da Fase 4).

- Modo real:  OpenAIEmbeddings (text-embedding-3-small).
- Modo dummy: HashingEmbeddings, um embedding lexical determinístico
  (bag-of-words com hashing), que não exige chave. Como só casa palavras,
  ele também evidencia, de forma até mais forte, as falhas por descompasso de vocabulário.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from viagens.config import ROOT, USE_DUMMY

DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "faiss_index" / ("dummy" if USE_DUMMY else "openai")
EMBEDDING_MODEL = "text-embedding-3-small"

_STOP = set("""a o e é de do da dos das em no na nos nas um uma para por com sem que se ao à as os
ou mais menos como mas não sim seu sua seus suas ser ter há já até pelo pela este esta isso
the of and to in""".split())


class HashingEmbeddings(Embeddings):
    """Embedding lexical: unigramas + bigramas normalizados, hashing em `dim` posições."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vetor(self, texto: str) -> List[float]:
        from viagens.dummy import normalizar

        toks = [t for t in re.findall(r"[a-z0-9]+", normalizar(texto)) if t not in _STOP and len(t) > 1]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vetor(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vetor(text)


def get_embeddings() -> Embeddings:
    if USE_DUMMY:
        return HashingEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def carregar_documentos() -> List[Document]:
    """Lê data/docs/*.md. Cabeçalho 'chave: valor' até '---' vira metadado."""
    docs = []
    for arq in sorted(DOCS_DIR.glob("*.md")):
        bruto = arq.read_text(encoding="utf-8")
        cab, corpo = bruto.split("\n---\n", 1)
        meta = dict(linha.split(": ", 1) for linha in cab.strip().splitlines())
        meta["fonte"] = arq.name
        docs.append(Document(page_content=corpo.strip(), metadata=meta))
    return docs


def dividir(docs: List[Document]) -> List[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100, separators=["\n## ", "\n\n", "\n", ". ", " "])
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['fonte']}#{i}"
    return chunks


def construir_indice(salvar: bool = True):
    from langchain_community.vectorstores import FAISS

    chunks = dividir(carregar_documentos())
    vs = FAISS.from_documents(chunks, get_embeddings())
    if salvar:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_DIR))
    return vs


def carregar_indice():
    """Carrega o índice FAISS já construído (sem reindexar)."""
    from langchain_community.vectorstores import FAISS

    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"Índice não encontrado em {INDEX_DIR}. Rode viagens.conhecimento.construir_indice() "
            "ou o notebook da Fase 2.")
    # O arquivo .pkl foi gerado por este próprio projeto, então a desserialização é segura.
    return FAISS.load_local(str(INDEX_DIR), get_embeddings(), allow_dangerous_deserialization=True)
