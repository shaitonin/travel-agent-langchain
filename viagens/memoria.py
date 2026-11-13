"""Memória de longo prazo do Assistente de Viagens (taxonomia CoALA).

- Semântica: fatos estáveis sobre o viajante (preferências, restrições), na forma
  chave -> valor. A fonte da verdade é perfil.json; um índice FAISS dos fatos
  serve para RECUPERAR só os relevantes para a mensagem atual.
  Escrita no padrão do Mem0: ADD (chave nova), UPDATE (chave existente com valor
  diferente, o valor antigo vai para o histórico) e NOOP (nada mudou).
- Episódica: um resumo por sessão encerrada (o que foi planejado, decisões),
  com data, indexado em FAISS para recuperação por similaridade.
- Procedural: fica no prompt PTCF e no código das chains (viagens/agente_react.py);
  não é gravada em tempo de execução neste projeto.

Os dados ficam em data/memoria/<modo>/<usuario>/, separados por modo porque
os embeddings do modo real e do dummy têm dimensões diferentes.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from viagens.config import ROOT, USE_DUMMY
from viagens.conhecimento import get_embeddings

BASE_DIR = ROOT / "data" / "memoria" / ("dummy" if USE_DUMMY else "openai")

CHAVES_PERFIL = {
    "preferencia_voo": "Preferência de voo (direto, horário, classe, assento)",
    "preferencia_hospedagem": "Tipo de hospedagem preferido",
    "restricao_alimentar": "Dieta ou restrição alimentar",
    "companhia_viagem": "Com quem costuma viajar (filhos, cônjuge, pet)",
    "orcamento_habitual": "Faixa de orçamento habitual",
    "interesses": "Interesses de viagem recorrentes",
    "documentos": "Documentos que possui (passaporte europeu, visto americano...)",
    "mobilidade": "Necessidades de acessibilidade ou mobilidade",
}


class MemoriaLongoPrazo:
    def __init__(self, usuario_id: str):
        self.usuario_id = usuario_id
        self.dir = BASE_DIR / usuario_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._emb = get_embeddings()
        self._perfil_path = self.dir / "perfil.json"
        self.perfil = json.loads(self._perfil_path.read_text(encoding="utf-8")) if self._perfil_path.exists() else {}
        self._vs_fatos = self._carregar("fatos")
        self._vs_episodios = self._carregar("episodios")

    # ------------------------------------------------------------------ util
    def _carregar(self, nome):
        from langchain_community.vectorstores import FAISS

        caminho = self.dir / nome
        if (caminho / "index.faiss").exists():
            # índices gerados por este próprio projeto: desserialização segura
            return FAISS.load_local(str(caminho), self._emb, allow_dangerous_deserialization=True)
        return None

    def _adicionar(self, nome: str, doc: Document, doc_id: str):
        from langchain_community.vectorstores import FAISS

        attr = "_vs_fatos" if nome == "fatos" else "_vs_episodios"
        vs = getattr(self, attr)
        if vs is None:
            vs = FAISS.from_documents([doc], self._emb, ids=[doc_id])
        else:
            if doc_id in vs.index_to_docstore_id.values():
                vs.delete([doc_id])
            vs.add_documents([doc], ids=[doc_id])
        setattr(self, attr, vs)
        vs.save_local(str(self.dir / nome))

    def _salvar_perfil(self):
        self._perfil_path.write_text(json.dumps(self.perfil, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- semântica
    @staticmethod
    def _canonico(valor: str) -> str:
        """Compara valores sem acento, caixa ou plural: 'Voo direto' == 'voos diretos'."""
        from viagens.dummy import normalizar

        return " ".join(w.rstrip("s") for w in normalizar(valor).split())

    def lembrar_fato(self, chave: str, valor: str, sessao_id: str = "") -> str:
        """Grava um fato estável. Devolve 'ADD', 'UPDATE' ou 'NOOP'."""
        valor = valor.strip()
        atual = self.perfil.get(chave)
        if atual and self._canonico(atual["valor"]) == self._canonico(valor):
            return "NOOP"
        operacao = "UPDATE" if atual else "ADD"
        historico = (atual or {}).get("historico", [])
        if atual:
            historico.append({"valor": atual["valor"], "ate": datetime.now().isoformat(timespec="seconds")})
        self.perfil[chave] = {"valor": valor, "atualizado_em": datetime.now().isoformat(timespec="seconds"),
                              "sessao": sessao_id, "historico": historico}
        self._salvar_perfil()
        descricao = CHAVES_PERFIL.get(chave, chave)
        self._adicionar("fatos", Document(page_content=f"{descricao}: {valor}",
                                          metadata={"chave": chave, "valor": valor}), f"fato:{chave}")
        return operacao

    def buscar_fatos(self, consulta: str, k: int = 3) -> List[dict]:
        if self._vs_fatos is None:
            return []
        res = self._vs_fatos.similarity_search_with_score(consulta, k=min(k, len(self.perfil)))
        return [{"chave": d.metadata["chave"], "valor": d.metadata["valor"], "distancia": round(float(s), 3)}
                for d, s in res]

    # ------------------------------------------------------------- episódica
    def registrar_episodio(self, resumo: str, sessao_id: str) -> None:
        data = datetime.now().date().isoformat()
        self._adicionar("episodios", Document(page_content=resumo, metadata={"data": data, "sessao": sessao_id}),
                        f"episodio:{sessao_id}")

    def buscar_episodios(self, consulta: str, k: int = 2) -> List[dict]:
        if self._vs_episodios is None:
            return []
        n = len(self._vs_episodios.index_to_docstore_id)
        res = self._vs_episodios.similarity_search_with_score(consulta, k=min(k, n))
        return [{"data": d.metadata["data"], "sessao": d.metadata["sessao"], "resumo": d.page_content,
                 "distancia": round(float(s), 3)} for d, s in res]

    # --------------------------------------------------------------- gestão
    def apagar_tudo(self) -> None:
        """Direito ao esquecimento: remove todas as memórias deste usuário."""
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.perfil, self._vs_fatos, self._vs_episodios = {}, None, None
