"""Callbacks de logging e depuração das execuções do agente.

LogExecucaoHandler registra cada evento relevante (pensamento/ação do agente,
início/fim/erro de ferramenta, chamadas ao LLM com tokens e latência, erros de
parsing) em:
  - logs/agente.log  (arquivo rotativo, JSON por linha: fácil de filtrar)
  - self.eventos     (lista em memória, para virar DataFrame no notebook)

É complementar ao LangSmith: o LangSmith guarda o trace completo na nuvem; este
handler funciona offline (modo dummy), não depende de conta e produz métricas
locais (contagem de ferramentas, latência, erros) para depuração rápida.
"""
from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from viagens.config import ROOT

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_logger = logging.getLogger("assistente_viagens")
if not _logger.handlers:
    _h = RotatingFileHandler(LOG_DIR / "agente.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


class LogExecucaoHandler(BaseCallbackHandler):
    def __init__(self, sessao: str = "default", verbose: bool = False):
        self.sessao = sessao
        self.verbose = verbose
        self.eventos: List[Dict[str, Any]] = []
        self._inicio: Dict[UUID, float] = {}

    # -- utilitário -------------------------------------------------------
    def _reg(self, evento: str, run_id: Optional[UUID] = None, **dados):
        dur = None
        if run_id is not None and run_id in self._inicio and evento.endswith(("_fim", "_erro")):
            dur = round(time.perf_counter() - self._inicio.pop(run_id), 3)
        item = {"ts": time.strftime("%H:%M:%S"), "sessao": self.sessao, "evento": evento,
                "duracao_s": dur, **dados}
        self.eventos.append(item)
        _logger.info(json.dumps(item, ensure_ascii=False, default=str))
        if self.verbose:
            extra = {k: v for k, v in dados.items() if v is not None}
            print(f"[{item['ts']}] {evento:<16} {('%.2fs' % dur) if dur else '':>6} {str(extra)[:160]}")

    # -- LLM ---------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._inicio[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kw):
        self._inicio[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kw):
        uso = (response.llm_output or {}).get("token_usage") or {}
        if not uso:
            try:
                um = response.generations[0][0].message.usage_metadata or {}
                uso = {"total_tokens": um.get("total_tokens")}
            except (AttributeError, IndexError):
                uso = {}
        self._reg("llm_fim", run_id, tokens=uso.get("total_tokens"))

    def on_llm_error(self, error, *, run_id, **kw):
        self._reg("llm_erro", run_id, erro=str(error)[:300])

    # -- Agente ------------------------------------------------------------
    def on_agent_action(self, action, *, run_id, **kw):
        pensamento = action.log.split("Action:")[0].replace("Thought:", "").strip() if action.log else ""
        self._reg("agente_acao", None, ferramenta=action.tool, entrada=str(action.tool_input)[:200],
                  pensamento=pensamento[:200] or None)

    def on_agent_finish(self, finish, *, run_id, **kw):
        self._reg("agente_fim", None, resposta=str(finish.return_values.get("output"))[:200])

    # -- Ferramentas -------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kw):
        self._inicio[run_id] = time.perf_counter()
        self._reg("tool_inicio", None, ferramenta=(serialized or {}).get("name"), entrada=str(input_str)[:200])

    def on_tool_end(self, output, *, run_id, **kw):
        texto = str(getattr(output, "content", output))
        self._reg("tool_fim", run_id, ferramenta=kw.get("name"), saida=texto[:200],
                  erro_tratado=texto.startswith("Erro") or None)

    def on_tool_error(self, error, *, run_id, **kw):
        self._reg("tool_erro", run_id, erro=str(error)[:300])

    # -- Chains / parsing --------------------------------------------------
    def on_chain_error(self, error, *, run_id, **kw):
        self._reg("chain_erro", None, tipo=type(error).__name__, erro=str(error)[:300])

    def on_text(self, text, **kw):
        # o AgentExecutor emite aqui as observações de erro de parsing (handle_parsing_errors)
        if "Invalid Format" in text or "Could not parse" in text:
            self._reg("parsing_erro_tratado", None, detalhe=text[:200])

    # -- resumo ------------------------------------------------------------
    def resumo(self):
        import pandas as pd

        return pd.DataFrame(self.eventos)
