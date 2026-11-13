"""Configuração central: carrega o .env e decide entre modelo real e dummy.

USE_DUMMY=true  -> nenhuma chave é necessária; o LLM é simulado (viagens.dummy)
                   e o tracing do LangSmith é desligado.
USE_DUMMY=false -> usa OPENAI_API_KEY e LANGSMITH_API_KEY do .env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# No macOS, faiss-cpu e torch trazem cada um sua cópia da libomp: com o FAISS carregado
# primeiro, o modelo NLI da Fase 5 aborta o processo ("OMP: Error #15" ou falha silenciosa).
# Testado: importar o torch ANTES do faiss resolve. Não é necessário no Linux (Colab).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401  (só para fixar a ordem de carregamento da libomp)
    except ImportError:
        pass

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_settings() -> bool:
    """Carrega .env (ou .env.example como fallback) e devolve USE_DUMMY."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file = ROOT / ".env.example"
    load_dotenv(env_file, override=False)

    dummy = os.getenv("USE_DUMMY", "true").strip().lower() == "true"
    key = os.getenv("OPENAI_API_KEY", "")
    if not dummy and (not key or key.startswith("sk-...") or "COLE_SUA" in key):
        print("[config] OPENAI_API_KEY ausente: ativando modo dummy.")
        dummy = True
    if dummy:
        # Sem chave, não há para onde enviar traces.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return dummy


USE_DUMMY = load_settings()


def get_llm(temperature: float = 0.0, **kwargs):
    """Devolve o chat model do projeto (gpt-4o-mini) ou o dummy equivalente."""
    if USE_DUMMY:
        from viagens.dummy import DummyChatModel

        return DummyChatModel()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        **kwargs,
    )
