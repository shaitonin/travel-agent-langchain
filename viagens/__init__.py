"""Pacote compartilhado do Assistente de Viagens (LangChain Agents).

Reúne configuração, modelo dummy e, nas fases seguintes, ferramentas,
dados e o agente integrador, para que todos os notebooks usem o mesmo código.
"""
from viagens.config import USE_DUMMY, get_llm, load_settings

__all__ = ["USE_DUMMY", "get_llm", "load_settings"]
