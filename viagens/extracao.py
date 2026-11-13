"""Chain de extração da Fase 1, empacotada para reuso nas fases seguintes.

Padrão de composição adotado na Fase 1: o fluxo em LCEL e a lógica de negócio
em um nó @chain pequeno (validar_pedido).
"""
from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from pydantic import BaseModel, Field, field_validator


class PedidoViagem(BaseModel):
    """Contrato de saída da chain. As próximas fases dependem destes campos."""
    intencao: Literal["planejar_viagem", "consultar_clima", "consultar_cambio",
                      "buscar_voo", "buscar_hotel", "consultar_regras", "outro"] = Field(
        description="Intenção principal do usuário")
    origem: Optional[str] = Field(None, description="Cidade de origem, se mencionada")
    destino: Optional[str] = Field(None, description="Cidade de destino, se mencionada")
    data_ida: Optional[str] = Field(None, description="Data de ida no formato YYYY-MM-DD; se só o mês for dito, use o dia 01")
    data_volta: Optional[str] = Field(None, description="Data de volta no formato YYYY-MM-DD, se dedutível")
    num_viajantes: Optional[int] = Field(None, description="Número de viajantes")
    orcamento: Optional[float] = Field(None, description="Valor total do orçamento, apenas o número")
    moeda_orcamento: Optional[str] = Field(None, description="Código ISO da moeda do orçamento (BRL, EUR, USD)")
    interesses: List[str] = Field(default_factory=list, description="Interesses citados, no singular")
    etapas: List[str] = Field(default_factory=list,
                              description="Decomposição da tarefa em etapas curtas e ordenadas")

    @field_validator("interesses", "etapas", mode="before")
    @classmethod
    def _nulo_vira_lista(cls, v):
        # o LLM às vezes devolve null em vez de [] (observado na execução real da Fase 2)
        return [] if v is None else v


parser_pedido = JsonOutputParser(pydantic_object=PedidoViagem)


def prompt_extracao() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         "Você é o módulo de entendimento de um assistente de viagens.\n"
         "Hoje é {hoje}. Toda data da viagem deve ser POSTERIOR a hoje: se o usuário citar só um mês "
         "que já passou neste ano, use o ano seguinte.\n"
         "Extraia do pedido do usuário a intenção e as entidades da viagem, e decomponha a tarefa "
         "em etapas curtas e ordenadas. Não invente valores: campos não mencionados ficam null.\n"
         "Responda APENAS com o JSON, sem texto adicional.\n\n{format_instructions}"),
        ("human", "Pedido: {pedido}"),
    ]).partial(format_instructions=parser_pedido.get_format_instructions(),
               hoje=date.today().isoformat())


def _no_futuro(iso: Optional[str]) -> Optional[str]:
    if not iso:
        return iso
    d = date.fromisoformat(iso)
    while d <= date.today():
        d = d.replace(year=d.year + 1)
    return d.isoformat()


@chain
def validar_pedido(dados: dict) -> dict:
    pedido = PedidoViagem.model_validate(dados)
    if pedido.destino and pedido.destino.islower():
        pedido.destino = pedido.destino.strip().title()
    ida_original = pedido.data_ida
    pedido.data_ida = _no_futuro(pedido.data_ida)
    if pedido.data_volta and ida_original and pedido.data_ida != ida_original:
        delta = date.fromisoformat(pedido.data_ida) - date.fromisoformat(ida_original)
        pedido.data_volta = (date.fromisoformat(pedido.data_volta) + delta).isoformat()
    return pedido.model_dump()


def criar_chain_extracao(llm):
    """prompt | llm | JsonOutputParser | validar_pedido  (entrada: {"pedido": str})."""
    return (prompt_extracao() | llm | parser_pedido | validar_pedido).with_config(run_name="extracao_pedido")
