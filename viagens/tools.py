"""Ferramentas do Assistente de Viagens.

Cada ferramenta existe em duas formas, que compartilham a mesma lógica:

1. **Estruturada** (`StructuredTool` com schema Pydantic): usada via *function
   calling* (llm.bind_tools / create_openai_tools_agent). O modelo preenche
   argumentos tipados.
2. **Texto** (`Tool` de entrada única): exigida pelo ReAct clássico
   ZERO_SHOT_REACT_DESCRIPTION, em que o modelo escreve `Action Input: <texto>`.

Erros das APIs viram `ToolException` e, com handle_tool_error=True, a mensagem
volta ao agente como observação, em vez de derrubar a execução.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from functools import lru_cache
from typing import Optional

from langchain_core.tools import StructuredTool, Tool, ToolException
from pydantic import BaseModel, Field

from viagens import apis


def _json(d) -> str:
    return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Lógica central
# ---------------------------------------------------------------------------
def _previsao(cidade: str, data_inicio: Optional[str] = None, dias: int = 3) -> str:
    try:
        return _json(apis.previsao_tempo(cidade, data_inicio, dias))
    except (apis.APIError, ValueError) as e:
        raise ToolException(f"Erro em previsao_tempo: {e}")


def _converter(valor: float, de: str, para: str) -> str:
    try:
        return _json(apis.converter_moeda(float(valor), de, para))
    except (apis.APIError, ValueError) as e:
        raise ToolException(f"Erro em converter_moeda: {e}")


@lru_cache(maxsize=1)
def _indice():
    from viagens.conhecimento import carregar_indice

    return carregar_indice()


def _buscar_regras(consulta: str, k: int = 3) -> str:
    resultados = _indice().similarity_search_with_score(consulta, k=k)
    trechos = [{"fonte": d.metadata["fonte"], "titulo": d.metadata.get("titulo"),
                "distancia": round(float(s), 3), "trecho": d.page_content} for d, s in resultados]
    return _json(trechos)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def _calcular(expressao: str) -> str:
    """Avalia aritmética com segurança (sem eval): só números e + - * / ** %."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ToolException(f"Expressão não permitida: {expressao!r}")
    try:
        expr = expressao.replace(",", ".").replace("x", "*").strip()
        return str(round(ev(ast.parse(expr, mode="eval").body), 4))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ToolException(f"Erro na calculadora: {e}")


# ---------------------------------------------------------------------------
# Descrições (o modelo decide qual ferramenta usar lendo estes textos)
# ---------------------------------------------------------------------------
DESC_PREVISAO = ("Consulta a previsão do tempo REAL (API Open-Meteo) de uma cidade: temperaturas máx./mín., "
                 "chuva e condição por dia. Para datas a mais de 16 dias, devolve a referência histórica do "
                 "mesmo período no ano anterior (clima típico).")
DESC_CAMBIO = ("Converte um valor entre moedas com a cotação REAL do dia (API Frankfurter, Banco Central Europeu). "
               "Use códigos ISO: BRL, EUR, USD, GBP, JPY, CHF... Não cobre ARS nem CLP.")
DESC_REGRAS = ("Busca semântica na base de conhecimento de viagens (índice FAISS): regras de entrada e visto, "
               "seguro-viagem, bagagem, documentos de menores, IOF e guias de destino (Lisboa, Paris, Roma, "
               "Barcelona, Buenos Aires). Use para qualquer pergunta sobre regras, exigências ou dicas.")
DESC_CALC = "Calculadora aritmética exata. Use para somar custos, dividir orçamentos, porcentagens."


# ---------------------------------------------------------------------------
# 1) Ferramentas estruturadas (function calling)
# ---------------------------------------------------------------------------
class PrevisaoInput(BaseModel):
    cidade: str = Field(description="Nome da cidade, ex.: 'Lisboa'")
    data_inicio: Optional[str] = Field(None, description="Data inicial YYYY-MM-DD; omita para hoje")
    dias: int = Field(3, ge=1, le=7, description="Quantidade de dias (1 a 7)")


class CambioInput(BaseModel):
    valor: float = Field(description="Valor a converter")
    de: str = Field(description="Moeda de origem, código ISO de 3 letras (ex.: BRL)")
    para: str = Field(description="Moeda de destino, código ISO de 3 letras (ex.: EUR)")


class RegrasInput(BaseModel):
    consulta: str = Field(description="Pergunta ou tema a buscar na base de conhecimento")


class CalcInput(BaseModel):
    expressao: str = Field(description="Expressão aritmética, ex.: '15000 * 0.035'")


def ferramentas_estruturadas() -> list:
    return [
        StructuredTool.from_function(_previsao, name="previsao_tempo", description=DESC_PREVISAO,
                                     args_schema=PrevisaoInput, handle_tool_error=True),
        StructuredTool.from_function(_converter, name="converter_moeda", description=DESC_CAMBIO,
                                     args_schema=CambioInput, handle_tool_error=True),
        StructuredTool.from_function(_buscar_regras, name="buscar_regras_viagem", description=DESC_REGRAS,
                                     args_schema=RegrasInput, handle_tool_error=True),
        StructuredTool.from_function(_calcular, name="calculadora", description=DESC_CALC,
                                     args_schema=CalcInput, handle_tool_error=True),
    ]


# ---------------------------------------------------------------------------
# 2) Ferramentas de texto (ReAct clássico, entrada única)
# ---------------------------------------------------------------------------
def _previsao_txt(entrada: str) -> str:
    partes = [p.strip() for p in entrada.strip().strip("'\"").split(",")]
    cidade = partes[0]
    data = next((p for p in partes[1:] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p)), None)
    dias = next((int(m.group(1)) for p in partes[1:] if (m := re.fullmatch(r"(\d{1,2})\s*(?:dias?)?", p))), 3)
    return _previsao(cidade, data, dias)


def _converter_txt(entrada: str) -> str:
    m = re.search(r"([\d.,]+)\s*([A-Za-z]{3})\s*(?:para|to|->|em)?\s*([A-Za-z]{3})", entrada)
    if not m:
        raise ToolException("Formato inválido. Use: '<valor> <MOEDA_ORIGEM> <MOEDA_DESTINO>', ex.: '100 BRL EUR'.")
    valor = m.group(1)
    if "," in valor:                                   # 1.500,50 -> 1500.50
        valor = valor.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", valor):    # 15.000 -> 15000
        valor = valor.replace(".", "")
    return _converter(float(valor), m.group(2), m.group(3))


def ferramentas_react() -> list:
    return [
        Tool(name="previsao_tempo", func=_previsao_txt, handle_tool_error=True,
             description=DESC_PREVISAO + " Entrada: 'Cidade' ou 'Cidade, YYYY-MM-DD' ou 'Cidade, YYYY-MM-DD, dias'."),
        Tool(name="converter_moeda", func=_converter_txt, handle_tool_error=True,
             description=DESC_CAMBIO + " Entrada: '<valor> <ORIGEM> <DESTINO>', ex.: '1500 BRL EUR'."),
        Tool(name="buscar_regras_viagem", func=_buscar_regras, handle_tool_error=True,
             description=DESC_REGRAS + " Entrada: a pergunta em linguagem natural."),
        Tool(name="calculadora", func=_calcular, handle_tool_error=True,
             description=DESC_CALC + " Entrada: a expressão, ex.: '2409.97 / 10'."),
    ]


# ---------------------------------------------------------------------------
# 3) Ferramentas da Fase 4: text-to-SQL, API interna (microsserviço) e função serverless
# ---------------------------------------------------------------------------
class BancoInput(BaseModel):
    pergunta: str = Field(description="Pergunta em linguagem natural sobre voos, hotéis ou reservas do usuário")


class ReservasInput(BaseModel):
    usuario_id: str = Field("viajante_demo", description="Identificador do usuário")


class NovaReservaInput(BaseModel):
    voo_id: int = Field(description="id do voo (coluna id da tabela voos)")
    pessoas: int = Field(1, ge=1, le=9, description="Quantidade de passageiros")
    usuario_id: str = Field("viajante_demo", description="Identificador do usuário")


class CustoInput(BaseModel):
    dias: int = Field(description="Duração da viagem em dias")
    pessoas: int = Field(1, description="Número de viajantes")
    passagem_por_pessoa_brl: float = Field(0, description="Preço da passagem por pessoa, em BRL")
    diaria_hotel_brl: float = Field(0, description="Diária do hotel, em BRL")
    perfil_gastos: str = Field("moderado", description="economico | moderado | confortavel")


def ferramentas_fase4(llm, url_reservas: Optional[str] = None, usuario_id: str = "viajante_demo") -> list:
    """consultar_banco (text-to-SQL), listar_reservas / criar_reserva (API interna) e estimar_custo (serverless)."""
    import requests

    from viagens.servicos import funcao_custo, reservas_api
    from viagens.sql import TextToSQL

    t2s = TextToSQL(llm, usuario_id=usuario_id)
    url = url_reservas or reservas_api.iniciar_servico()

    def _consultar_banco(pergunta: str) -> str:
        reg = t2s.consultar(pergunta)
        return _json({"sql": reg.get("sql"), "resultado": reg.get("resultado"), "resposta": reg.get("resposta")})

    def _listar(usuario_id: str = usuario_id) -> str:
        try:
            r = requests.get(f"{url}/reservas", params={"usuario_id": usuario_id}, timeout=10)
        except requests.RequestException as e:
            raise ToolException(f"API de reservas indisponível: {e}")
        return r.text if r.ok else f"Erro {r.status_code} da API de reservas: {r.text}"

    def _criar(voo_id: int, pessoas: int = 1, usuario_id: str = usuario_id) -> str:
        try:
            r = requests.post(f"{url}/reservas", json={"usuario_id": usuario_id, "voo_id": voo_id, "pessoas": pessoas},
                              timeout=10)
        except requests.RequestException as e:
            raise ToolException(f"API de reservas indisponível: {e}")
        return r.text if r.ok else f"Erro {r.status_code} da API de reservas: {r.text}"

    def _custo(dias: int, pessoas: int = 1, passagem_por_pessoa_brl: float = 0, diaria_hotel_brl: float = 0,
               perfil_gastos: str = "moderado") -> str:
        resp = funcao_custo.invocar({"dias": dias, "pessoas": pessoas, "passagem_por_pessoa_brl": passagem_por_pessoa_brl,
                                     "diaria_hotel_brl": diaria_hotel_brl, "perfil_gastos": perfil_gastos})
        return _json(resp)

    return [
        StructuredTool.from_function(_consultar_banco, name="consultar_banco", args_schema=BancoInput, handle_tool_error=True,
                                     description="Consulta o banco de viagens (SQLite) em linguagem natural: voos da AeroSul "
                                                 "(datas, preços, escalas), hotéis (estrelas, diária, nota, opção vegana) e as "
                                                 "reservas do usuário. Traduz a pergunta para SQL somente leitura."),
        StructuredTool.from_function(_listar, name="listar_reservas", args_schema=ReservasInput, handle_tool_error=True,
                                     description="API interna de reservas: lista as reservas do usuário com status atualizado."),
        StructuredTool.from_function(_criar, name="criar_reserva", args_schema=NovaReservaInput, handle_tool_error=True,
                                     description="API interna de reservas: cria uma reserva de voo PENDENTE. Use só quando o "
                                                 "usuário pedir explicitamente para reservar e o voo_id for conhecido."),
        StructuredTool.from_function(_custo, name="estimar_custo", args_schema=CustoInput, handle_tool_error=True,
                                     description="Função serverless que estima o custo total da viagem em BRL (passagens, "
                                                 "hospedagem, gastos locais, IOF de 3,5% e margem cambial de 10%)."),
    ]
