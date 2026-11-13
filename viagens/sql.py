"""Text-to-query (SQL) sobre o banco de viagens.

Pipeline (LCEL + nós imperativos):
  pergunta --(LLM + schema + few-shot)--> SQL
           --(validação: só SELECT, uma instrução, sem tabelas de sistema, LIMIT)--> SQL seguro
           --(execução em conexão READ-ONLY)--> linhas
           --(erro? devolve o erro ao LLM e corrige, 1 tentativa)-->
           --(LLM)--> resposta em linguagem natural citando os dados

Defesa em profundidade: mesmo que a validação textual falhe, a conexão
SQLite é aberta em modo 'ro' e não consegue escrever.
"""
from __future__ import annotations

import json
import re
from datetime import date

from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from viagens.banco import AEROPORTOS, DB_PATH, conectar_somente_leitura, garantir_banco

IATA_POR_CIDADE = {cidade: iata for iata, (cidade, _) in AEROPORTOS.items()}
TABELA_IATA = ", ".join(f"{c}={i}" for c, i in IATA_POR_CIDADE.items())

PROIBIDAS = r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|trigger)\b"
LIMITE_LINHAS = 20

FEW_SHOT = """Pergunta: Qual o voo direto mais barato de São Paulo para Lisboa em março de 2027?
SQL: SELECT numero, data_partida, hora_partida, preco_brl FROM voos WHERE origem = 'GRU' AND destino = 'LIS' AND escalas = 0 AND classe = 'economica' AND data_partida BETWEEN '2027-03-01' AND '2027-03-31' ORDER BY preco_brl LIMIT 1;

Pergunta: Hotéis 4 estrelas ou mais em Paris com café da manhã, do mais bem avaliado ao pior
SQL: SELECT nome, bairro, estrelas, diaria_brl, nota FROM hoteis WHERE cidade = 'Paris' AND estrelas >= 4 AND cafe_incluso = 1 ORDER BY nota DESC LIMIT 20;

Pergunta: Quanto eu já gastei em reservas confirmadas? (usuario_id = viajante_demo)
SQL: SELECT SUM(valor_total_brl) AS total_brl FROM reservas WHERE usuario_id = 'viajante_demo' AND status = 'confirmada';"""

_PROMPT_SQL = ChatPromptTemplate.from_messages([
    ("system",
     "Você converte perguntas sobre viagens em UMA consulta SQL para SQLite (consulta SQL somente leitura).\n"
     "Hoje é {hoje}. O usuário atual é usuario_id = '{usuario_id}'.\n"
     "Regras:\n- use só SELECT; preços em BRL; datas 'YYYY-MM-DD'; inclua LIMIT;\n"
     "- aeroportos (cidade=IATA): {iata}. Use SEMPRE o IATA das cidades citadas NA PERGUNTA "
     "(os exemplos abaixo usam outras cidades: não copie os códigos deles);\n"
     "- os voos cobrem de 2026-12 a 2027-06; mês sem ano = a próxima ocorrência a partir de hoje;\n"
     "- ao combinar condições com OR, use parênteses;\n"
     "- em COMPARAÇÕES entre períodos ou categorias, use GROUP BY para devolver uma linha por período/categoria;\n"
     "- prefira classe='economica' quando a classe não for dita;\n"
     "- se o pedido for para inserir, alterar ou apagar dados, responda apenas: RECUSAR;\n"
     "- responda NAO_EXISTE somente se NENHUMA coluna do esquema se relaciona com o pedido (ex.: taxa de "
     "embarque, bagagem). Atenção: opção vegana (hoteis.opcao_vegana), café da manhã (hoteis.cafe_incluso), "
     "estrelas, nota, escalas, classe e assentos EXISTEM e devem ser consultados.\n"
     "Responda APENAS com o SQL, sem explicações e sem ```.\n\nEsquema:\n{schema}\n\nExemplos:\n{exemplos}"),
    ("human", "Pergunta: {pergunta}"),
])

_PROMPT_CORRIGIR = ChatPromptTemplate.from_messages([
    ("system", "Você corrige consultas SQL para SQLite (consulta SQL somente leitura). Responda APENAS com o SQL "
               "corrigido, sem ```.\nAeroportos (cidade=IATA): {iata}. Os voos cobrem de 2026-12 a 2027-06.\n\n"
               "Esquema:\n{schema}"),
    ("human", "Pergunta: {pergunta}\nSQL com erro: {sql}\nErro do banco: {erro}"),
])

_PROMPT_RESPOSTA = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de viagens. Responda à pergunta usando SOMENTE o resultado da consulta ao "
               "banco de viagens (resultado SQL). Seja direto (até 4 frases), cite valores em R$ e diga "
               "[banco: viagens.db] no fim. Se o resultado vier vazio, diga que não há registros que atendam ao pedido. "
               "Se a pergunta compara períodos ou itens e algum deles não aparece no resultado, diga explicitamente que "
               "não há dados para ele."),
    ("human", "Pergunta: {pergunta}\nSQL executado: {sql}\nResultado SQL (colunas e linhas): {resultado}"),
])


# termos da pergunta que têm coluna no esquema: com eles, "NAO_EXISTE" é falso
COLUNAS_POR_TERMO = {"vegan": "hoteis.opcao_vegana", "cafe": "hoteis.cafe_incluso", "estrela": "hoteis.estrelas",
                     "nota": "hoteis.nota", "avalia": "hoteis.nota", "diaria": "hoteis.diaria_brl",
                     "escala": "voos.escalas", "diret": "voos.escalas", "assento": "voos.assentos_disponiveis",
                     "preco": "voos.preco_brl", "barat": "voos.preco_brl", "reserva": "reservas"}


class SQLInseguroError(ValueError):
    pass


class SQLIncoerenteError(ValueError):
    """O SQL é seguro, mas não corresponde à pergunta (ex.: IATA de outra cidade)."""


def validar_coerencia(pergunta: str, sql: str) -> None:
    """Validação semântica barata: cada cidade citada na pergunta precisa aparecer no SQL
    (como IATA, em consultas de voos, ou pelo nome, em consultas de hotéis)."""
    from viagens.dummy import _cidades_na_ordem, normalizar

    cidades = [c for c in _cidades_na_ordem(normalizar(pergunta)) if c in IATA_POR_CIDADE]
    s = sql.lower()
    faltando = []
    for c in cidades:
        if "voos" in s and IATA_POR_CIDADE[c].lower() not in s:
            faltando.append(f"{c} (IATA {IATA_POR_CIDADE[c]})")
        elif "hoteis" in s and "voos" not in s and normalizar(c) not in normalizar(sql):
            faltando.append(c)
    if faltando:
        raise SQLIncoerenteError("O SQL não filtra pelas cidades citadas na pergunta: " + ", ".join(faltando))
    _checar_precedencia_or(sql)


def _checar_precedencia_or(sql: str) -> None:
    """AND e OR no mesmo nível do WHERE, sem parênteses, quase sempre é um erro de precedência:
    'a AND b AND março OR julho' vira '(a AND b AND março) OR julho'."""
    m = re.search(r"\bwhere\b(.*?)(\bgroup\s+by\b|\border\s+by\b|\blimit\b|\bunion\b|$)", sql, flags=re.I | re.S)
    if not m:
        return
    nivel, tokens_topo = 0, []
    for tok in re.findall(r"\(|\)|\band\b|\bor\b", m.group(1), flags=re.I):
        if tok == "(":
            nivel += 1
        elif tok == ")":
            nivel -= 1
        elif nivel == 0:
            tokens_topo.append(tok.lower())
    if "or" in tokens_topo and "and" in tokens_topo:
        raise SQLIncoerenteError("AND e OR no mesmo nível do WHERE sem parênteses: agrupe as alternativas do OR "
                                 "entre parênteses para não perder os demais filtros.")


def limpar_sql(texto: str) -> str:
    texto = re.sub(r"```(?:sql)?", "", texto, flags=re.I).strip()
    if texto.lower().startswith("sql:"):
        texto = texto[4:].strip()
    return texto.rstrip(";").strip()


def validar_sql(sql: str) -> str:
    """Barreira textual: uma única instrução SELECT/WITH, sem comandos de escrita, com LIMIT."""
    s = sql.strip()
    if ";" in s:
        raise SQLInseguroError("Mais de uma instrução SQL não é permitida.")
    if not re.match(r"^\s*(select|with)\b", s, flags=re.I):
        raise SQLInseguroError("Somente consultas SELECT são permitidas.")
    if re.search(PROIBIDAS, s, flags=re.I):
        raise SQLInseguroError("A consulta contém um comando de escrita ou administração.")
    if re.search(r"sqlite_master|sqlite_schema", s, flags=re.I):
        raise SQLInseguroError("Acesso a tabelas de sistema não é permitido.")
    if not re.search(r"\blimit\s+\d+", s, flags=re.I):
        s = f"{s} LIMIT {LIMITE_LINHAS}"
    return s


def executar_sql(sql: str) -> dict:
    con = conectar_somente_leitura()
    try:
        cur = con.execute(sql)
        colunas = [c[0] for c in cur.description]
        linhas = cur.fetchmany(LIMITE_LINHAS)
        return {"colunas": colunas, "linhas": [list(l) for l in linhas]}
    finally:
        con.close()


class TextToSQL:
    def __init__(self, llm, usuario_id: str = "viajante_demo"):
        garantir_banco()
        self.db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}", sample_rows_in_table_info=2)
        self.schema = self.db.get_table_info()
        self.usuario_id = usuario_id
        self._gerar = (_PROMPT_SQL | llm | StrOutputParser()).with_config(run_name="text_to_sql")
        self._corrigir = (_PROMPT_CORRIGIR | llm | StrOutputParser()).with_config(run_name="corrigir_sql")
        self._responder = (_PROMPT_RESPOSTA | llm | StrOutputParser()).with_config(run_name="resposta_sql")

    def gerar_sql(self, pergunta: str, config=None) -> str:
        return limpar_sql(self._gerar.invoke({"pergunta": pergunta, "schema": self.schema, "exemplos": FEW_SHOT,
                                              "hoje": date.today().isoformat(), "usuario_id": self.usuario_id,
                                              "iata": TABELA_IATA}, config=config))

    def consultar(self, pergunta: str, config=None) -> dict:
        """Devolve um registro completo: sql gerado, sql executado, tentativas, resultado e resposta."""
        reg = {"pergunta": pergunta, "tentativas": []}
        sql = self.gerar_sql(pergunta, config)
        if sql.strip().upper().startswith("NAO_EXISTE"):
            # o modelo disse NAO_EXISTE para "opção vegana" mesmo com a coluna citada no prompt (observado);
            # se a pergunta cita algo que TEM coluna no esquema, o NAO_EXISTE é recusado e o SQL é exigido
            from viagens.dummy import normalizar
            colunas = sorted({c for termo, c in COLUNAS_POR_TERMO.items() if termo in normalizar(pergunta)})
            if colunas:
                reg["tentativas"].append({"sql": "NAO_EXISTE", "erro": f"NAO_EXISTE recusado: o esquema tem {colunas}"})
                sql = limpar_sql(self._corrigir.invoke(
                    {"pergunta": pergunta, "sql": "NAO_EXISTE", "schema": self.schema, "iata": TABELA_IATA,
                     "erro": f"A informação EXISTE no esquema: use {', '.join(colunas)}. Gere o SELECT."}, config=config))
        if sql.strip().upper().startswith("NAO_EXISTE"):
            reg["tentativas"].append({"sql": None, "erro": "NAO_EXISTE: dado fora do esquema"})
            reg.update(sql=None, resultado=None, resposta="O banco de viagens não tem essa informação (ele guarda voos, "
                       "hotéis e reservas). Não há dado para responder a essa pergunta.")
            return reg
        if sql.strip().upper().startswith("RECUSAR"):
            reg["tentativas"].append({"sql": None, "erro": "RECUSADO pelo modelo (pedido de escrita)"})
            reg.update(sql=None, resultado=None, resposta="Não posso alterar ou apagar dados: meu acesso ao banco de "
                       "viagens é somente leitura. Posso consultar as suas reservas, se quiser.")
            return reg
        for tentativa in range(2):
            try:
                seguro = validar_sql(sql)
                validar_coerencia(pergunta, seguro)
                resultado = executar_sql(seguro)
                reg["tentativas"].append({"sql": seguro, "erro": None})
                reg.update(sql=seguro, resultado=resultado)
                break
            except SQLInseguroError as e:      # bloqueio de segurança: não tenta "corrigir" para burlar
                reg["tentativas"].append({"sql": sql, "erro": f"BLOQUEADO: {e}"})
                reg.update(sql=sql, resultado=None, resposta=f"Consulta recusada por segurança: {e}")
                return reg
            except Exception as e:             # erro de execução (coluna inexistente etc.): autocorreção
                reg["tentativas"].append({"sql": sql, "erro": str(e)})
                if tentativa == 1:
                    reg.update(sql=sql, resultado=None,
                               resposta=f"Não consegui consultar o banco: {e}")
                    return reg
                sql = limpar_sql(self._corrigir.invoke({"pergunta": pergunta, "sql": sql, "erro": str(e),
                                                        "schema": self.schema, "iata": TABELA_IATA}, config=config))
        reg["resposta"] = self._responder.invoke(
            {"pergunta": pergunta, "sql": reg["sql"],
             "resultado": json.dumps(reg["resultado"], ensure_ascii=False, default=str)}, config=config)
        return reg
