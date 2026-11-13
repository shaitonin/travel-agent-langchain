"""Avaliação do agente integrador: dataset, avaliadores e execução (LangSmith ou local).

Avaliadores (cada um devolve {key, score, comment}):
  trajetoria_recall    fração dos grupos de ferramentas obrigatórias que foram chamados
  trajetoria_precisao  fração das ferramentas chamadas que eram esperadas/aceitáveis
  trajetoria_exata     1 se chamou todas as obrigatórias e nenhuma indevida
  correcao             LLM-juiz compara a resposta com a referência (1 / 0,5 / 0)
  fundamentacao_nli    fração das afirmações sustentadas pelas evidências (Verificador)
  verificacao_aprovada 1 se o Verificador aprovou a resposta final
  cita_fonte           1 se a resposta cita fonte quando usou ferramentas
"""
from __future__ import annotations

import json
import os
import re
import uuid
from types import SimpleNamespace
from typing import Callable, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from viagens.banco import conectar_somente_leitura
from viagens.dummy import normalizar
from viagens.verificacao import numeros

NOME_DATASET = "viajamais-avaliacao-v1"


def _sql1(sql: str):
    con = conectar_somente_leitura()
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _brl(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def construir_casos() -> List[dict]:
    """Casos de teste. Referências do banco e da função de custo são CALCULADAS aqui."""
    from viagens.servicos.funcao_custo import handler

    voo_paris = _sql1("SELECT numero, preco_brl FROM voos WHERE origem='GRU' AND destino='CDG' AND escalas=0 AND "
                      "classe='economica' AND data_partida LIKE '2027-05%' ORDER BY preco_brl LIMIT 1")[0]
    voo_londres = _sql1("SELECT numero, preco_brl FROM voos WHERE origem='GRU' AND destino='LHR' AND "
                        "classe='economica' AND data_partida LIKE '2027-04%' ORDER BY preco_brl LIMIT 1")[0]
    voo_rio_paris = _sql1("SELECT numero, preco_brl FROM voos WHERE origem='GIG' AND destino='CDG' AND escalas=0 AND "
                          "classe='economica' AND data_partida LIKE '2027-03%' ORDER BY preco_brl LIMIT 1")[0]
    hoteis_roma = [h[0] for h in _sql1("SELECT nome FROM hoteis WHERE cidade='Roma' AND opcao_vegana=1")]
    reservas = _sql1("SELECT id, status FROM reservas WHERE usuario_id='viajante_demo' ORDER BY id")
    custo = json.loads(handler({"dias": 7, "pessoas": 2, "passagem_por_pessoa_brl": 3000, "diaria_hotel_brl": 600})["body"])

    def caso(cid, categoria, pergunta, obrigatorias, aceitaveis, fatos, criterio, historico=None):
        return {"inputs": {"pergunta": pergunta, "historico": historico or []},
                "outputs": {"id": cid, "categoria": categoria, "obrigatorias": obrigatorias, "aceitaveis": aceitaveis,
                            "fatos_chave": fatos, "criterio": criterio}}

    return [
        caso("c01", "api_externa", "Vai chover em Lisboa amanhã?", [["previsao_tempo"]], [],
             ["Lisboa"], "Informa a previsão de amanhã em Lisboa (chance de chuva e/ou temperaturas). Os valores vêm "
             "da API em tempo real e não estão na referência."),
        caso("c02", "api_externa", "Quanto valem 2.000 reais em euros hoje?", [["converter_moeda"]], [],
             [], "Dá o valor de 2.000 reais convertido em euros. A cotação é do dia (API em tempo real) e não está na "
             "referência."),
        caso("c03", "rag", "Brasileiro precisa de visto para ir a Roma?", [["buscar_regras_viagem"]], [],
             ["90"], "Não precisa de visto para turismo de até 90 dias (a cada 180), por ser Espaço Schengen."),
        caso("c04", "rag", "Posso levar carregador portátil na mala despachada?", [["buscar_regras_viagem"]], [],
             ["bagagem de mão"], "Não: power banks/baterias de lítio só na bagagem de mão."),
        caso("c05", "rag", "Preciso fazer algum cadastro online antes de ir pra Inglaterra?", [["buscar_regras_viagem"]], [],
             ["ETA"], "Sim: a ETA (autorização eletrônica de viagem) do Reino Unido, solicitada antes do embarque."),
        caso("c06", "sql", "Qual o voo direto mais barato de São Paulo para Paris em maio de 2027?", [["consultar_banco"]], [],
             [voo_paris[0], _brl(voo_paris[1])], f"Voo {voo_paris[0]} por R$ {_brl(voo_paris[1])}."),
        caso("c07", "sql", "Quais hotéis em Roma têm opção vegana?", [["consultar_banco"]], [],
             hoteis_roma, "Lista exatamente: " + ", ".join(hoteis_roma) + "."),
        caso("c08", "api_interna", "Quais são as minhas reservas e o status de cada uma?",
             [["listar_reservas", "consultar_banco"]], [], [s for _, s in reservas],
             "Lista as reservas do usuário: " + "; ".join(f"#{i} {s}" for i, s in reservas) + "."),
        caso("c09", "serverless", "Quanto custaria no total uma viagem de 7 dias para 2 pessoas, com passagem de 3.000 "
             "reais por pessoa e hotel de 600 reais a diária?", [["estimar_custo"]], ["calculadora"],
             [_brl(custo["total_brl"])], f"Total estimado de R$ {_brl(custo['total_brl'])} (inclui IOF e margem cambial)."),
        caso("c10", "combinada", "Tenho 10 mil reais para 5 dias em Barcelona: quanto dá por dia em euros e que "
             "documentos preciso para entrar?", [["converter_moeda"], ["buscar_regras_viagem"]], ["calculadora"],
             ["passaporte"], "Converte 10 mil reais para euros com a cotação do dia e divide por 5 dias (valor por "
             "dia em euros; a cotação é de API em tempo real); lista documentos de entrada no Espaço Schengen "
             "(passaporte, seguro de 30 mil euros, passagem de volta...)."),
        caso("c11", "combinada", "Qual o voo mais barato de São Paulo para Londres em abril de 2027 e quanto é isso em "
             "libras?", [["consultar_banco"], ["converter_moeda"]], ["calculadora"], [voo_londres[0]],
             f"Voo {voo_londres[0]} por R$ {_brl(voo_londres[1])}, convertido para libras (GBP) com a cotação do dia "
             "(valor em libras vem de API em tempo real e não está na referência)."),
        caso("c12", "memoria", "Qual o voo direto mais barato saindo do Rio para lá nesse mês?", [["consultar_banco"]], [],
             [voo_rio_paris[0]], f"Usa o contexto (Paris, março de 2027): voo {voo_rio_paris[0]} por "
             f"R$ {_brl(voo_rio_paris[1])}.", historico=["Vou para Paris em março de 2027 com meu marido."]),
        caso("c13", "fora_escopo", "Qual a capital da Austrália?", [], [], [],
             "Não usa ferramentas; responde brevemente e/ou diz que o foco é viagem."),
        caso("c14", "alucinacao", "Qual é o valor da taxa de embarque cobrada pela AeroSul em voos para Lisboa?", [],
             ["buscar_regras_viagem", "consultar_banco"], [],
             "A informação não existe na base nem no banco: deve dizer que não tem esse dado, SEM inventar valor."),
        caso("c15", "erro_ferramenta", "Quanto valem 500 reais em pesos argentinos?", [],
             ["converter_moeda", "buscar_regras_viagem"], [],
             "Explica que não consegue converter para pesos argentinos (ARS não é coberta), sem inventar cotação."),
    ]


# ---------------------------------------------------------------------------
# Alvo: o agente responde à pergunta (reproduzindo o histórico, se houver)
# ---------------------------------------------------------------------------
def criar_alvo(agente, callbacks=None) -> Callable[[dict], dict]:
    def alvo(inputs: dict) -> dict:
        sessao = f"aval-{uuid.uuid4().hex[:8]}"
        for msg in inputs.get("historico", []):
            agente.responder(msg, sessao, callbacks=callbacks)
        r = agente.responder(inputs["pergunta"], sessao, callbacks=callbacks)
        return {"resposta": r.resposta, "ferramentas": [t for t, _ in r.ferramentas],
                "verificacao": {k: v for k, v in r.verificacao.items() if k != "afirmacoes"},
                "autocorrigida": r.autocorrigida, "resposta_original": r.resposta_original}
    return alvo


# ---------------------------------------------------------------------------
# Avaliadores
# ---------------------------------------------------------------------------
def _ferramentas(run) -> List[str]:
    return list(dict.fromkeys(run.outputs.get("ferramentas", [])))


def trajetoria_recall(run, example) -> dict:
    grupos = example.outputs["obrigatorias"]
    usadas = set(_ferramentas(run))
    if not grupos:
        return {"key": "trajetoria_recall", "score": 1.0}
    ok = [any(t in usadas for t in g) for g in grupos]
    return {"key": "trajetoria_recall", "score": sum(ok) / len(grupos),
            "comment": f"esperado {grupos}, chamado {sorted(usadas)}"}


def trajetoria_precisao(run, example) -> dict:
    usadas = _ferramentas(run)
    permitidas = {t for g in example.outputs["obrigatorias"] for t in g} | set(example.outputs["aceitaveis"])
    if not usadas:
        return {"key": "trajetoria_precisao", "score": 1.0}
    return {"key": "trajetoria_precisao", "score": sum(t in permitidas for t in usadas) / len(usadas),
            "comment": f"indevidas: {[t for t in usadas if t not in permitidas]}"}


def trajetoria_exata(run, example) -> dict:
    r = trajetoria_recall(run, example)["score"]
    p = trajetoria_precisao(run, example)["score"]
    return {"key": "trajetoria_exata", "score": float(r == 1.0 and p == 1.0)}


def fundamentacao_nli(run, example) -> dict:
    v = run.outputs.get("verificacao") or {}
    if not v.get("evidencias"):   # sem ferramenta não há evidência: a métrica não se aplica
        return {"key": "fundamentacao_nli", "score": None, "comment": "sem evidências (nenhuma ferramenta usada)"}
    return {"key": "fundamentacao_nli", "score": v.get("fundamentacao"), "comment": json.dumps(v.get("rotulos"))}


def verificacao_aprovada(run, example) -> dict:
    v = run.outputs.get("verificacao") or {}
    return {"key": "verificacao_aprovada", "score": float(bool(v.get("aprovada")))}


def cita_fonte(run, example) -> dict:
    if not _ferramentas(run):
        return {"key": "cita_fonte", "score": 1.0}
    return {"key": "cita_fonte", "score": float(bool(re.search(r"\[[^\]]+\]", run.outputs["resposta"])))}


_PROMPT_JUIZ = ChatPromptTemplate.from_messages([
    ("system", "Você é o avaliador de um assistente de viagens que consulta ferramentas (APIs, banco de dados, "
               "base de documentos). A REFERÊNCIA lista o MÍNIMO que a resposta precisa conter, não tudo o que ela "
               "pode dizer. Avalie em dois passos:\n"
               "1) A resposta contém o essencial pedido pela referência?\n"
               "2) Algum valor da resposta CONTRADIZ a referência, ou a resposta inventa um dado que a referência "
               "diz NÃO existir?\n"
               "Regras: detalhes ADICIONAIS (datas, horários, bairros, composição de custos, cotação usada) NÃO "
               "reduzem a nota, pois vêm das ferramentas. Valores de APIs em tempo real (clima, câmbio) não podem ser "
               "conferidos com a referência: NÃO os trate como inventados. Diferenças de formato (4.132,65 x 4132.65) "
               "não são erro.\n"
               "Exemplo: referência 'Voo AS1 por R$ 100,00'; resposta 'O voo AS1 sai dia 6/5 às 23h e custa R$ 100,00' "
               "-> nota 1.\n"
               "Nota: 1 = passo 1 sim e passo 2 não; 0.5 = essencial incompleto; 0 = contradiz a referência ou "
               "inventa o que não existe. Responda só com JSON: "
               '{{"nota": 0|0.5|1, "justificativa": "..."}}'),
    ("human", "PERGUNTA: {pergunta}\nREFERÊNCIA: {criterio}\nFATOS-CHAVE: {fatos}\nRESPOSTA: {resposta}"),
])


def _fato_presente(fato: str, resposta: str) -> bool:
    nums = numeros(fato)
    if nums and re.fullmatch(r"[\d.,\s]+", fato.strip()):
        return any(abs(n - nums[0]) <= 0.011 for n in numeros(resposta))
    return normalizar(fato) in normalizar(resposta)


def criar_avaliador_correcao(llm_juiz) -> Callable:
    cadeia = (_PROMPT_JUIZ | llm_juiz | JsonOutputParser()).with_config(run_name="juiz_correcao")

    def correcao(run, example) -> dict:
        ref = example.outputs
        try:
            j = cadeia.invoke({"pergunta": run.inputs.get("pergunta", ""), "criterio": ref["criterio"],
                               "fatos": ", ".join(ref["fatos_chave"]) or "-", "resposta": run.outputs["resposta"]})
            return {"key": "correcao", "score": float(j["nota"]), "comment": j.get("justificativa", "")}
        except Exception as e:
            return {"key": "correcao", "score": None, "comment": f"falha do juiz: {e}"}
    return correcao


def fatos_chave(run, example) -> dict:
    fatos = example.outputs["fatos_chave"]
    if not fatos:
        return {"key": "fatos_chave", "score": None}
    ok = [f for f in fatos if _fato_presente(f, run.outputs["resposta"])]
    return {"key": "fatos_chave", "score": len(ok) / len(fatos), "comment": f"presentes: {ok}"}


def avaliadores(llm_juiz) -> list:
    return [trajetoria_recall, trajetoria_precisao, trajetoria_exata, criar_avaliador_correcao(llm_juiz), fatos_chave,
            fundamentacao_nli, verificacao_aprovada, cita_fonte]


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def criar_dataset_langsmith(client, casos: List[dict], nome: str = NOME_DATASET):
    if client.has_dataset(dataset_name=nome):
        client.delete_dataset(dataset_name=nome)
    ds = client.create_dataset(nome, description="Assistente de viagens (Viaja+): casos de RAG, SQL, APIs, memória, "
                                                 "fora de escopo, alucinação e erro de ferramenta.")
    client.create_examples(inputs=[c["inputs"] for c in casos], outputs=[c["outputs"] for c in casos], dataset_id=ds.id)
    return ds


def avaliar_local(alvo: Callable, casos: List[dict], evals: list) -> list:
    """Mesmos avaliadores, sem LangSmith (modo dummy/offline)."""
    linhas = []
    for c in casos:
        out = alvo(c["inputs"])
        run = SimpleNamespace(inputs=c["inputs"], outputs=out)
        ex = SimpleNamespace(inputs=c["inputs"], outputs=c["outputs"])
        linha = {"id": c["outputs"]["id"], "categoria": c["outputs"]["categoria"], "pergunta": c["inputs"]["pergunta"],
                 "resposta": out["resposta"], "ferramentas": out["ferramentas"], "autocorrigida": out["autocorrigida"]}
        for ev in evals:
            r = ev(run, ex)
            linha[r["key"]] = r["score"]
            if r.get("comment") and r["key"] in ("correcao", "trajetoria_recall"):
                linha[f"{r['key']}_comentario"] = r["comment"]
        linhas.append(linha)
    return linhas


def resultados_langsmith_para_linhas(resultados) -> list:
    linhas = []
    for item in resultados:
        run, ex = item["run"], item["example"]
        linha = {"id": ex.outputs["id"], "categoria": ex.outputs["categoria"], "pergunta": ex.inputs["pergunta"],
                 "resposta": (run.outputs or {}).get("resposta"), "ferramentas": (run.outputs or {}).get("ferramentas"),
                 "autocorrigida": (run.outputs or {}).get("autocorrigida")}
        for r in item["evaluation_results"]["results"]:
            linha[r.key] = r.score
            if r.comment and r.key in ("correcao", "trajetoria_recall"):
                linha[f"{r.key}_comentario"] = r.comment
        linhas.append(linha)
    return sorted(linhas, key=lambda l: l["id"])
