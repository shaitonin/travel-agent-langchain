"""Modelo de chat simulado para rodar o projeto sem chaves de API.

O DummyChatModel é um BaseChatModel de verdade: aceita invoke, batch e
stream, entra em chains LCEL e aparece nos callbacks como qualquer outro
modelo. As respostas vêm de regras determinísticas (regex + palavras-chave)
registradas em RULES. Cada fase do projeto acrescenta as regras de que
precisa, então o comportamento fica previsível para a avaliação offline.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from typing import Any, Callable, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# --------------------------------------------------------------------------
# Vocabulário do domínio usado pelas regras
# --------------------------------------------------------------------------
CIDADES = {
    "sao paulo": "São Paulo", "rio de janeiro": "Rio de Janeiro", "rio": "Rio de Janeiro",
    "salvador": "Salvador", "recife": "Recife", "florianopolis": "Florianópolis",
    "porto alegre": "Porto Alegre", "belo horizonte": "Belo Horizonte",
    "lisboa": "Lisboa", "porto": "Porto", "paris": "Paris", "roma": "Roma",
    "madri": "Madri", "barcelona": "Barcelona", "londres": "Londres",
    "buenos aires": "Buenos Aires", "santiago": "Santiago", "nova york": "Nova York",
    "toquio": "Tóquio", "milao": "Milão",
    "pasargada": "Pasárgada",  # cidade fictícia: usada para testar o erro de geocodificação
}
MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
INTERESSES = ["praia", "museu", "museus", "gastronomia", "historia", "natureza",
              "compras", "vida noturna", "trilha", "cultura", "vinho"]
NUM_PALAVRAS = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5}


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos, para casar palavras-chave."""
    sem_acento = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sem_acento if not unicodedata.combining(c)).lower()


def _cidades_na_ordem(t: str) -> list[str]:
    achadas = []
    for chave in sorted(CIDADES, key=len, reverse=True):
        for m in re.finditer(rf"\b{chave}\b", t):
            if not any(a <= m.start() < b for a, b, _ in achadas):
                achadas.append((m.start(), m.end(), CIDADES[chave]))
    return [nome for _, _, nome in sorted(achadas)]


def extrair_pedido(texto: str, hoje: Optional[date] = None) -> dict:
    """Extração por regras: a 'imitação' do que o LLM faz na Fase 1."""
    hoje = hoje or date.today()
    t = normalizar(texto)
    cidades = _cidades_na_ordem(t)

    origem = destino = None
    m_de = re.search(r"\b(?:de|do|da|saindo de|saindo do|partindo de)\s+([a-z ]+?)\s+(?:para|pra|a)\b", t)
    if m_de:
        o = _cidades_na_ordem(m_de.group(1))
        origem = o[0] if o else None
    resto = [c for c in cidades if c != origem]
    destino = resto[0] if resto else None

    data_ida = data_volta = None
    m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m_iso:
        data_ida = date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
    for nome_mes, num in ([] if m_iso else MESES.items()):
        if nome_mes in t:
            ano = hoje.year + (1 if num < hoje.month else 0)
            data_ida = date(ano, num, 1)
            break
    m_dias = re.search(r"(\d+)\s*(?:dias|noites)", t)
    if data_ida and m_dias:
        data_volta = data_ida + timedelta(days=int(m_dias.group(1)))

    viajantes = None
    m_v = re.search(r"(\d+|um|uma|dois|duas|tres|quatro|cinco)\s+(?:pessoas|adultos|viajantes)", t)
    m_mais = re.search(r"eu e mais (\d+|um|uma|dois|duas|tres|quatro)", t)
    if m_mais:
        v = m_mais.group(1)
        viajantes = 1 + (int(v) if v.isdigit() else NUM_PALAVRAS[v])
    elif m_v:
        v = m_v.group(1)
        viajantes = int(v) if v.isdigit() else NUM_PALAVRAS[v]
    elif re.search(r"\b(casal|eu e (minha|meu))\b", t):
        viajantes = 2

    orcamento = moeda = None
    m_o = (re.search(r"(?:orcamento|ate|gastar)[^\d]{0,20}(r\$|€|us\$)?\s*(\d[\d.]*)\s*(mil\b)?\s*(reais|euros|dolares)?", t)
           or re.search(r"(r\$|€|us\$)?\s*(\d[\d.]*)\s*(mil\b)?\s*(reais|euros|dolares)", t))
    if m_o:
        orcamento = float(m_o.group(2).replace(".", "")) * (1000 if m_o.group(3) else 1)
        sinal = (m_o.group(1) or m_o.group(4) or "r$")
        moeda = {"r$": "BRL", "reais": "BRL", "€": "EUR", "euros": "EUR",
                 "us$": "USD", "dolares": "USD"}.get(sinal, "BRL")

    interesses = sorted({i.rstrip("s") if i == "museus" else i for i in INTERESSES if i in t})

    if any(p in t for p in ["clima", "tempo", "chover", "temperatura", "frio", "calor", "sol "]) and not data_ida:
        intencao = "consultar_clima"
    elif any(p in t for p in ["cambio", "cotacao", "converter"]) or len(moedas_no_texto(t)) >= 2:
        intencao = "consultar_cambio"
    elif any(p in t for p in ["visto", "bagagem", "mala", "documento", "passaporte", "seguro", "power bank"]):
        intencao = "consultar_regras"
    elif "voo" in t or "passagem" in t:
        intencao = "buscar_voo"
    elif "hotel" in t or "hospedagem" in t:
        intencao = "buscar_hotel"
    elif destino:
        intencao = "planejar_viagem"
    else:
        intencao = "outro"

    etapas = {
        "planejar_viagem": ["verificar documentos e regras de entrada",
                            "buscar voos", "buscar hospedagem",
                            "consultar previsão do tempo", "converter orçamento para moeda local",
                            "montar roteiro diário"],
        "consultar_clima": ["identificar cidade e data", "consultar previsão do tempo"],
        "consultar_cambio": ["identificar moedas", "consultar cotação atual"],
        "consultar_regras": ["identificar destino", "buscar regras na base de conhecimento"],
        "buscar_voo": ["identificar origem, destino e data", "consultar voos no banco"],
        "buscar_hotel": ["identificar cidade e datas", "consultar hotéis no banco"],
        "outro": ["pedir esclarecimento ao usuário"],
    }[intencao]

    return {
        "intencao": intencao,
        "origem": origem,
        "destino": destino,
        "data_ida": data_ida.isoformat() if data_ida else None,
        "data_volta": data_volta.isoformat() if data_volta else None,
        "num_viajantes": viajantes,
        "orcamento": orcamento,
        "moeda_orcamento": moeda,
        "interesses": interesses,
        "etapas": etapas,
    }


# --------------------------------------------------------------------------
# Planejamento de chamadas de ferramenta (usado pelo ReAct e pelo tool calling)
# --------------------------------------------------------------------------
_MOEDAS_TXT = [
    (r"\b(reais|real|brl|r\$)", "BRL"), (r"\b(euros?|eur)\b|€", "EUR"), (r"\b(dolar(es)?|usd|us\$)", "USD"),
    (r"\b(libras?|gbp)\b", "GBP"), (r"\b(ienes?|jpy)\b", "JPY"), (r"\b(pesos? argentinos?|ars)\b", "ARS"),
]


def moedas_no_texto(t: str) -> list[str]:
    achadas = []
    for padrao, cod in _MOEDAS_TXT:
        m = re.search(padrao, t)
        if m:
            achadas.append((m.start(), cod))
    return [c for _, c in sorted(achadas)]


def _valor_no_texto(t: str) -> Optional[float]:
    m = (re.search(r"(?:r\$|us\$|€)\s*(\d[\d.,]*)\s*(mil\b)?", t)
         or re.search(r"(\d[\d.,]*)\s*(mil\b)?\s*(?:reais|real|euros?|dolares|libras|pesos|brl|eur|usd|gbp)\b", t)
         or re.search(r"(\d[\d.,]*)\s*(mil\b)?", t))
    if not m:
        return None
    v = m.group(1).rstrip(".,")
    v = v.replace(".", "").replace(",", ".") if ("," in v or re.fullmatch(r"\d{1,3}(\.\d{3})+", v)) else v
    return float(v) * (1000 if m.group(2) else 1)


def planejar_chamadas(texto: str, historico: str = "", disponiveis: Optional[list[str]] = None) -> list[dict]:
    """Decide quais ferramentas chamar, com quais argumentos, imitando o raciocínio do LLM."""
    from viagens.apis import MOEDA_POR_CIDADE

    t = normalizar(texto)
    p = extrair_pedido(texto)
    destino = p["destino"]
    if not destino and historico:
        anteriores = _cidades_na_ordem(normalizar(historico))
        destino = anteriores[-1] if anteriores else None
    data = p["data_ida"]
    if "amanha" in t:
        data = (date.today() + timedelta(days=1)).isoformat()
    elif re.search(r"daqui a uma semana|semana que vem|proxima semana", t):
        data = (date.today() + timedelta(days=7)).isoformat()

    intencao = p["intencao"]
    moedas, valor = moedas_no_texto(t), _valor_no_texto(t)
    quer_clima = bool(destino) and any(w in t for w in
                                       ["clima", "tempo", "chover", "chuva", "frio", "calor", "temperatura", "sol"])
    quer_cambio = any(w in t for w in ["cambio", "cotacao", "converter"]) or bool(moedas and valor)
    quer_regras = any(w in t for w in ["visto", "bagagem", "mala", "documento", "passaporte", "seguro",
                                       "regra", "exig", "menor", "iof", "cadastro", "autorizacao", "vacina"])
    quer_banco = any(w in t for w in ["voo", "passagem", "hotel", "hospedagem", "reserva"])

    def clima(cidade, d):
        args = {"cidade": cidade, "data_inicio": d, "dias": 3}
        return {"tool": "previsao_tempo", "args": args,
                "react": f"{cidade}, {d}" if d else cidade}

    def cambio(valor, de, para):
        return {"tool": "converter_moeda", "args": {"valor": valor, "de": de, "para": para},
                "react": f"{valor:g} {de} {para}"}

    def regras(q):
        return {"tool": "buscar_regras_viagem", "args": {"consulta": q}, "react": q}

    def cambio_do_texto():
        local = MOEDA_POR_CIDADE.get(destino or "", "EUR")
        if len(moedas) >= 2:
            de, para = moedas[0], moedas[1]
        elif moedas == ["BRL"]:
            de, para = "BRL", local
        elif moedas:
            de, para = moedas[0], "BRL"
        else:
            de, para = "BRL", local
        if p["orcamento"] and (p["moeda_orcamento"] or "BRL") == de:
            return cambio(p["orcamento"], de, para)
        return cambio(valor or 1.0, de, para)

    chamadas = []
    if intencao == "planejar_viagem" and destino and not (quer_clima or quer_regras or quer_banco):
        chamadas.append(regras(f"regras de entrada e documentos para {destino}"))
        chamadas.append(clima(destino, data))
        if p["orcamento"]:
            chamadas.append(cambio(p["orcamento"], p["moeda_orcamento"] or "BRL",
                                   MOEDA_POR_CIDADE.get(destino, "EUR")))
    else:
        if quer_regras:
            chamadas.append(regras(texto.strip() + (f" {destino}" if destino and normalizar(destino) not in t else "")))
        if quer_clima:
            chamadas.append(clima(destino, data))
        if quer_cambio:
            chamadas.append(cambio_do_texto())
        if quer_banco:
            chamadas.append({"tool": "consultar_banco", "args": {"pergunta": texto.strip()}, "react": texto.strip()})

    if disponiveis is not None:
        chamadas = [c for c in chamadas if c["tool"] in disponiveis]
    return chamadas


def _brl(v: float) -> str:
    """Formato numérico brasileiro: 8000.5 -> '8.000,50'."""
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def resumir_observacao(tool: str, obs: str) -> str:
    """Transforma a saída JSON de uma ferramenta em uma frase (resposta final do dummy)."""
    try:
        d = json.loads(obs)
    except (json.JSONDecodeError, TypeError):
        m = re.match(r"\[fonte: ([^\]]+)\]\s*(.*)", str(obs), flags=re.S)
        if m:   # saída de create_retriever_tool com document_prompt
            return f"Segundo a base [fonte: {m.group(1)}]: {' '.join(m.group(2).split())[:250]}..."
        return f"{tool}: {str(obs)[:250]}"
    if tool == "previsao_tempo" and isinstance(d, dict):
        dias = "; ".join(f"{x['data']}: {x['temp_min']}–{x['temp_max']} °C, {x['condicao']}" for x in d["dias"])
        tipo = "referência histórica" if d.get("tipo") == "referencia_historica" else "previsão"
        return f"Tempo em {d['cidade']} ({tipo}, Open-Meteo): {dias}."
    if tool == "converter_moeda" and isinstance(d, dict):
        return (f"{_brl(d['valor'])} {d['de']} = {_brl(d['convertido'])} {d['para']} "
                f"(cotação {d.get('data_cotacao', '')}, {d['fonte']}).")
    if tool == "consultar_banco" and isinstance(d, dict):
        return str(d.get("resposta") or "Sem resultado no banco.").replace("[dummy] ", "")
    if tool == "estimar_custo" and isinstance(d, dict) and isinstance(d.get("body"), dict) and "total_brl" in d["body"]:
        b = d["body"]
        return f"Custo total estimado: R$ {_brl(b['total_brl'])} (R$ {_brl(b['por_pessoa_brl'])} por pessoa) [função de custo]."
    if tool == "listar_reservas" and isinstance(d, dict):
        return "Reservas: " + "; ".join(f"#{r['id']} {r['tipo']} {r['status']}" for r in d.get("reservas", [])) + " [API de reservas]."
    if tool == "buscar_regras_viagem" and isinstance(d, list) and d:
        t0 = d[0]
        return f"Segundo '{t0['titulo']}' ({t0['fonte']}): {t0['trecho'][:280].replace(chr(10), ' ')}..."
    return f"{tool}: {str(d)[:250]}"


# --------------------------------------------------------------------------
# Regras: (nome, predicado sobre o prompt, gerador de resposta)
# --------------------------------------------------------------------------
Rule = tuple[str, Callable[[str], bool], Callable[[str, List[BaseMessage]], str]]


def _ultimo_humano(msgs: List[BaseMessage]) -> str:
    for m in reversed(msgs):
        if m.type == "human":
            return str(m.content)
    return str(msgs[-1].content) if msgs else ""


def _entre(texto: str, ini: str, fim: str) -> str:
    if ini not in texto:
        return ""
    trecho = texto.split(ini, 1)[1]
    return trecho.split(fim, 1)[0] if fim in trecho else trecho


def _regra_react(prompt: str, msgs: List[BaseMessage]) -> str:
    """ReAct clássico: emite um passo Thought/Action por chamada, ou a Final Answer."""
    ferramentas = re.findall(r"should be one of \[([^\]]*)\]", prompt)
    disponiveis = [f.strip() for f in ferramentas[-1].split(",")] if ferramentas else None
    idx = prompt.rfind("\nQuestion:")
    pergunta_e_rascunho = prompt[idx + len("\nQuestion:"):]
    pergunta = pergunta_e_rascunho.split("\nThought:")[0].strip()
    historico = _entre(prompt[:idx], "Histórico da conversa:", "\nQuestion:")
    observacoes = re.findall(r"Action: (\S+)\s*\nAction Input:.*?\nObservation: (.*?)(?=\nThought:|\Z)",
                             pergunta_e_rascunho, flags=re.S)
    plano = planejar_chamadas(pergunta, historico, disponiveis)
    feitos = len(observacoes)
    if feitos < len(plano):
        c = plano[feitos]
        motivo = {"previsao_tempo": "consultar o tempo no destino",
                  "converter_moeda": "converter os valores com a cotação do dia",
                  "buscar_regras_viagem": "verificar as regras na base de conhecimento",
                  "consultar_banco": "consultar o banco de dados de viagens"}.get(c["tool"], "usar uma ferramenta")
        return (f"Thought: [dummy] Etapa {feitos + 1} de {len(plano)}: preciso {motivo}.\n"
                f"Action: {c['tool']}\nAction Input: {c['react']}")
    if not plano and re.search(r"ultima (conversa|vez|sessao)|tinha planejado|planejamos antes", normalizar(pergunta)):
        ep = _entre(prompt, "Sessões ANTERIORES relevantes (memória episódica; não são a sessão atual):", "\nEstado do pedido").strip()
        return ("Thought: [dummy] A pergunta é sobre uma sessão anterior; uso a memória episódica do contexto.\n"
                "Final Answer: Pelo meu registro da sessão anterior: " + " ".join(ep.split())[:500])
    if not plano and historico.strip():
        return ("Thought: [dummy] Não preciso de ferramenta; respondo com base no histórico da conversa.\n"
                "Final Answer: Resumo do que conversamos: "
                + (_entre(prompt, "Estado do pedido da sessão ATUAL:", "\n").strip() + ". " if "Estado do pedido da sessão ATUAL:" in prompt else "")
                + " ".join(historico.split())[-400:])
    if not plano:
        return ("Thought: [dummy] O pedido não traz destino nem valor suficiente para usar uma ferramenta.\n"
                "Final Answer: Para ajudar, preciso saber o destino (e, se for o caso, datas e orçamento) da viagem.")
    partes = [resumir_observacao(tool, obs.strip()) for tool, obs in observacoes]
    return "Thought: [dummy] Já tenho as informações necessárias.\nFinal Answer: " + " ".join(partes)


def _regra_router(prompt: str, msgs: List[BaseMessage]) -> str:
    entrada = _entre(prompt, "<< INPUT >>", "<< OUTPUT").strip()
    intencao = extrair_pedido(entrada)["intencao"]
    destino = {"consultar_clima": "clima", "consultar_cambio": "cambio", "consultar_regras": "regras",
               "planejar_viagem": "planejamento"}.get(intencao, "DEFAULT")
    return "```json\n" + json.dumps({"destination": destino, "next_inputs": entrada}, ensure_ascii=False) + "\n```"


def _regra_fixing(prompt: str, msgs: List[BaseMessage]) -> str:
    """OutputFixingParser: conserta JSON 'estilo Python' (aspas simples, vírgula sobrando)."""
    import ast

    completion = _entre(prompt, "Completion:\n--------------\n", "\n--------------").strip()
    try:
        return json.dumps(ast.literal_eval(completion), ensure_ascii=False)
    except (ValueError, SyntaxError):
        corrigido = re.sub(r",\s*([}\]])", r"\1", completion.replace("'", '"'))
        return corrigido


def _regra_formatar(prompt: str, msgs: List[BaseMessage]) -> str:
    texto = _entre(prompt, "Texto do agente:", "\n\n###").strip()
    fontes = [n for n in ["previsao_tempo", "converter_moeda", "buscar_regras_viagem", "Open-Meteo",
                          "Frankfurter"] if n.lower() in texto.lower()]
    return json.dumps({"resposta": texto or "[dummy] sem conteúdo", "fontes": fontes,
                       "confianca": "media" if fontes else "baixa", "precisa_mais_info": not fontes},
                      ensure_ascii=False)


def _regra_resumo(prompt: str, msgs: List[BaseMessage]) -> str:
    # o prompt do LangChain traz um EXEMPLO antes; o bloco real é sempre o último
    if "Resumo atual:" in prompt:   # prompt em português do projeto
        m_atual, m_novas, m_fim = "Resumo atual:", "Novas linhas da conversa:", "Novo resumo:"
    else:                           # prompt padrão do LangChain
        m_atual, m_novas, m_fim = "Current summary:", "New lines of conversation:", "New summary:"
    ultimo = prompt[prompt.rfind(m_atual):]
    atual = _entre(ultimo, m_atual, m_novas).strip()
    novas = _entre(ultimo, m_novas, m_fim).strip()
    frases = []
    for linha in novas.splitlines():
        if linha.startswith("Human:"):
            frases.append("O usuário pediu: " + linha[6:].strip()[:120] + ".")
        elif linha.startswith("AI:"):
            frases.append("O assistente respondeu: " + linha[3:].strip()[:120] + ".")
    return ("[dummy] " + (atual.replace("[dummy] ", "") + " " + " ".join(frases)).strip())[:900]


def _regra_sintese(prompt: str, msgs: List[BaseMessage]) -> str:
    blocos = [l.strip() for l in prompt.splitlines() if l.strip().startswith(("CLIMA:", "CAMBIO:", "REGRAS:"))]
    return "[dummy] Síntese do plano:\n" + "\n".join(f"- {b[:300]}" for b in blocos)


def _regra_rag(prompt: str, msgs: List[BaseMessage]) -> str:
    contexto = _entre(prompt, "Contexto:", "Pergunta:").strip()
    primeiro = contexto.split("\n\n")[0] if contexto else ""
    return "[dummy] Com base na base de conhecimento: " + " ".join(primeiro.split())[:400]


_PREFS = [
    (r"(prefiro|sempre prefiro|so viajo em|gosto de) voos? diretos?", "preferencia_voo", "voos diretos"),
    (r"(sou|estou|virei) vegan[ao]", "restricao_alimentar", "vegana"),
    (r"(sou|estou) vegetarian[ao]", "restricao_alimentar", "vegetariana"),
    (r"prefiro hote(l|is)|nao gosto de hostel", "preferencia_hospedagem", "hotel (não hostel)"),
    (r"viajo (sempre )?com (meu|minha) (\w+)", "companhia_viagem", None),
    (r"tenho passaporte (europeu|italiano|portugues)", "documentos", None),
]


def preferencias_no_texto(texto: str) -> list[tuple[str, str, str]]:
    t = normalizar(texto)
    achadas = []
    for padrao, chave, valor in _PREFS:
        m = re.search(padrao, t)
        if m and chave not in [c for c, _, _ in achadas]:
            achadas.append((chave, valor or m.group(0), m.group(0)))
    return achadas


def _regra_analise(prompt: str, msgs: List[BaseMessage]) -> str:
    """NLU da Fase 3: intenção + entidades por regras."""
    msg = prompt[prompt.rfind("Mensagem atual:") + len("Mensagem atual:"):].strip()
    t = normalizar(msg)
    p = extrair_pedido(msg)
    prefs = preferencias_no_texto(msg)
    duracao = re.search(r"(\d+)\s*(?:dias|noites)", t)
    viajantes = p["num_viajantes"] or (1 if re.search(r"\bso eu\b|sozinh", t) else None)
    estado_txt = _entre(prompt, "Estado atual do pedido:", "Histórico recente:")
    tem_destino = '"destino"' in estado_txt
    tem_dados = any([p["data_ida"], duracao, viajantes, p["orcamento"]])

    if re.search(r"ultima (conversa|vez|sessao)|tinha planejado|planejamos antes", t):
        intencao = "consultar_memoria"
    elif re.search(r"\b(na verdade|mudamos|mudei|e se|trocar|em vez de)\b", t) and (p["destino"] or tem_dados):
        intencao = "alterar_pedido"
    elif re.search(r"obrigad", t):
        intencao = "agradecimento"
    elif p["intencao"] in ("consultar_clima", "consultar_cambio", "consultar_regras"):
        intencao = p["intencao"]
    elif p["destino"]:
        intencao = "planejar_viagem"
    elif tem_dados and tem_destino:
        intencao = "fornecer_dados"
    elif prefs:
        intencao = "informar_preferencia"
    elif re.search(r"^(oi|ola|bom dia|boa tarde|boa noite)\b", t):
        intencao = "saudacao"
    else:
        intencao = "fora_escopo"

    return json.dumps({
        "intencao": intencao,
        "entidades": {"destino": p["destino"], "origem": p["origem"], "data_ida": p["data_ida"],
                      "duracao_dias": int(duracao.group(1)) if duracao else None, "num_viajantes": viajantes,
                      "orcamento": p["orcamento"], "moeda_orcamento": p["moeda_orcamento"]},
        "preferencias": [frase for _, _, frase in prefs],
        "subtarefas": p["etapas"] if intencao == "planejar_viagem" else [],
    }, ensure_ascii=False)


def _regra_fatos(prompt: str, msgs: List[BaseMessage]) -> str:
    return json.dumps({"fatos": [{"chave": c, "valor": v} for c, v, _ in preferencias_no_texto(_ultimo_humano(msgs))]},
                      ensure_ascii=False)


def _regra_episodio(prompt: str, msgs: List[BaseMessage]) -> str:
    estado = _entre(prompt, "Estado final do pedido:", "\n").strip()
    return f"[dummy] Sessão de planejamento de viagem. Pedido final: {estado}."


# Glossário que imita o "conhecimento de mundo" do LLM na reescrita de consultas
_GLOSSARIO = [
    (r"\b(barcelona|madri)\b", "Espanha, Espaço Schengen"), (r"\bparis\b", "França, Espaço Schengen"),
    (r"\b(roma|milao)\b", "Itália, Espaço Schengen"), (r"\bamsterda\b", "Holanda, Espaço Schengen"),
    (r"\b(lisboa|porto)\b", "Portugal, Espaço Schengen"), (r"\beuropa\b", "Espaço Schengen 90 dias"),
    (r"\b(inglaterra|londres)\b", "Reino Unido, ETA autorização eletrônica de viagem"),
    (r"\b(disney|orlando|nova york)\b", "Estados Unidos, visto"),
    (r"\b(bariloche|buenos aires)\b", "Argentina, Mercosul, RG"), (r"\bchile\b", "Mercosul, RG"),
    (r"\bgringa\b", "exterior, IOF"), (r"\bimposto\b", "IOF"), (r"\bidentidade\b", "RG carteira de identidade"),
    (r"carregador portatil", "power bank, bateria de lítio, bagagem de mão"), (r"\bcnh\b", "documento de identidade"),
    (r"cadastro online", "autorização eletrônica"), (r"\bvacina\b", "febre amarela certificado"),
]


def expandir_consulta(pergunta: str) -> str:
    t = normalizar(pergunta)
    extras = [exp for padrao, exp in _GLOSSARIO if re.search(padrao, t)]
    return pergunta + (" (" + "; ".join(extras) + ")" if extras else "")


def _regra_reescrita(prompt: str, msgs: List[BaseMessage]) -> str:
    return expandir_consulta(_ultimo_humano(msgs))


def _regra_multi(prompt: str, msgs: List[BaseMessage]) -> str:
    q = prompt.split("Pergunta:")[-1].strip()
    exp = expandir_consulta(q)
    return "\n".join([exp, exp.split("(")[-1].rstrip(")") if "(" in exp else q, f"regras de entrada {q}"])


def _regra_hyde(prompt: str, msgs: List[BaseMessage]) -> str:
    q = prompt.split("Pergunta:")[-1].strip()
    return f"[dummy] Guia oficial: para brasileiros, {expandir_consulta(q)}: confira documentos, visto e seguro exigidos."


def _regra_sql(prompt: str, msgs: List[BaseMessage]) -> str:
    """Text-to-SQL por padrões (imita o LLM para os tipos de pergunta do projeto)."""
    from viagens.banco import AEROPORTOS

    q = _ultimo_humano(msgs).split("Pergunta:")[-1].strip()
    t = normalizar(q)
    if re.search(r"\b(apague|apagar|delete|remova|remover|altere|alterar|cancele)\b", t):
        return "RECUSAR"
    iata = {normalizar(c): i for i, (c, _) in AEROPORTOS.items()}
    iata["rio"] = "GIG"
    cidades = [c for c in _cidades_na_ordem(t)]
    mes = next((n for nome, n in MESES.items() if nome in t), None)
    if "reserva" in t:
        filtro = " AND status = 'confirmada'" if "confirmad" in t else ""
        return ("SELECT id, tipo, status, data_inicio, ROUND(valor_total_brl, 2) AS valor_total_brl FROM reservas "
                f"WHERE usuario_id = 'viajante_demo'{filtro} LIMIT 20")
    if "hote" in t:
        conds = [f"cidade = '{cidades[-1]}'"] if cidades else []
        if "vegan" in t:
            conds.append("opcao_vegana = 1")
        if "cafe" in t:
            conds.append("cafe_incluso = 1")
        m = re.search(r"nota (?:acima de|maior que) (\d+)", t)
        if m:
            conds.append(f"nota > {m.group(1)}")
        m = re.search(r"(\d) estrelas", t)
        if m:
            conds.append(f"estrelas >= {m.group(1)}")
        where = " WHERE " + " AND ".join(conds) if conds else ""
        return f"SELECT nome, bairro, estrelas, diaria_brl, nota FROM hoteis{where} ORDER BY nota DESC LIMIT 20"
    codigos = [iata.get(normalizar(c)) for c in cidades if iata.get(normalizar(c))]
    origem, destino = (codigos[0], codigos[1]) if len(codigos) >= 2 else ("GRU", codigos[0] if codigos else "LIS")
    conds = [f"origem = '{origem}'", f"destino = '{destino}'", "classe = 'economica'"]
    if re.search(r"diret|sem escala", t):
        conds.append("escalas = 0")
    if mes and "media" not in t:
        ano = 2026 if mes == 12 else 2027
        conds.append(f"strftime('%Y-%m', data_partida) = '{ano}-{mes:02d}'")
    if "media" in t:
        return ("SELECT strftime('%Y-%m', data_partida) AS mes, ROUND(AVG(preco_brl), 2) AS preco_medio FROM voos WHERE "
                + " AND ".join(conds) + " GROUP BY mes ORDER BY mes LIMIT 20")
    limite = 1 if "mais barato" in t else 5
    return ("SELECT id, numero, data_partida, hora_partida, escalas, preco_brl FROM voos WHERE " + " AND ".join(conds)
            + f" ORDER BY preco_brl LIMIT {limite}")


def _regra_sql_corrigir(prompt: str, msgs: List[BaseMessage]) -> str:
    return _entre(_ultimo_humano(msgs), "SQL com erro:", "\nErro").strip()


def _regra_sql_resposta(prompt: str, msgs: List[BaseMessage]) -> str:
    res = _ultimo_humano(msgs).split("Resultado SQL (colunas e linhas):")[-1].strip()
    try:
        d = json.loads(res)
    except json.JSONDecodeError:
        d = None
    if not d or not d.get("linhas"):
        return "[dummy] Não há registros que atendam ao pedido. [banco: viagens.db]"
    linhas = "; ".join(", ".join(f"{c}={v}" for c, v in zip(d["colunas"], l)) for l in d["linhas"][:5])
    return f"[dummy] Resultado do banco: {linhas}. [banco: viagens.db]"


def _regra_juiz(prompt: str, msgs: List[BaseMessage]) -> str:
    """LLM-juiz simulado: proporção de fatos-chave presentes na resposta."""
    h = _ultimo_humano(msgs)
    fatos = [f.strip() for f in _entre(h, "FATOS-CHAVE:", "\nRESPOSTA:").split(",") if f.strip() and f.strip() != "-"]
    resposta = normalizar(h.split("RESPOSTA:")[-1])
    if not fatos:
        return json.dumps({"nota": 0.5, "justificativa": "[dummy] sem fatos-chave; nota neutra"})
    ok = sum(normalizar(f) in resposta for f in fatos)
    nota = 1 if ok == len(fatos) else (0.5 if ok else 0)
    return json.dumps({"nota": nota, "justificativa": f"[dummy] {ok}/{len(fatos)} fatos-chave presentes"})


def _regra_extracao(prompt: str, msgs: List[BaseMessage]) -> str:
    pedido = _ultimo_humano(msgs).split("Pedido:")[-1]
    return json.dumps(extrair_pedido(pedido), ensure_ascii=False, indent=2)


def _regra_dica(prompt: str, msgs: List[BaseMessage]) -> str:
    cidades = _cidades_na_ordem(normalizar(prompt))
    destino = cidades[0] if cidades else "o destino"
    return (f"[dummy] Em {destino}, compre o transporte público em passe semanal "
            f"e reserve as atrações mais procuradas com antecedência.")


RULES: list[Rule] = [
    ("react", lambda p: "Action Input:" in p and "Final Answer:" in p, _regra_react),
    ("nlu_analise", lambda p: "AnaliseMensagem" in p, _regra_analise),
    ("fatos_memoria", lambda p: "FatosMemoria" in p, _regra_fatos),
    ("resumo_episodio", lambda p: "resumo do episodio" in normalizar(p), _regra_episodio),
    ("reescrita_consulta", lambda p: "Reescreva a pergunta do viajante" in p, _regra_reescrita),
    ("multi_query", lambda p: "Gere 3 versoes" in p, _regra_multi),
    ("hyde", lambda p: "Escreva um trecho de um guia oficial" in p, _regra_hyde),
    ("text_to_sql", lambda p: "Você converte perguntas sobre viagens em UMA consulta SQL" in p, _regra_sql),
    ("sql_corrigir", lambda p: "Você corrige consultas SQL" in p, _regra_sql_corrigir),
    ("sql_resposta", lambda p: "Resultado SQL (colunas e linhas)" in p, _regra_sql_resposta),
    ("juiz", lambda p: "Você é o avaliador de um assistente de viagens" in p, _regra_juiz),
    ("router", lambda p: "<< CANDIDATE PROMPTS >>" in p, _regra_router),
    ("output_fixing", lambda p: "did not satisfy the constraints" in p, _regra_fixing),
    ("formatar_resposta", lambda p: "RespostaViagem" in p, _regra_formatar),
    ("resumo_memoria", lambda p: "Progressively summarize" in p or "Resuma progressivamente" in p, _regra_resumo),
    ("sintese_plano", lambda p: "sintese do plano" in normalizar(p), _regra_sintese),
    ("rag_resposta", lambda p: "usando apenas o contexto" in normalizar(p), _regra_rag),
    ("extracao_pedido", lambda p: "pedidoviagem" in normalizar(p) or '"intencao"' in p, _regra_extracao),
    ("dica_destino", lambda p: "de uma dica" in normalizar(p), _regra_dica),
]


def register_rule(name: str, predicate: Callable[[str], bool],
                  responder: Callable[[str, List[BaseMessage]], str], first: bool = True) -> None:
    """Permite que outras fases adicionem regras ao modelo dummy."""
    global RULES
    RULES = [r for r in RULES if r[0] != name]
    RULES.insert(0, (name, predicate, responder)) if first else RULES.append((name, predicate, responder))


class DummyChatModel(BaseChatModel):
    """Chat model determinístico que segue RULES e suporta tool calling."""

    model_name: str = "dummy-travel-llm"

    @property
    def _llm_type(self) -> str:
        return "dummy-travel"

    # tokens aproximados (4 caracteres por token), para as memórias com limite de tokens
    def get_num_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _responder(self, messages: List[BaseMessage]) -> str:
        prompt = "\n".join(str(m.content) for m in messages)
        for _, pred, resp in RULES:
            if pred(prompt):
                return resp(prompt, messages)
        return "[dummy] Entendi. Pode me dar mais detalhes sobre a sua viagem?"

    def _responder_com_tools(self, messages: List[BaseMessage], tools: list) -> AIMessage:
        nomes = [t["function"]["name"] for t in tools]
        i_humano = max(i for i, m in enumerate(messages) if m.type == "human")
        posteriores = messages[i_humano + 1:]
        resultados = [m for m in posteriores if m.type == "tool"]
        if resultados:
            # create_openai_tools_agent não preenche ToolMessage.name: recupera pelo tool_call_id
            nome_por_id = {tc["id"]: tc["name"] for m in posteriores if m.type in ("ai", "AIMessageChunk")
                           for tc in (getattr(m, "tool_calls", None) or [])}
            partes = [resumir_observacao(getattr(m, "name", None) or nome_por_id.get(m.tool_call_id, ""), str(m.content))
                      for m in resultados]
            return AIMessage(content="[dummy] " + " ".join(partes))
        historico = "\n".join(str(m.content) for m in messages[:i_humano] if m.type in ("human", "ai"))
        pergunta = str(messages[i_humano].content)
        plano = planejar_chamadas(pergunta, historico, nomes)
        if not plano and len(nomes) == 1:       # agente com uma só ferramenta (ex.: agente de recuperação)
            plano = [{"tool": nomes[0], "args": {"consulta": pergunta}}]
        if not plano:
            return AIMessage(content=self._responder(messages))
        props = {t["function"]["name"]: list(t["function"].get("parameters", {}).get("properties", {})) for t in tools}
        for c in plano:                          # adapta o nome do argumento ao schema real da ferramenta
            params = props.get(c["tool"], [])
            if params and not set(c["args"]) <= set(params) and len(params) == 1:
                c["args"] = {params[0]: next(iter(c["args"].values()))}
        return AIMessage(content="", tool_calls=[
            {"name": c["tool"], "args": c["args"], "id": f"call_dummy_{k}", "type": "tool_call"}
            for k, c in enumerate(plano)])

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        if kwargs.get("tools"):
            msg = self._responder_com_tools(messages, kwargs["tools"])
        else:
            msg = AIMessage(content=self._responder(messages))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if kwargs.get("tools"):
            msg = self._responder_com_tools(messages, kwargs["tools"])
            if msg.tool_calls:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"], ensure_ascii=False), "id": c["id"], "index": k}
                    for k, c in enumerate(msg.tool_calls)]))
                return
            texto = str(msg.content)
        else:
            texto = self._responder(messages)
        for token in re.findall(r"\S+\s*|\s+", texto):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk
