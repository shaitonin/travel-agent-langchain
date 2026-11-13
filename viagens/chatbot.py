"""Chatbot de viagens (Fase 3): agente ReAct + memória de curto e longo prazo.

Fluxo de cada turno
-------------------
1. NLU: uma chain LCEL reconhece a INTENÇÃO e extrai ENTIDADES da mensagem,
   usando o estado do pedido e o histórico recente para resolver referências
   ("lá", "nessa época").
2. Slot filling: as entidades novas atualizam o estado do pedido da sessão.
3. Memória semântica: se a mensagem traz preferências estáveis, uma chain
   extrai fatos chave->valor, gravados no padrão ADD/UPDATE/NOOP.
4. Política de diálogo: faltam dados obrigatórios -> pergunta de coleta;
   pedido completo -> plano via agente ReAct; demais intenções -> agente.
5. Curto prazo: o turno vai para a ConversationBufferMemory. Se a janela
   passar do limite de tokens, as mensagens antigas são resumidas e saem do
   buffer (sliding window + sumarização).
Ao encerrar a sessão, um resumo vira memória EPISÓDICA.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import List, Literal, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field, field_validator

from viagens.agente_react import criar_agente_react
from viagens.config import get_llm
from viagens.extracao import _no_futuro
from viagens.memoria import CHAVES_PERFIL, MemoriaLongoPrazo
from viagens.tools import ferramentas_react

# ---------------------------------------------------------------------------
# NLU: intenção + entidades
# ---------------------------------------------------------------------------
INTENCOES = {
    "planejar_viagem": "quer planejar uma viagem (destino, com ou sem detalhes)",
    "fornecer_dados": "responde dados que faltavam (datas, duração, pessoas, orçamento) sem mudar o pedido",
    "alterar_pedido": "muda algo já definido (outro destino, outra data, outro orçamento)",
    "consultar_clima": "pergunta sobre tempo/clima",
    "consultar_cambio": "pergunta sobre câmbio/conversão",
    "consultar_regras": "pergunta sobre visto, documentos, seguro, bagagem, regras",
    "consultar_memoria": "pergunta sobre conversas ou viagens anteriores",
    "informar_preferencia": "só informa preferências/restrições pessoais",
    "saudacao": "cumprimento",
    "agradecimento": "agradece ou encerra",
    "fora_escopo": "assunto que não é viagem",
}


class Entidades(BaseModel):
    destino: Optional[str] = Field(None, description="Cidade de destino")
    origem: Optional[str] = Field(None, description="Cidade de origem")
    data_ida: Optional[str] = Field(None, description="YYYY-MM-DD; se só o mês for dito, dia 01; sempre no futuro")
    duracao_dias: Optional[int] = Field(None, description="Duração da viagem em dias")
    num_viajantes: Optional[int] = Field(None, description="Número total de viajantes, só se dito nesta mensagem")
    orcamento: Optional[float] = Field(None, description="Orçamento total, só o número")
    moeda_orcamento: Optional[str] = Field(None, description="BRL, EUR, USD...")


class AnaliseMensagem(BaseModel):
    intencao: Literal[tuple(INTENCOES)] = Field(description="Intenção principal da mensagem")
    entidades: Entidades = Field(default_factory=Entidades,
                                 description="Somente entidades ditas ou alteradas NESTA mensagem")
    preferencias: List[str] = Field(default_factory=list,
                                    description="Somente preferências/restrições PESSOAIS e PERMANENTES ditas NESTA mensagem (dieta, tipo de "
                                                "voo, hospedagem, acessibilidade, documentos). NÃO inclua dados desta "
                                                "viagem (com quem vai, orçamento, datas, destino)")
    subtarefas: List[str] = Field(default_factory=list,
                                  description="Se o pedido for composto, as etapas para atendê-lo")

    @field_validator("preferencias", "subtarefas", mode="before")
    @classmethod
    def _nulo_vira_lista(cls, v):
        return [] if v is None else v

    @field_validator("entidades", mode="before")
    @classmethod
    def _nulo_vira_vazio(cls, v):
        return {} if v is None else v


_parser_analise = JsonOutputParser(pydantic_object=AnaliseMensagem)
_PROMPT_ANALISE = ChatPromptTemplate.from_messages([
    ("system",
     "Você é o módulo de NLU (AnaliseMensagem) do chatbot de viagens. Hoje é {hoje}.\n"
     "Classifique a intenção da MENSAGEM ATUAL e extraia as entidades. Use o estado do pedido e o histórico "
     "para resolver referências ('lá', 'nessa época', 'o mesmo orçamento'), mas em 'entidades' coloque apenas o "
     "que foi dito ou alterado NESTA mensagem: nunca deduza um valor que a mensagem não diz. Datas sempre no futuro.\n"
     "Intenções possíveis:\n{intencoes}\n\nResponda só com o JSON.\n{format_instructions}"),
    ("human", "Estado atual do pedido:\n{estado}\n\nHistórico recente:\n{historico}\n\nMensagem atual:\n{mensagem}"),
]).partial(intencoes="\n".join(f"- {k}: {v}" for k, v in INTENCOES.items()),
           format_instructions=_parser_analise.get_format_instructions())

# ---------------------------------------------------------------------------
# Escrita na memória semântica
# ---------------------------------------------------------------------------
_PROMPT_FATOS = ChatPromptTemplate.from_messages([
    ("system",
     "Extraia FatosMemoria: fatos ESTÁVEIS sobre o viajante, que valeriam também para viagens FUTURAS. "
     "Grave só hábitos e características pessoais ('sou', 'sempre', 'costumo', 'prefiro', 'nunca').\n"
     "NÃO são fatos estáveis (ignore): com quem vai NESTA viagem ('eu e meu marido vamos', 'só eu'), o orçamento "
     "DESTA viagem ('temos 15 mil'), datas e destino.\n"
     "Chaves permitidas:\n{chaves}\n"
     "Perfil atual do viajante:\n{perfil}\n"
     "Se o fato já existe com o mesmo sentido, repita EXATAMENTE o valor atual. Se a mensagem muda um fato "
     "(ex.: 'agora sou vegana'), use a mesma chave com o novo valor. Valores canônicos e curtos, ex.: "
     "'voos diretos', 'vegetariana'. Responda só com JSON no formato "
     '{{"fatos": [{{"chave": "...", "valor": "..."}}]}}; lista vazia se não houver.'),
    ("human", "{mensagem}"),
]).partial(chaves="\n".join(f"- {k}: {v}" for k, v in CHAVES_PERFIL.items()))

# ---------------------------------------------------------------------------
# Sumarização (janela da sessão) e episódio
# ---------------------------------------------------------------------------
_PROMPT_RESUMO = PromptTemplate.from_template(
    "Resuma progressivamente a conversa entre o viajante (Human) e o assistente (AI), acrescentando as novas "
    "linhas ao resumo atual. Preserve SEMPRE: destino, datas, número de viajantes, orçamento, preferências e "
    "conclusões já obtidas (valores convertidos, regras). Escreva em português e responda APENAS com o texto "
    "do novo resumo, sem títulos.\n\nResumo atual:\n{summary}\n\nNovas linhas da conversa:\n{new_lines}\n\nNovo resumo:")

_PROMPT_EPISODIO = ChatPromptTemplate.from_messages([
    ("system", "Escreva o resumo do episodio desta sessão de atendimento, em até 4 frases e em português: "
               "destinos considerados, datas, viajantes, orçamento, valores e regras encontrados, decisões e "
               "pendências. Não repita preferências permanentes (elas já ficam no perfil)."),
    ("human", "Data: {data}\nEstado final do pedido: {estado}\n\nResumo dos turnos antigos: {resumo}\n\n"
              "Últimas mensagens:\n{mensagens}"),
])

OBRIGATORIOS = {"destino": "o destino", "data_ida": "a data (ou o mês) de ida",
                "duracao_dias": "quantos dias de viagem", "num_viajantes": "quantas pessoas vão"}
INTENCOES_PLANO = {"planejar_viagem", "fornecer_dados", "alterar_pedido"}


class ChatbotViagens:
    def __init__(self, usuario_id: str, llm=None, limite_tokens_janela: int = 900,
                 manter_mensagens: int = 4, callbacks: Optional[list] = None):
        self.llm = llm or get_llm(temperature=0)
        self.cfg = {"callbacks": callbacks or []}
        self.usuario_id = usuario_id
        self.sessao_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.memoria_curta = ConversationBufferMemory(memory_key="chat_history", input_key="input",
                                                      output_key="output")
        self.longo = MemoriaLongoPrazo(usuario_id)
        self.agente = criar_agente_react(self.llm, ferramentas_react())
        self.estado: dict = {}
        self.resumo_sessao = ""
        self.limite_tokens = limite_tokens_janela
        self.manter = manter_mensagens
        self.registros: list = []

        self._analisar = (_PROMPT_ANALISE | self.llm | _parser_analise).with_config(run_name="nlu_intencao_entidades")
        self._fatos = (_PROMPT_FATOS | self.llm | JsonOutputParser()).with_config(run_name="extrair_fatos_memoria")
        self._resumir = (_PROMPT_RESUMO | self.llm | StrOutputParser()).with_config(run_name="resumir_janela")
        self._episodio = (_PROMPT_EPISODIO | self.llm | StrOutputParser()).with_config(run_name="resumo_episodio")

    # ------------------------------------------------------------- leitura
    def historico_texto(self) -> str:
        return self.memoria_curta.load_memory_variables({})["chat_history"] or "(início da conversa)"

    def contexto_usuario(self, consulta: str):
        fatos = self.longo.buscar_fatos(consulta, k=3)
        episodios = self.longo.buscar_episodios(consulta, k=1)
        linhas = ["Perfil do viajante (memória semântica, recuperada por similaridade):"]
        linhas += [f"- {f['chave']}: {f['valor']}" for f in fatos] or ["- (nada registrado)"]
        linhas.append("Sessões ANTERIORES relevantes (memória episódica; não são a sessão atual):")
        linhas += [f"- [{e['data']}] {e['resumo']}" for e in episodios] or ["- (nenhuma)"]
        estado = ", ".join(f"{k}={v}" for k, v in self.estado.items()) or "(vazio)"
        linhas.append(f"Estado do pedido da sessão ATUAL: {estado}")
        if self.resumo_sessao:
            linhas.append(f"Resumo dos turnos antigos desta sessão: {self.resumo_sessao}")
        return "\n".join(linhas), {"fatos": fatos, "episodios": episodios}

    # -------------------------------------------------------------- etapas
    def _nlu(self, mensagem: str) -> AnaliseMensagem:
        msgs = self.memoria_curta.chat_memory.messages[-4:]
        hist = "\n".join(f"{m.type}: {str(m.content)[:300]}" for m in msgs) or "(nenhum)"
        try:
            bruto = self._analisar.invoke({"mensagem": mensagem, "hoje": date.today().isoformat(),
                                           "estado": json.dumps(self.estado, ensure_ascii=False) or "{}",
                                           "historico": hist}, config=self.cfg)
            return AnaliseMensagem.model_validate(bruto)
        except Exception as e:  # NLU falhou: segue pelo agente, que lida com texto livre
            self._erro_nlu = str(e)[:200]
            return AnaliseMensagem(intencao="fora_escopo")

    def _atualizar_estado(self, ent: Entidades) -> dict:
        novos = {k: v for k, v in ent.model_dump().items() if v not in (None, "", 0)}
        if "data_ida" in novos:
            novos["data_ida"] = _no_futuro(novos["data_ida"])
        mudancas = {k: (self.estado.get(k), v) for k, v in novos.items() if self.estado.get(k) != v}
        self.estado.update(novos)
        return mudancas

    def _gravar_preferencias(self, mensagem: str) -> list:
        try:
            perfil = "\n".join(f"- {k}: {v['valor']}" for k, v in self.longo.perfil.items()) or "(vazio)"
            fatos = self._fatos.invoke({"mensagem": mensagem, "perfil": perfil}, config=self.cfg).get("fatos", [])
        except Exception:
            return []
        ops = []
        for f in fatos:
            if f.get("chave") in CHAVES_PERFIL and f.get("valor"):
                ops.append((self.longo.lembrar_fato(f["chave"], str(f["valor"]), self.sessao_id), f["chave"], f["valor"]))
        return ops

    def _pedido_planejamento(self, mensagem: str) -> str:
        e = self.estado
        partes = [f"Planeje a viagem para {e['destino']}", f"com ida em {e['data_ida']}",
                  f"por {e['duracao_dias']} dias", f"para {e['num_viajantes']} pessoa(s)"]
        if e.get("orcamento"):
            partes.append(f"com orçamento total de {e['orcamento']:g} {e.get('moeda_orcamento') or 'BRL'}")
        if e.get("origem"):
            partes.append(f"saindo de {e['origem']}")
        return (", ".join(partes) + ". Execute TODAS as etapas com as ferramentas, uma por ação, e só então "
                "escreva a Final Answer com o plano completo: (1) regras de entrada e documentos do país; "
                "(2) clima do período; (3) converter o orçamento para a moeda local e calcular o valor por dia; "
                "(4) duas recomendações coerentes com as preferências do perfil. "
                f"Mensagem original do usuário: \"{mensagem}\"")

    def _etapas_obrigatorias(self) -> dict:
        etapas = {"buscar_regras_viagem": "regras de entrada e documentos", "previsao_tempo": "clima do período"}
        if self.estado.get("orcamento"):
            etapas["converter_moeda"] = "conversão do orçamento"
        return etapas

    def _executar_plano(self, entrada: str, contexto: str):
        """Camada deliberativa: o agente executa o plano; se pular etapas obrigatórias,
        é chamado de novo com a lista do que falta (no máximo 1 nova tentativa)."""
        r = self.agente.invoke({"input": entrada, "chat_history": self.historico_texto(),
                                "contexto_usuario": contexto}, config=self.cfg)
        passos = list(r["intermediate_steps"])
        usadas = {a.tool for a, _ in passos}
        faltam = {k: v for k, v in self._etapas_obrigatorias().items() if k not in usadas}
        if not faltam:
            return r["output"], passos, []
        reforco = (entrada + "\n\nATENÇÃO: na tentativa anterior a resposta foi escrita SEM executar estas etapas: "
                   + "; ".join(f"{v} (ferramenta {k})" for k, v in faltam.items())
                   + ". Execute TODAS as etapas com as ferramentas agora; não use números que não vieram de uma ferramenta.")
        r2 = self.agente.invoke({"input": reforco, "chat_history": self.historico_texto(),
                                 "contexto_usuario": contexto}, config=self.cfg)
        passos += list(r2["intermediate_steps"])
        usadas = {a.tool for a, _ in passos}
        ainda = [k for k in faltam if k not in usadas]
        if not ainda:
            return r2["output"], passos, list(faltam)

        # 2ª falha: o orquestrador executa as etapas pendentes com os argumentos do estado
        # e o agente só sintetiza a partir dos dados coletados.
        from langchain_core.agents import AgentAction
        from viagens.apis import MOEDA_POR_CIDADE

        e = self.estado
        tools = {t.name: t for t in ferramentas_react()}
        entradas = {"buscar_regras_viagem": f"regras de entrada e documentos para {e['destino']}",
                    "previsao_tempo": f"{e['destino']}, {e['data_ida']}, {min(int(e['duracao_dias']), 7)}",
                    "converter_moeda": f"{e.get('orcamento', 0):g} {e.get('moeda_orcamento') or 'BRL'} "
                                       f"{MOEDA_POR_CIDADE.get(e['destino'], 'EUR')}"}
        coletados = []
        for nome in ainda:
            obs = tools[nome].run(entradas[nome], callbacks=self.cfg["callbacks"])
            passos.append((AgentAction(tool=nome, tool_input=entradas[nome], log="[orquestrador]"), obs))
            coletados.append(f"- {nome}({entradas[nome]}): {obs[:1500]}")
        sintese = (entrada + "\n\nDados JÁ coletados pelas ferramentas (use-os e cite as fontes; não invente outros "
                   "números):\n" + "\n".join(coletados))
        r3 = self.agente.invoke({"input": sintese, "chat_history": self.historico_texto(),
                                 "contexto_usuario": contexto}, config=self.cfg)
        return r3["output"], passos + list(r3["intermediate_steps"]), list(faltam)

    def _gerenciar_janela(self) -> bool:
        msgs = self.memoria_curta.chat_memory.messages
        if self.llm.get_num_tokens_from_messages(msgs) <= self.limite_tokens or len(msgs) <= self.manter:
            return False
        antigas, recentes = msgs[:-self.manter], msgs[-self.manter:]
        linhas = "\n".join(f"{'Human' if m.type == 'human' else 'AI'}: {m.content}" for m in antigas)
        self.resumo_sessao = self._resumir.invoke({"summary": self.resumo_sessao or "(vazio)",
                                                   "new_lines": linhas}, config=self.cfg).strip()
        self.memoria_curta.chat_memory.messages = list(recentes)
        return True

    # --------------------------------------------------------------- turno
    def turno(self, mensagem: str) -> dict:
        analise = self._nlu(mensagem)
        mudancas = self._atualizar_estado(analise.entidades)
        ops_memoria = self._gravar_preferencias(mensagem) if analise.preferencias else []
        faltantes = [k for k in OBRIGATORIOS if not self.estado.get(k)]

        passos, recuperado, etapas_refeitas = [], None, []
        if analise.intencao in INTENCOES_PLANO and faltantes:
            acao = "coletar_dados"
            conhecido = f" para {self.estado['destino']}" if self.estado.get("destino") else ""
            resposta = (f"Ótimo! Para planejar sua viagem{conhecido}, preciso saber: "
                        + "; ".join(OBRIGATORIOS[k] for k in faltantes) + ". (O orçamento também ajuda, se tiver.)")
        elif analise.intencao == "informar_preferencia" and ops_memoria:
            acao = "registrar_preferencia"
            resposta = "Anotado! Vou lembrar disso nas próximas viagens: " + "; ".join(
                f"{CHAVES_PERFIL[c].split(' (')[0].lower()}: {v}" for _, c, v in ops_memoria) + "."
        else:
            consulta = f"{mensagem} {self.estado.get('destino', '')}"
            contexto, recuperado = self.contexto_usuario(consulta)
            if analise.intencao in INTENCOES_PLANO:
                acao = "planejar_com_agente"
                resposta, passos, etapas_refeitas = self._executar_plano(self._pedido_planejamento(mensagem), contexto)
            else:
                acao = "responder_com_agente"
                entrada = mensagem
                if analise.intencao == "consultar_memoria":
                    # o registro episódico vai DENTRO da pergunta: só no contexto, o modelo às vezes
                    # confundia "a última conversa" com a conversa atual (observado na execução real)
                    registros = "\n".join(f"- [{e['data']}] {e['resumo']}" for e in recuperado["episodios"]) \
                        or "- (nenhuma sessão anterior registrada)"
                    entrada += ("\n\nRegistros das sessões ANTERIORES (memória episódica):\n" + registros +
                                "\nResponda com base nesses registros; o pedido da sessão atual NÃO é a resposta.")
                r = self.agente.invoke({"input": entrada, "chat_history": self.historico_texto(),
                                        "contexto_usuario": contexto}, config=self.cfg)
                resposta, passos = r["output"], r["intermediate_steps"]

        self.memoria_curta.save_context({"input": mensagem}, {"output": resposta})
        janela_resumida = self._gerenciar_janela()

        reg = {"turno": len(self.registros) + 1, "mensagem": mensagem, "intencao": analise.intencao,
               "entidades": {k: v for k, v in analise.entidades.model_dump().items() if v is not None},
               "preferencias": analise.preferencias, "subtarefas": analise.subtarefas,
               "mudancas_estado": mudancas, "estado": dict(self.estado), "faltantes": faltantes,
               "memoria_semantica": ops_memoria, "memorias_recuperadas": recuperado, "acao": acao,
               "etapas_refeitas": etapas_refeitas,
               "ferramentas": [(a.tool, str(a.tool_input)[:120]) for a, _ in passos], "resposta": resposta,
               "tokens_janela": self.llm.get_num_tokens_from_messages(self.memoria_curta.chat_memory.messages),
               "janela_resumida": janela_resumida}
        self.registros.append(reg)
        return reg

    def encerrar_sessao(self) -> str:
        msgs = "\n".join(f"{m.type}: {str(m.content)[:400]}" for m in self.memoria_curta.chat_memory.messages)
        resumo = self._episodio.invoke({"data": date.today().isoformat(),
                                        "estado": json.dumps(self.estado, ensure_ascii=False),
                                        "resumo": self.resumo_sessao or "(nenhum)", "mensagens": msgs},
                                       config=self.cfg).strip()
        self.longo.registrar_episodio(resumo, self.sessao_id)
        return resumo
