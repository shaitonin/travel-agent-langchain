"""Agente integrador (Fase 5).

- Executor: create_openai_tools_agent + AgentExecutor.
- Ferramentas: APIs externas (clima, câmbio), RAG com reescrita de consulta
  (Fase 4), text-to-SQL, API interna de reservas, função serverless e calculadora.
- Memória de curto prazo: RunnableWithMessageHistory (histórico por session_id),
  com janela por tokens (trim_messages) antes do prompt.
- Memória de longo prazo: perfil semântico e episódios da Fase 3
  (MemoriaLongoPrazo), recuperados por busca vetorial a cada turno.
- Verificação: toda resposta passa pelo Verificador (NLI + números + coerência).
  Se for reprovada, o agente é chamado de novo com as evidências e os problemas
  apontados (autocorreção, 1x). Se continuar reprovada, a resposta sai com aviso.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, trim_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from viagens.agente_react import DIAS_SEMANA
from viagens.config import get_llm
from viagens.memoria import CHAVES_PERFIL, MemoriaLongoPrazo
from viagens.verificacao import Verificador, evidencias_da_trajetoria

SISTEMA_PTCF = """### PERSONA
Você é a Viaja+, agente de viagens sênior que atende viajantes brasileiros. É cuidadosa com fatos: nunca inventa previsões, cotações, preços, voos ou regras. Todo dado que muda com o tempo ou que pertence a um sistema vem de uma ferramenta.

### TAREFA
Responder perguntas e planejar viagens. Decomponha pedidos compostos e chame uma ferramenta para cada parte. Chamadas INDEPENDENTES podem ir em paralelo; se uma etapa precisa do resultado de outra (ex.: converter o preço de um voo que ainda vai ser consultado), chame primeiro uma, espere o resultado e só depois a outra:
- previsao_tempo: clima (API Open-Meteo);
- converter_moeda: câmbio do dia (API Frankfurter);
- buscar_regras_viagem: regras de entrada, visto, seguro, bagagem, menores, IOF e guias (base de conhecimento);
- consultar_banco: voos, hotéis e reservas no banco de dados (SQL);
- listar_reservas / criar_reserva: API interna de reservas (só reserve se o usuário pedir explicitamente);
- estimar_custo: custo total da viagem (função serverless);
- calculadora: contas.

### CONTEXTO
- Hoje é {hoje}. A moeda do usuário é o real (BRL).
- Memória de longo prazo do usuário (perfil e sessões anteriores):
{contexto_usuario}
- Use o histórico da conversa para resolver referências ("lá", "nessa época", "esse voo").
- Se uma ferramenta falhar ou não cobrir o caso, diga isso; não complete com conhecimento próprio.

### FORMATO
Português, até 8 frases. Cada número citado deve vir de uma ferramenta. TODA resposta que usou ferramenta termina com as fontes entre colchetes, exatamente neste formato: [Open-Meteo], [Frankfurter], [fonte: arquivo.md], [banco: viagens.db], [API de reservas], [função de custo]."""

FONTE_DA_FERRAMENTA = {"previsao_tempo": "Open-Meteo", "converter_moeda": "Frankfurter", "consultar_banco": "banco: viagens.db",
                       "listar_reservas": "API de reservas", "criar_reserva": "API de reservas",
                       "estimar_custo": "função de custo"}


def garantir_fontes(resposta: str, passos) -> str:
    """Se o modelo esqueceu a citação (observado em 7/15 casos na 1ª avaliação), acrescenta as fontes
    a partir da trajetória: é a ferramenta que de fato produziu o dado, não um palpite."""
    if not passos or re.search(r"\[[^\]]+\]", resposta):
        return resposta
    fontes = []
    for acao, obs in passos:
        if acao.tool == "buscar_regras_viagem":
            fontes += [f"fonte: {f}" for f in re.findall(r"\[fonte: ([^\]]+)\]", str(obs))[:2]]
        elif acao.tool in FONTE_DA_FERRAMENTA:
            fontes.append(FONTE_DA_FERRAMENTA[acao.tool])
    fontes = list(dict.fromkeys(fontes))
    return resposta + ("\n\nFontes: " + " ".join(f"[{f}]" for f in fontes) if fontes else "")


def _hoje() -> str:
    return f"{DIAS_SEMANA[date.today().weekday()]}, {date.today().isoformat()}"


@dataclass
class ResultadoTurno:
    pergunta: str
    resposta: str
    ferramentas: List[tuple]
    verificacao: dict
    autocorrigida: bool = False
    resposta_original: Optional[str] = None
    verificacao_original: Optional[dict] = None
    memoria_semantica: List[tuple] = field(default_factory=list)
    latencia_s: float = 0.0


class AgenteViagens:
    def __init__(self, usuario_id: str = "viajante_demo", llm=None, verificar: bool = True, autocorrecao: bool = True,
                 max_tokens_historico: int = 1500, url_reservas: Optional[str] = None, nli_backend: str = "auto",
                 memoria_id: Optional[str] = None):
        """usuario_id: dono das reservas e do banco. memoria_id: perfil de longo prazo (padrão: o mesmo usuário).
        Separar os dois isola a memória da avaliação da memória de quem usa o app ao mesmo tempo."""
        from viagens.chatbot import _PROMPT_FATOS
        from viagens.recuperacao import Buscador, RetrieverTransformado
        from viagens.tools import ferramentas_estruturadas, ferramentas_fase4
        from langchain.tools.retriever import create_retriever_tool
        from langchain_core.prompts import PromptTemplate

        self.llm = llm or get_llm(temperature=0)
        self.usuario_id = usuario_id
        self.verificar, self.autocorrecao = verificar, autocorrecao
        self.longo = MemoriaLongoPrazo(memoria_id or usuario_id)
        self.verificador = Verificador(nli_backend) if verificar else None

        # RAG da Fase 4: índice existente + reescrita de consulta
        rag = create_retriever_tool(
            RetrieverTransformado(buscador=Buscador(self.llm, k=3), nome_estrategia="reescrita"),
            "buscar_regras_viagem",
            "Base de conhecimento de viagens: regras de entrada e visto por país/bloco, seguro-viagem, bagagem, "
            "viagem com menores, IOF e guias de destino. Use para qualquer pergunta sobre regras ou exigências.",
            document_prompt=PromptTemplate.from_template("[fonte: {fonte}] {page_content}"),
            document_separator="\n---\n")
        base = [t for t in ferramentas_estruturadas() if t.name != "buscar_regras_viagem"]
        self.tools = base + [rag] + ferramentas_fase4(self.llm, url_reservas=url_reservas, usuario_id=usuario_id)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SISTEMA_PTCF),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agente = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agente, tools=self.tools, return_intermediate_steps=True,
                                      max_iterations=8, handle_parsing_errors=True)

        # janela de curto prazo: só as mensagens mais recentes que cabem em max_tokens_historico
        janela = RunnablePassthrough.assign(chat_history=lambda x: trim_messages(
            x["chat_history"], max_tokens=max_tokens_historico, strategy="last", token_counter=self.llm,
            start_on="human", include_system=False))
        self._historicos: Dict[str, BaseChatMessageHistory] = {}
        self.com_memoria = RunnableWithMessageHistory(
            janela | self.executor, self.historico, input_messages_key="input",
            history_messages_key="chat_history", output_messages_key="output")
        self._fatos = (_PROMPT_FATOS | self.llm | JsonOutputParser()).with_config(run_name="extrair_fatos_memoria")

    # ----------------------------------------------------------------- memória
    def historico(self, session_id: str) -> BaseChatMessageHistory:
        return self._historicos.setdefault(session_id, InMemoryChatMessageHistory())

    def contexto_usuario(self, consulta: str) -> str:
        fatos = self.longo.buscar_fatos(consulta, k=3)
        eps = self.longo.buscar_episodios(consulta, k=1)
        linhas = [f"  - {f['chave']}: {f['valor']}" for f in fatos] or ["  - (perfil vazio)"]
        linhas += [f"  - sessão anterior [{e['data']}]: {e['resumo']}" for e in eps]
        return "\n".join(linhas)

    def _gravar_preferencias(self, mensagem: str, config) -> list:
        perfil = "\n".join(f"- {k}: {v['valor']}" for k, v in self.longo.perfil.items()) or "(vazio)"
        try:
            fatos = self._fatos.invoke({"mensagem": mensagem, "perfil": perfil}, config=config).get("fatos", [])
        except Exception:
            return []
        return [(self.longo.lembrar_fato(f["chave"], str(f["valor"])), f["chave"], f["valor"])
                for f in fatos if f.get("chave") in CHAVES_PERFIL and f.get("valor")]

    @staticmethod
    def _com_retentativa(chamada, tentativas: int = 3):
        """Quedas de conexão no meio do streaming da OpenAI não são cobertas pelo max_retries do cliente
        (observado: 'peer closed connection'); repete a chamada inteira com espera crescente."""
        import httpx
        import openai

        for i in range(tentativas):
            try:
                return chamada()
            except (httpx.RemoteProtocolError, httpx.ReadError, openai.APIConnectionError):
                if i == tentativas - 1:
                    raise
                time.sleep(2 * (i + 1))

    # ------------------------------------------------------------------- turno
    def responder(self, mensagem: str, session_id: str = "padrao", callbacks: Optional[list] = None) -> ResultadoTurno:
        t0 = time.perf_counter()
        config = {"configurable": {"session_id": session_id}, "callbacks": callbacks or []}
        ops = self._gravar_preferencias(mensagem, config)
        entrada = {"input": mensagem, "hoje": _hoje(), "contexto_usuario": self.contexto_usuario(mensagem)}
        r = self._com_retentativa(lambda: self.com_memoria.invoke(entrada, config=config))
        passos = list(r["intermediate_steps"])
        resposta = garantir_fontes(r["output"], passos)
        res = ResultadoTurno(mensagem, resposta, [(a.tool, a.tool_input) for a, _ in passos], {}, memoria_semantica=ops)
        if not self.verificar:
            res.latencia_s = time.perf_counter() - t0
            return res

        verif = self.verificador.verificar(resposta, passos)
        res.verificacao = verif.to_dict()
        if not verif.aprovada and self.autocorrecao:
            res.resposta_original, res.verificacao_original = resposta, res.verificacao
            problemas = "\n".join(f"- \"{a.texto}\" ({a.rotulo}"
                                  + (f"; números sem evidência: {a.numeros_sem_evidencia}" if a.numeros_sem_evidencia else "")
                                  + ")" for a in verif.problemas)
            problemas += "".join(f"\n- contradição interna: \"{p}\" x \"{h}\"" for p, h in verif.contradicoes_internas)
            evid = "\n".join(f"- {e}" for e in evidencias_da_trajetoria(passos)) or "- (nenhuma)"
            revisao = (f"{mensagem}\n\n[REVISÃO INTERNA] Sua resposta anterior tinha afirmações não sustentadas pelas "
                       f"evidências:\n{problemas}\n\nEvidências coletadas pelas ferramentas:\n{evid}\n\nReescreva a "
                       "resposta usando SOMENTE essas evidências (ou chame ferramentas para o que faltar). Se não houver "
                       "evidência para algo, diga que não tem essa informação.")
            hist = self.historico(session_id)
            r2 = self._com_retentativa(lambda: self.executor.invoke(
                {**entrada, "input": revisao, "chat_history": hist.messages[:-2]}, config={"callbacks": callbacks or []}))
            passos2 = passos + list(r2["intermediate_steps"])
            verif2 = self.verificador.verificar(r2["output"], passos2)
            resposta = garantir_fontes(r2["output"], passos2)
            if not verif2.aprovada:
                trechos = "; ".join(f"\"{a.texto[:80]}\"" for a in verif2.problemas) or "contradição interna"
                resposta += f"\n\n⚠️ Não consegui confirmar nas fontes consultadas: {trechos}."
            # o histórico guarda a versão corrigida, não a reprovada
            hist.messages[-1] = AIMessage(content=resposta)
            res.resposta, res.verificacao, res.autocorrigida = resposta, verif2.to_dict(), True
            res.ferramentas = [(a.tool, a.tool_input) for a, _ in passos2]
        res.latencia_s = time.perf_counter() - t0
        return res
