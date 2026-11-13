"""Agente de recuperação sobre o índice FAISS EXISTENTE + transformação de consulta.

O índice é apenas CARREGADO (viagens.conhecimento.carregar_indice); o pipeline
de indexação não é refeito aqui.

Técnicas de transformação comparadas:
  - reescrita: o LLM reescreve a pergunta no vocabulário da base
               (cidade -> país + bloco, gíria -> termo oficial);
  - multi-query: o LLM gera 3 variações; os resultados são unidos (RRF);
  - HyDE: o LLM escreve um trecho hipotético de guia que responderia à
          pergunta, e esse trecho é usado como consulta.
"""
from __future__ import annotations

from typing import Callable, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever

from viagens.conhecimento import carregar_indice

# ---------------------------------------------------------------------------
# Casos de teste difíceis (pergunta, fontes aceitas)
# ---------------------------------------------------------------------------
CASOS_DIFICEIS = [
    # cidade citada, regra escrita por país/bloco (falhas observadas no índice real)
    ("Quais documentos preciso para entrar em Barcelona?", ["schengen_regras.md"]),
    ("O que preciso para entrar em Paris?", ["schengen_regras.md"]),
    ("Quais documentos levar para Milão?", ["schengen_regras.md"]),
    ("Brasileiro precisa de visto pra Roma?", ["schengen_regras.md"]),
    ("Vou para Barcelona e depois Paris, quantos dias posso ficar?", ["schengen_regras.md"]),
    ("Que documentos preciso apresentar na imigração em Madri?", ["schengen_regras.md"]),
    ("Vou para Amsterdã, preciso de visto?", ["schengen_regras.md"]),
    # controles: vocabulário informal ou indireto (precisam continuar funcionando)
    ("Preciso fazer algum cadastro online antes de ir pra Inglaterra?", ["reino_unido_eta.md"]),
    ("Meu filho de 15 anos vai pra Disney com a avó, o que precisa?", ["menores_viagem.md", "eua_visto.md"]),
    ("Quanto de imposto pago usando o cartão na gringa?", ["dinheiro_iof.md"]),
    ("Dá pra levar carregador portátil no avião?", ["bagagem_aerosul.md"]),
    ("Vou pra Bariloche, posso ir só com a identidade?", ["mercosul_documentos.md"]),
    ("Posso usar a CNH para entrar no Chile?", ["mercosul_documentos.md"]),
    ("Vou pra Madri, preciso de seguro viagem?", ["schengen_regras.md", "saude_seguro.md"]),
    ("Preciso de visto para ir a Lisboa?", ["portugal_entrada.md", "schengen_regras.md"]),
    ("Qual o valor da ETA?", ["reino_unido_eta.md"]),
]

# ---------------------------------------------------------------------------
# Prompts das transformações
# ---------------------------------------------------------------------------
PROMPT_REESCRITA = ChatPromptTemplate.from_messages([
    ("system",
     "Reescreva a pergunta do viajante para a busca na base de conhecimento de regras de viagem. A base é escrita "
     "em linguagem oficial e organizada por PAÍS e por BLOCO (Espaço Schengen, Mercosul, Reino Unido, Estados "
     "Unidos), e não por cidade. Regras:\n"
     "- troque ou complemente cidades pelo país e pelo bloco (ex.: Milão -> Itália, Espaço Schengen);\n"
     "- troque gírias e termos informais pelos termos oficiais (ex.: 'na gringa' -> 'no exterior'; 'carregador "
     "portátil' -> 'power bank, bateria de lítio'; 'identidade' -> 'RG, carteira de identidade');\n"
     "- mantenha o sentido e o sujeito (brasileiro); não responda à pergunta.\n"
     "Devolva só a pergunta reescrita, em uma linha."),
    ("human", "{pergunta}"),
])

PROMPT_MULTI = PromptTemplate.from_template(
    "Gere 3 versoes diferentes da pergunta do viajante para buscar em uma base de regras de viagem organizada por "
    "país e bloco (Espaço Schengen, Mercosul...). Varie o vocabulário (termos oficiais, país em vez de cidade). "
    "Uma versão por linha, sem numeração.\nPergunta: {pergunta}")

PROMPT_HYDE = PromptTemplate.from_template(
    "Escreva um trecho de um guia oficial de viagens para brasileiros (4 a 6 frases, linguagem formal) que "
    "responderia à pergunta abaixo. Mencione o país e o bloco de países relevantes.\nPergunta: {pergunta}")


def _rrf(listas: List[List[Document]], k: int, c: int = 60) -> List[Document]:
    """Reciprocal Rank Fusion: une rankings de várias consultas sem depender da escala dos scores."""
    pontos, docs = {}, {}
    for lista in listas:
        for pos, d in enumerate(lista):
            cid = d.metadata.get("chunk_id", d.page_content[:60])
            pontos[cid] = pontos.get(cid, 0) + 1 / (c + pos + 1)
            docs[cid] = d
    return [docs[cid] for cid in sorted(pontos, key=pontos.get, reverse=True)[:k]]


class Buscador:
    """Agrupa o índice existente e as quatro estratégias de consulta."""

    def __init__(self, llm, k: int = 3):
        self.vs = carregar_indice()
        self.k = k
        self._reescrever = (PROMPT_REESCRITA | llm | StrOutputParser()).with_config(run_name="reescrita_consulta")
        self._multi = (PROMPT_MULTI | llm | StrOutputParser()).with_config(run_name="multi_query")
        self._hyde = (PROMPT_HYDE | llm | StrOutputParser()).with_config(run_name="hyde")
        self.ultima_consulta: dict = {}

    def base(self, pergunta: str, config=None) -> List[Document]:
        self.ultima_consulta = {"estrategia": "base", "consultas": [pergunta]}
        return self.vs.similarity_search(pergunta, k=self.k)

    def reescrita(self, pergunta: str, config=None) -> List[Document]:
        q = self._reescrever.invoke({"pergunta": pergunta}, config=config).strip()
        self.ultima_consulta = {"estrategia": "reescrita", "consultas": [q]}
        return self.vs.similarity_search(q, k=self.k)

    def multi_query(self, pergunta: str, config=None) -> List[Document]:
        variacoes = [l.strip("-• ").strip() for l in self._multi.invoke({"pergunta": pergunta}, config=config).splitlines()
                     if l.strip()][:3]
        consultas = [pergunta] + variacoes
        self.ultima_consulta = {"estrategia": "multi_query", "consultas": consultas}
        return _rrf([self.vs.similarity_search(q, k=self.k) for q in consultas], self.k)

    def hyde(self, pergunta: str, config=None) -> List[Document]:
        trecho = self._hyde.invoke({"pergunta": pergunta}, config=config).strip()
        self.ultima_consulta = {"estrategia": "hyde", "consultas": [trecho]}
        return self.vs.similarity_search(trecho, k=self.k)

    def estrategia(self, nome: str) -> Callable:
        return {"base": self.base, "reescrita": self.reescrita, "multi_query": self.multi_query, "hyde": self.hyde}[nome]


def avaliar(buscar: Callable, casos=CASOS_DIFICEIS, config=None) -> list:
    """Para cada caso: fontes devolvidas, acerto no top-1, no top-k e reciprocal rank."""
    linhas = []
    for pergunta, esperadas in casos:
        docs = buscar(pergunta, config=config)
        fontes = [d.metadata["fonte"] for d in docs]
        pos = next((i + 1 for i, f in enumerate(fontes) if f in esperadas), None)
        linhas.append({"pergunta": pergunta, "esperado": esperadas, "fontes": fontes,
                       "hit@1": pos == 1, "hit@k": pos is not None, "rr": 1 / pos if pos else 0.0})
    return linhas


class RetrieverTransformado(BaseRetriever):
    """Retriever LangChain que aplica uma estratégia do Buscador antes de buscar no FAISS."""

    buscador: Buscador
    nome_estrategia: str = "reescrita"

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.buscador.estrategia(self.nome_estrategia)(query, config={"callbacks": run_manager.get_child()})


def criar_agente_recuperacao(llm, retriever, nome_tool: str = "buscar_regras_viagem"):
    """Agente de tool calling cuja única ferramenta é o retriever (índice FAISS existente)."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools.retriever import create_retriever_tool

    tool = create_retriever_tool(
        retriever, nome_tool,
        "Busca na base de conhecimento de regras de viagem (entrada, visto, seguro, bagagem, documentos de menores, "
        "IOF, guias de destino). Use para qualquer pergunta sobre regras e exigências.",
        document_prompt=PromptTemplate.from_template("[fonte: {fonte}] {page_content}"),
        document_separator="\n---\n",
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é o agente de recuperação do assistente de viagens. SEMPRE consulte a ferramenta "
                   f"{nome_tool} antes de responder e responda SOMENTE com base nos trechos devolvidos. Em até 3 "
                   "frases, cite a fonte entre colchetes, como [fonte: arquivo.md]. Se os trechos não trouxerem a "
                   "resposta, diga que a base não cobre o caso."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agente = create_tool_calling_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agente, tools=[tool], return_intermediate_steps=True, max_iterations=4)
