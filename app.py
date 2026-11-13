"""Interface Streamlit do Assistente de Viagens (Viaja+).

Executar:  streamlit run app.py
Usa o mesmo agente integrador da Fase 5 (viagens/agente_final.py): executor,
memória (RunnableWithMessageHistory + perfil de longo prazo) e verificação NLI.
"""
from __future__ import annotations

import uuid

import streamlit as st

st.set_page_config(page_title="Viaja+ · Assistente de Viagens", page_icon="✈️", layout="wide")

from viagens.config import USE_DUMMY  # noqa: E402  (depois do set_page_config)

ROTULOS = {"SUSTENTADA": "✅ sustentada", "CONTRADITA": "❌ contradita", "NAO_VERIFICAVEL": "➖ não verificável",
           "NUMERO_SEM_EVIDENCIA": "⚠️ número sem evidência"}


@st.cache_resource(show_spinner="Carregando agente, índice FAISS e modelo de verificação…")
def carregar_agente(usuario_id: str, autocorrecao: bool):
    from viagens.agente_final import AgenteViagens

    return AgenteViagens(usuario_id=usuario_id, autocorrecao=autocorrecao)


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("✈️ Viaja+")
    st.caption("Assistente de viagens com LangChain · Shaiane Alves Tonin")
    st.markdown(f"**Modo:** {'🧪 dummy (sem chaves)' if USE_DUMMY else '🔑 real (OpenAI + APIs)'}")
    usuario = st.text_input("Usuário (memória de longo prazo)", value="viajante_demo")
    autocorrecao = st.toggle("Autocorreção quando a verificação reprovar", value=True)
    mostrar_detalhes = st.toggle("Mostrar trajetória e verificação", value=True)
    if "sessao" not in st.session_state:
        st.session_state.sessao = f"st-{uuid.uuid4().hex[:8]}"
        st.session_state.mensagens = []
    if st.button("Nova conversa", width="stretch"):
        st.session_state.sessao = f"st-{uuid.uuid4().hex[:8]}"
        st.session_state.mensagens = []
    st.caption(f"Sessão: `{st.session_state.sessao}`")

    agente = carregar_agente(usuario, autocorrecao)
    with st.expander("🧠 Perfil do viajante (memória semântica)"):
        perfil = agente.longo.perfil
        if perfil:
            for chave, info in perfil.items():
                st.markdown(f"- **{chave}**: {info['valor']}")
        else:
            st.caption("Nada registrado ainda. Conte suas preferências (ex.: 'sou vegetariana').")
        if perfil and st.button("🗑️ Esquecer meu perfil", width="stretch"):
            agente.longo.apagar_tudo()   # direito ao esquecimento: remove perfil e episódios deste usuário
            st.rerun()
    st.markdown("**Experimente:**")
    st.caption("• Brasileiro precisa de visto para ir a Roma?\n\n• Qual o voo direto mais barato de São Paulo "
               "para Paris em maio de 2027?\n\n• Tenho 10 mil reais para 5 dias lá: quanto dá por dia em euros?\n\n"
               "• Quais são as minhas reservas?")


def painel_detalhes(r: dict):
    v = r.get("verificacao") or {}
    cols = st.columns(4)
    cols[0].metric("Ferramentas", len(r["ferramentas"]))
    cols[1].metric("Fundamentação (NLI)", f"{v.get('fundamentacao', 0):.0%}" if v else "—")
    cols[2].metric("Verificação", "aprovada" if v.get("aprovada") else "reprovada" if v else "—")
    cols[3].metric("Tempo", f"{r['latencia_s']:.1f}s")
    if r["ferramentas"]:
        st.markdown("**Trajetória de ferramentas**")
        for i, (tool, entrada) in enumerate(r["ferramentas"], 1):
            st.code(f"{i}. {tool}({entrada})", language="text")
    if v.get("afirmacoes"):
        st.markdown(f"**Verificação afirmação por afirmação** (backend NLI: `{v.get('backend_nli')}`)")
        st.dataframe([{"afirmação": a["texto"], "resultado": ROTULOS.get(a["rotulo"], a["rotulo"]),
                       "P(sustenta)": a["p_sustenta"], "P(contradiz)": a["p_contradiz"],
                       "evidência mais próxima": a["evidencia"]} for a in v["afirmacoes"]],
                     width="stretch", hide_index=True)
    if r.get("autocorrigida"):
        st.warning("A primeira resposta foi reprovada pela verificação e reescrita com base nas evidências.")
        with st.expander("Ver a resposta original (reprovada)"):
            st.write(r["resposta_original"])
    if r.get("memoria_semantica"):
        st.info("Memória de longo prazo atualizada: " + ", ".join(f"{op} {c}={val}" for op, c, val in r["memoria_semantica"]))


# ------------------------------------------------------------------ conversa
st.header("Converse com a Viaja+")
for m in st.session_state.mensagens:
    with st.chat_message(m["papel"]):
        st.markdown(m["texto"])
        if m["papel"] == "assistant" and mostrar_detalhes and m.get("detalhes"):
            with st.expander("🔎 Como cheguei a esta resposta"):
                painel_detalhes(m["detalhes"])

pergunta = st.chat_input("Pergunte sobre clima, câmbio, documentos, voos, hotéis ou reservas…")
if pergunta:
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)
    with st.chat_message("assistant"):
        with st.spinner("Pesquisando e verificando…"):
            r = agente.responder(pergunta, st.session_state.sessao)
        detalhes = {"ferramentas": r.ferramentas, "verificacao": r.verificacao, "latencia_s": r.latencia_s,
                    "autocorrigida": r.autocorrigida, "resposta_original": r.resposta_original,
                    "memoria_semantica": r.memoria_semantica}
        st.markdown(r.resposta)
        if mostrar_detalhes:
            with st.expander("🔎 Como cheguei a esta resposta", expanded=True):
                painel_detalhes(detalhes)
    st.session_state.mensagens.append({"papel": "assistant", "texto": r.resposta, "detalhes": detalhes})
