# Viaja+ · Assistente de Viagens com LangChain Agents

Projeto da disciplina de LangChain Agents. **Autora:** Shaiane Alves Tonin

É um agente de viagens que planeja, responde e verifica as próprias respostas. Ele raciocina sobre a tarefa (ReAct e tool calling), usa ferramentas reais (APIs externas de clima e câmbio, base de conhecimento com FAISS, banco SQL, API interna e função serverless), mantém memória de curto e de longo prazo e confere cada resposta contra as evidências com um modelo de NLI. A avaliação de qualidade e de trajetória é feita no LangSmith.

## Como rodar

### Opção 1: sem chaves (modo dummy, recomendado para avaliação)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # já vem com USE_DUMMY=true
jupyter notebook notebooks/     # rode os notebooks 01 → 05 em ordem
```
No modo dummy, o LLM é substituído por um `DummyChatModel` determinístico (`viagens/dummy.py`), que implementa a mesma interface Runnable (invoke/batch/stream/tool calling). As APIs de clima e câmbio são simuladas, os embeddings são lexicais e o LangSmith é desligado (a avaliação roda localmente, com os mesmos avaliadores). O banco SQLite, o índice FAISS, o microsserviço, a função serverless e o modelo de NLI funcionam igual.

### Opção 2: com chaves
No `.env`, preencha `OPENAI_API_KEY` e `LANGSMITH_API_KEY` e use `USE_DUMMY=false`. Se a conta do LangSmith for da região EU, use `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.

### Google Colab
Envie a pasta do projeto para `/content/projeto-langchain-agentes` (ou monte o Drive). A primeira célula de cada notebook detecta o Colab, entra na pasta e instala o `requirements.txt`.

### Interface Streamlit
```bash
streamlit run app.py
```

> Na primeira execução da Fase 5, o modelo de NLI (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, cerca de 1 GB) é baixado do Hugging Face. Sem internet, o verificador cai automaticamente para um backend heurístico. Para forçar esse backend, use `Verificador("heuristico")`.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `notebooks/01_mvp_chain.ipynb` | Fase 1: anatomia e arquitetura do agente, prompts, chain com saída JSON, `@chain` vs. LCEL |
| `notebooks/02_agente_react.ipynb` | Fase 2: orquestradores, ReAct (`initialize_agent`), function calling, PTCF, RouterChain, chains paralelas, parsers, memória, FAISS, callbacks |
| `notebooks/03_memoria_chatbot.ipynb` | Fase 3: CoALA, janela de contexto, LangMem/Mem0/Letta, chatbot com `ConversationBufferMemory`, intenção e entidades, memória de longo prazo |
| `notebooks/04_ferramentas_rag.ipynb` | Fase 4: ferramentas por tipo, API interna, serverless, text-to-SQL, agente de recuperação, diagnóstico de falha e transformação de consulta |
| `notebooks/05_agente_integrador.ipynb` | Fase 5: verificação por NLI, agente integrador (`create_openai_tools_agent` + `RunnableWithMessageHistory`), avaliação no LangSmith, Streamlit |
| `viagens/` | Pacote compartilhado por todos os notebooks (detalhes abaixo) |
| `app.py` | Interface Streamlit |
| `data/docs/` | Base de conhecimento (14 documentos didáticos) |
| `data/faiss_index/` | Índice FAISS (construído na Fase 2 e só **carregado** nas fases seguintes) |
| `data/viagens.db` | Banco SQLite fictício (voos, hotéis, reservas), recriado por `viagens.banco.criar_banco()` |
| `avaliacao/` | Resultados exportados da avaliação (CSV e gráfico) |
| `logs/agente.log` | Log JSON das execuções (callback) |

### Pacote `viagens/`
`config` (modo real/dummy) · `dummy` (LLM simulado) · `apis` (Open-Meteo, Frankfurter) · `tools` (todas as ferramentas) · `conhecimento` (FAISS) · `extracao` (chain da Fase 1) · `agente_react` (ReAct + PTCF) · `callbacks` · `memoria` (longo prazo) · `chatbot` (Fase 3) · `banco` + `sql` (text-to-SQL) · `servicos/` (API interna e função serverless) · `recuperacao` (RAG + transformação de consulta) · `verificacao` (NLI) · `agente_final` (integrador) · `avaliacao` (dataset e avaliadores).

## Registro de avaliação no LangSmith (links públicos)
- Dataset `viajamais-avaliacao-v1` com os experimentos A (sem autocorreção) e B (com autocorreção): https://eu.smith.langchain.com/public/8c8c3c18-ec39-4273-a998-6913582989f4/d
- Trace de exemplo, caso c11 (`consultar_banco` → `converter_moeda`): https://eu.smith.langchain.com/public/6290cd73-d83d-45f6-8ef0-c8319e3c6ca8/r/9846be1c-f2d0-479f-b9a9-9816437ec056?start_time=2026-11-13T18%3A37%3A33.291563Z
- Relatório técnico: `relatorio/shaiane_tonin_ai-agents_langchain-agents.pdf`

## Observações
- Dados fictícios: a companhia AeroSul, os voos, os hotéis e as reservas. Os documentos de regras de viagem são **didáticos** e trazem o aviso para conferir as fontes oficiais.
- No macOS, `viagens/config.py` importa o `torch` antes do `faiss`, para evitar o conflito de `libomp` entre os dois (isso não ocorre no Linux/Colab).
