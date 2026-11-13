"""Verificação de respostas: consistência factual (NLI) + números + coerência lógica.

Para cada afirmação da resposta do agente:
  1. NLI (premissa = trecho de evidência, hipótese = afirmação) -> sustentação,
     contradição ou neutralidade. Modelo: mDeBERTa-v3 multilíngue treinado em
     XNLI/MNLI (funciona em português).
  2. Checagem numérica: modelos de NLI são fracos com números (no teste,
     "3.100 euros" foi 'sustentado' por uma evidência que dizia 2.409,97).
     Todo valor relevante citado precisa existir nas evidências, diretamente
     ou como divisão/multiplicação simples (ex.: total / dias).
Coerência lógica: NLI entre pares de afirmações da PRÓPRIA resposta, para
detectar contradições internas.

Evidência = o que as ferramentas devolveram nesta execução (trajetória),
convertido em frases (JSON de API não é linguagem natural).

Backends de NLI: 'hf' (modelo local, padrão) ou 'heuristico' (sobreposição
lexical, usado só quando o modelo não pode ser carregado, ex.: sem internet).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import List, Optional

from viagens.dummy import normalizar

MODELO_NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LIMIAR = 0.5
STOP = set("a o e de do da dos das em no na nos nas um uma para por com que se ao à os as ou é são "
           "mais seu sua você voce isso essa esse este esta há ter".split())

# frases sem conteúdo verificável (cortesia, convite)
_NAO_FACTUAL = re.compile(r"^(se precisar|estou à disposição|boa viagem|aproveite|espero ter|posso ajudar|"
                          r"conte comigo|qualquer dúvida|fico à disposição|por nada)", re.I)


# ---------------------------------------------------------------------------
# Evidências: observações das ferramentas -> frases
# ---------------------------------------------------------------------------
def observacao_em_frases(tool: str, entrada, obs) -> List[str]:
    texto = str(getattr(obs, "content", obs))
    try:
        d = json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        d = None
    frases = []
    if tool == "previsao_tempo" and isinstance(d, dict) and "dias" in d:
        tipo = ("referência histórica do mesmo período no ano anterior" if d.get("tipo") == "referencia_historica"
                else "previsão do tempo")
        for x in d["dias"]:
            frases.append(f"Em {d['cidade']}, {d.get('pais', '')}, no dia {x['data']} ({tipo}), a temperatura máxima é "
                          f"{x['temp_max']} °C e a mínima {x['temp_min']} °C, com {x['condicao']} e {x['chuva_mm']} mm "
                          f"de chuva" + (f" e {x['prob_chuva_%']}% de probabilidade de chuva" if 'prob_chuva_%' in x else "")
                          + ". Fonte: Open-Meteo.")
    elif tool == "converter_moeda" and isinstance(d, dict) and "convertido" in d:
        nome = {"BRL": "reais", "EUR": "euros", "USD": "dólares", "GBP": "libras", "JPY": "ienes", "CHF": "francos suíços"}
        frases.append(f"{d['valor']:g} {nome.get(d['de'], d['de'])} ({d['de']}) equivalem a {d['convertido']} "
                      f"{nome.get(d['para'], d['para'])} ({d['para']}) hoje, com taxa de {d.get('taxa')} "
                      f"(cotação de {d.get('data_cotacao', 'hoje')}). Fonte: {d.get('fonte')}.")
    elif tool == "estimar_custo" and isinstance(d, dict) and "body" in d:
        b = d["body"]
        if "itens_brl" in b:
            itens = ", ".join(f"{k} R$ {v}" for k, v in b["itens_brl"].items())
            frases.append(f"Custo estimado da viagem: {itens}; total de R$ {b['total_brl']} e R$ {b['por_pessoa_brl']} por "
                          f"pessoa. {b.get('premissas', '')}")
    elif tool == "consultar_banco" and isinstance(d, dict):
        res = d.get("resultado") or {}
        for linha in (res.get("linhas") or [])[:10]:
            frases.append("Registro do banco de viagens: " + "; ".join(f"{c} = {_com_cidade(v)}"
                                                                      for c, v in zip(res["colunas"], linha)) + ".")
        if res and not res.get("linhas"):
            frases.append("A consulta ao banco de viagens não retornou registros.")
    elif tool == "listar_reservas" and isinstance(d, dict):
        for r in d.get("reservas", []):
            frases.append("Reserva do usuário: " + "; ".join(f"{k} = {_com_cidade(v)}" for k, v in r.items()
                                                               if v is not None) + ".")
    elif tool == "calculadora":
        frases.append(f"O resultado do cálculo {entrada} é {texto}.")
    elif isinstance(d, list):   # buscar_regras_viagem no formato JSON (Fase 2)
        for t in d:
            frases += _quebrar(t.get("trecho", ""), t.get("fonte"))
    else:                       # texto livre (retriever com [fonte: ...], erros de ferramenta)
        for bloco in texto.split("\n---\n"):
            m = re.match(r"\[fonte: ([^\]]+)\]\s*(.*)", bloco.strip(), flags=re.S)
            frases += _quebrar(m.group(2), m.group(1)) if m else _quebrar(bloco)
    return [f for f in frases if f.strip()]


def _com_cidade(valor):
    """'GRU' -> 'GRU (São Paulo)': com o nome da cidade, as regras de mesmo lugar funcionam nas linhas do banco."""
    from viagens.banco import AEROPORTOS

    return f"{valor} ({AEROPORTOS[valor][0]})" if isinstance(valor, str) and valor in AEROPORTOS else valor


def _quebrar(texto: str, fonte: Optional[str] = None, max_chars: int = 420) -> List[str]:
    """Divide um trecho em passagens curtas (o modelo NLI aceita ~512 tokens)."""
    linhas = [l.strip("-# ").strip() for l in texto.splitlines() if l.strip("-# ").strip()]
    passagens, atual = [], ""
    for l in linhas:
        if len(atual) + len(l) > max_chars and atual:
            passagens.append(atual)
            atual = ""
        atual = f"{atual} {l}".strip()
    if atual:
        passagens.append(atual)
    return [f"{p} (fonte: {fonte})" if fonte else p for p in passagens]


def evidencias_da_trajetoria(passos) -> List[str]:
    ev = []
    for acao, obs in passos:
        ev += observacao_em_frases(acao.tool, acao.tool_input, obs)
    return ev


# ---------------------------------------------------------------------------
# Afirmações e números
# ---------------------------------------------------------------------------
def afirmacoes(resposta: str) -> List[str]:
    texto = re.sub(r"\[[^\]]{1,80}\]", "", resposta)            # remove citações [Open-Meteo]
    texto = re.sub(r"[*_#`]", "", texto)
    # Listas: subitens curtos ("Valor total: R$ 12.361,37") são verificados JUNTO com o item principal.
    # Soltos, eles perdem o sujeito e o NLI os compara com o item errado (observado na listagem de reservas).
    linhas, blocos = texto.split("\n"), []
    for linha in linhas:
        limpa = re.sub(r"^\s*(\d+[.)]|[-•])\s*", "", linha).strip()
        if not limpa:
            continue
        subitem = bool(re.match(r"^\s+[-•]|^\s*[-•]", linha)) and len(limpa) < 70 and blocos
        if subitem:
            blocos[-1] = blocos[-1].rstrip(".:") + "; " + limpa
        else:
            blocos.append(limpa)
    saida = []
    for bloco in blocos:
        for p in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])", bloco):
            p = p.strip()
            if len(p) >= 25 and not _NAO_FACTUAL.match(p) and not p.endswith(("?", ":")):
                saida.append(p)
    return saida


# separador de milhar nunca começa com 0 ("0.134307" é decimal, não 134 mil)
_NUM = re.compile(r"(?<![\w-])([1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(\s*mil\b)?")


def numeros(texto: str) -> List[float]:
    """Números relevantes (ignora datas, anos e inteiros pequenos como dias/pessoas)."""
    texto = re.sub(r"\d{4}-\d{2}-\d{2}", " ", texto)
    texto = re.sub(r"\d{1,2}/\d{1,2}(/\d{2,4})?", " ", texto)
    valores = []
    for m in _NUM.finditer(texto):
        bruto = m.group(1)
        if re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+(,\d+)?", bruto):
            v = float(bruto.replace(".", "").replace(",", "."))
        else:
            v = float(bruto.replace(",", "."))
        if m.group(2):
            v *= 1000
        if (v.is_integer() and v <= 31) or (1900 <= v <= 2100 and v.is_integer()):
            continue
        valores.append(v)
    return valores


def numero_sustentado(v: float, base: List[float], tol: float = 0.01, tol_derivado: float = 0.005) -> bool:
    perto = lambda a, b, t: abs(a - b) <= t * max(abs(b), 1e-9) or abs(a - b) <= 0.011
    if any(perto(v, b, tol) for b in base):
        return True
    # derivações simples (por dia, por pessoa, total = valor x dias). Tolerância menor, porque
    # com muitos multiplicadores coincidências aparecem (ex.: 180 x 17 = 3060, perto de 3100).
    return any(perto(v, b / k, tol_derivado) or perto(v, b * k, tol_derivado) for b in base for k in range(2, 32))


# ---------------------------------------------------------------------------
# NLI
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _pipeline_hf():
    import os

    import torch
    from transformers import pipeline

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return pipeline("text-classification", model=MODELO_NLI, top_k=None, device="cpu")


# ---------------------------------------------------------------------------
# Lugares: uma evidência só CONTRADIZ uma afirmação se falar do mesmo lugar.
# ("sem visto em Portugal" não contradiz "sem visto em Roma"; é só outra evidência.)
# ---------------------------------------------------------------------------
_PAIS_DA_CIDADE = {"lisboa": "portugal", "porto": "portugal", "paris": "franca", "roma": "italia", "milao": "italia",
                   "barcelona": "espanha", "madri": "espanha", "amsterda": "holanda", "londres": "reino unido",
                   "buenos aires": "argentina", "bariloche": "argentina", "santiago": "chile",
                   "nova york": "estados unidos", "orlando": "estados unidos"}
_SCHENGEN = {"portugal", "espanha", "franca", "italia", "alemanha", "holanda", "suica", "grecia", "austria", "belgica"}
_PAISES = _SCHENGEN | {"reino unido", "inglaterra", "argentina", "chile", "uruguai", "paraguai", "estados unidos", "eua"}


def lugares(texto: str) -> set:
    t = normalizar(texto)
    achados = {c for c in _PAIS_DA_CIDADE if re.search(rf"\b{c}\b", t)}
    achados |= {p for p in _PAISES if re.search(rf"\b{p}\b", t)}
    if "schengen" in t:
        achados.add("schengen")
    expandido = set(achados) | {_PAIS_DA_CIDADE[c] for c in achados if c in _PAIS_DA_CIDADE}
    if expandido & _SCHENGEN:
        expandido.add("schengen")
    return expandido


def mesmo_lugar(afirmacao: str, evidencia: str) -> bool:
    """False só quando as duas citam lugares e eles não se cruzam (ex.: Roma x Portugal).
    'Schengen' sozinho na evidência não basta para cruzar: ele aparece em textos de vários países."""
    la, le = lugares(afirmacao), lugares(evidencia)
    if not la or not le:
        return True
    ca, ce = la & set(_PAIS_DA_CIDADE), le & set(_PAIS_DA_CIDADE)
    if ca and ce and not (ca & ce):   # cidades diferentes (Lisboa x Porto), mesmo que no mesmo país
        return False
    especificos_ev = le - {"schengen"}
    if especificos_ev and not (la & especificos_ev) and (la - {"schengen"}):
        return False
    return bool(la & le)


def _numeros_canonicos(texto: str) -> str:
    """'2.409,97' -> '2409.97' e '15 mil' -> '15000', para premissa e hipótese usarem o mesmo formato."""
    def troca(m):
        bruto = m.group(1)
        if re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+(,\d+)?", bruto):
            v = bruto.replace(".", "").replace(",", ".")
        else:
            v = bruto.replace(",", ".")
        if m.group(2):
            v = f"{float(v) * 1000:g}"
        return v
    return _NUM.sub(troca, texto)


_MESES = {"janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7,
          "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}


def _datas_canonicas(texto: str) -> str:
    """'2027-05-03' e '3 de maio de 2027' -> '03/05/2027' (premissa e hipótese no mesmo formato)."""
    texto = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: f"{m.group(3)}/{m.group(2)}/{m.group(1)}", texto)
    return re.sub(r"\b(\d{1,2})º? de (" + "|".join(_MESES) + r")(?: de (\d{4}))?",
                  lambda m: f"{int(m.group(1)):02d}/{_MESES[m.group(2).lower()]:02d}" + (f"/{m.group(3)}" if m.group(3) else ""),
                  texto, flags=re.I)


def identificadores(texto: str) -> set:
    """Entidades nomeadas que distinguem itens de uma lista: nº de voo (AS1234) e nomes próprios compostos."""
    ids = set(re.findall(r"\bAS\d{3,4}\b", texto))
    ids |= {n for n in re.findall(r"\b([A-ZÁÉÍÓÚÂÊÔ][\wÀ-ú'ô]+(?:\s+[A-ZÁÉÍÓÚÂÊÔ][\wÀ-ú'ô]+)+)", texto)
            if not re.match(r"(Espaço Schengen|São Paulo|Rio de|Reino Unido|Estados Unidos|Open-Meteo|Banco Central)", n)}
    return ids


def mesma_entidade(a: str, b: str) -> bool:
    """False quando as duas citam identificadores e eles não se cruzam (voo AS2259 x linha do AS2257)."""
    ia, ib = identificadores(a), identificadores(b)
    return not (ia and ib and not (ia & ib))


def _tokens(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", normalizar(t)) if w not in STOP and len(w) > 2}


class NLI:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        if backend in ("auto", "hf"):
            try:
                _pipeline_hf()
                self.backend = "hf"
            except Exception as e:  # sem torch/transformers ou sem internet para baixar o modelo
                if backend == "hf":
                    raise
                print(f"[verificacao] modelo NLI indisponível ({type(e).__name__}); usando backend heurístico.")
                self.backend = "heuristico"

    def classificar(self, pares: List[tuple]) -> List[dict]:
        """pares = [(premissa, hipotese)] -> [{'entailment':p, 'neutral':p, 'contradiction':p}]"""
        if not pares:
            return []
        pares = [(_numeros_canonicos(_datas_canonicas(p)), _numeros_canonicos(_datas_canonicas(h))) for p, h in pares]
        if self.backend == "hf":
            saida = _pipeline_hf()([{"text": p, "text_pair": h} for p, h in pares], batch_size=8, truncation=True)
            return [{o["label"]: o["score"] for o in (s if isinstance(s, list) else [s])} for s in saida]
        res = []
        negacao = lambda t: bool(re.search(r"\b(nao|nunca|proibid\w*|vedad\w*)\b", normalizar(t)))
        for p, h in pares:   # heurística: cobertura lexical (por radical) da hipótese pela premissa
            th, tp = {w[:5] for w in _tokens(h)}, {w[:5] for w in _tokens(p)}
            cob = len(th & tp) / max(len(th), 1)
            neg = negacao(h) != negacao(p)
            ent = cob if cob >= 0.5 and not neg else cob * 0.3
            con = 0.7 if cob >= 0.5 and neg else 0.0
            res.append({"entailment": ent, "contradiction": con, "neutral": max(0.0, 1 - ent - con)})
        return res


# ---------------------------------------------------------------------------
# Verificador
# ---------------------------------------------------------------------------
@dataclass
class Afirmacao:
    texto: str
    rotulo: str                 # SUSTENTADA | CONTRADITA | NAO_VERIFICAVEL | NUMERO_SEM_EVIDENCIA
    p_sustenta: float
    p_contradiz: float
    evidencia: str = ""
    numeros_sem_evidencia: List[float] = field(default_factory=list)
    criterio: str = "nli"       # nli | numerico


@dataclass
class Verificacao:
    afirmacoes: List[Afirmacao]
    contradicoes_internas: List[tuple]
    n_evidencias: int
    backend: str

    @property
    def fundamentacao(self) -> float:
        """Fração de afirmações sustentadas pelas evidências."""
        return sum(a.rotulo == "SUSTENTADA" for a in self.afirmacoes) / len(self.afirmacoes) if self.afirmacoes else 1.0

    @property
    def problemas(self) -> List[Afirmacao]:
        return [a for a in self.afirmacoes if a.rotulo in ("CONTRADITA", "NUMERO_SEM_EVIDENCIA")]

    @property
    def aprovada(self) -> bool:
        return not self.problemas and not self.contradicoes_internas

    def resumo(self) -> dict:
        cont = {}
        for a in self.afirmacoes:
            cont[a.rotulo] = cont.get(a.rotulo, 0) + 1
        return {"aprovada": self.aprovada, "fundamentacao": round(self.fundamentacao, 2), "rotulos": cont,
                "contradicoes_internas": len(self.contradicoes_internas), "evidencias": self.n_evidencias,
                "backend_nli": self.backend}

    def to_dict(self) -> dict:
        return {**self.resumo(), "afirmacoes": [asdict(a) for a in self.afirmacoes],
                "pares_contraditorios": self.contradicoes_internas}


class Verificador:
    def __init__(self, backend: str = "auto", top_passagens: int = 5):
        self.nli = NLI(backend)
        self.top = top_passagens

    def _candidatas(self, afirmacao: str, evidencias: List[str]) -> List[str]:
        ta = _tokens(afirmacao)
        return sorted(evidencias, key=lambda e: len(ta & _tokens(e)), reverse=True)[: self.top]

    def verificar(self, resposta: str, passos) -> Verificacao:
        evid = evidencias_da_trajetoria(passos)
        nums_ev = [n for e in evid for n in numeros(e)]
        lista = []
        for af in afirmacoes(resposta):
            cands = self._candidatas(af, evid)
            probs = self.nli.classificar([(c, af) for c in cands])
            i_ent = max(range(len(probs)), key=lambda i: probs[i]["entailment"], default=None)
            p_ent = probs[i_ent]["entailment"] if probs else 0.0
            # contradição só vale de evidência sobre o MESMO lugar da afirmação
            # e só de evidência que trate do assunto (>= 2 termos em comum): um trecho sem relação
            # "contradizia" "Eles devem ir na bagagem de mão" (observado na avaliação)
            validos = [i for i in range(len(probs)) if mesmo_lugar(af, cands[i]) and mesma_entidade(af, cands[i])
                       and len(_tokens(af) & _tokens(cands[i])) >= 2]
            i_con = max(validos, key=lambda i: probs[i]["contradiction"], default=None)
            p_con = probs[i_con]["contradiction"] if i_con is not None else 0.0
            nums_af = numeros(af)
            sem_ev = [n for n in nums_af if not numero_sustentado(n, nums_ev)]
            criterio = "nli"
            if p_con >= LIMIAR and p_con > p_ent:
                rotulo, ev = "CONTRADITA", cands[i_con]
            elif sem_ev:
                rotulo, ev = "NUMERO_SEM_EVIDENCIA", cands[i_ent] if probs else ""
            elif p_ent >= LIMIAR:
                rotulo, ev = "SUSTENTADA", cands[i_ent]
            elif any(n >= 100 or not n.is_integer() for n in nums_af):
                # todos os valores (monetários/decimais) conferem com as evidências e o NLI não contradiz
                rotulo, ev, criterio = "SUSTENTADA", cands[i_ent] if probs else "", "numerico"
            else:
                rotulo, ev = "NAO_VERIFICAVEL", cands[i_ent] if probs else ""
            lista.append(Afirmacao(af, rotulo, round(p_ent, 3), round(p_con, 3), ev[:300], sem_ev, criterio))

        # coerência lógica: contradições entre afirmações da própria resposta
        # (só entre frases do mesmo assunto: o NLI dá falsos positivos para frases sem relação)
        textos = [a.texto for a in lista]
        pares = [(textos[i], textos[j]) for i in range(len(textos)) for j in range(len(textos))
                 if i != j and len(_tokens(textos[i]) & _tokens(textos[j])) >= 2
                 and mesma_entidade(textos[i], textos[j]) and mesmo_lugar(textos[i], textos[j])][:24]
        internas = [(p, h) for (p, h), pr in zip(pares, self.nli.classificar(pares)) if pr["contradiction"] >= 0.9]
        # duas afirmações SUSTENTADAS pelas mesmas evidências não se contradizem de fato: o NLI marcava
        # "total de R$ 19.506,50" x "R$ 9.753,25 por pessoa" como contraditórias só porque os números diferem
        sustentadas = {a.texto for a in lista if a.rotulo == "SUSTENTADA"}
        internas = [(p, h) for p, h in internas if not (p in sustentadas and h in sustentadas)]
        return Verificacao(lista, internas, len(evid), self.nli.backend)
