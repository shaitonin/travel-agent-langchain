"""Clientes das APIs externas reais (sem chave) com equivalentes simulados.

- Open-Meteo Geocoding  -> cidade para coordenadas
- Open-Meteo Forecast   -> previsão diária (até 16 dias à frente)
- Open-Meteo Archive    -> referência histórica (mesmas datas do ano anterior)
                           quando a viagem está além do horizonte de previsão
- Frankfurter           -> câmbio oficial do Banco Central Europeu

Com USE_DUMMY=true, as funções devolvem dados determinísticos marcados com
"simulado": True, mantendo o mesmo formato de saída.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Optional

import requests

from viagens.config import USE_DUMMY

TIMEOUT = 15
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FX_URL = "https://api.frankfurter.app/latest"

MOEDA_POR_CIDADE = {
    "Lisboa": "EUR", "Porto": "EUR", "Paris": "EUR", "Roma": "EUR", "Madri": "EUR",
    "Barcelona": "EUR", "Milão": "EUR", "Londres": "GBP", "Buenos Aires": "ARS",
    "Santiago": "CLP", "Nova York": "USD", "Tóquio": "JPY",
    "São Paulo": "BRL", "Rio de Janeiro": "BRL", "Salvador": "BRL", "Recife": "BRL",
    "Florianópolis": "BRL", "Porto Alegre": "BRL", "Belo Horizonte": "BRL",
}

CODIGOS_TEMPO = {
    0: "céu limpo", 1: "predominantemente limpo", 2: "parcialmente nublado", 3: "nublado",
    45: "neblina", 48: "neblina com geada", 51: "garoa fraca", 53: "garoa", 55: "garoa forte",
    61: "chuva fraca", 63: "chuva", 65: "chuva forte", 71: "neve fraca", 73: "neve", 75: "neve forte",
    80: "pancadas de chuva fracas", 81: "pancadas de chuva", 82: "pancadas de chuva fortes",
    95: "trovoadas", 96: "trovoadas com granizo", 99: "trovoadas com granizo forte",
}


class APIError(RuntimeError):
    """Falha de uma API externa, com mensagem legível para o agente."""


# ---------------------------------------------------------------------------
# Dados simulados (modo dummy)
# ---------------------------------------------------------------------------
_COORDS_DUMMY = {
    "lisboa": ("Lisboa", "Portugal", 38.72, -9.14), "porto": ("Porto", "Portugal", 41.15, -8.61),
    "paris": ("Paris", "França", 48.85, 2.35), "roma": ("Roma", "Itália", 41.89, 12.48),
    "madri": ("Madri", "Espanha", 40.42, -3.70), "barcelona": ("Barcelona", "Espanha", 41.39, 2.17),
    "londres": ("Londres", "Reino Unido", 51.51, -0.13),
    "buenos aires": ("Buenos Aires", "Argentina", -34.61, -58.38),
    "santiago": ("Santiago", "Chile", -33.45, -70.67), "nova york": ("Nova York", "Estados Unidos", 40.71, -74.01),
    "sao paulo": ("São Paulo", "Brasil", -23.55, -46.63), "rio de janeiro": ("Rio de Janeiro", "Brasil", -22.91, -43.17),
}
# unidades de moeda por 1 EUR (valores de referência fixos para o modo dummy)
_TAXAS_DUMMY = {"EUR": 1.0, "BRL": 6.20, "USD": 1.08, "GBP": 0.85, "JPY": 165.0, "CHF": 0.95}


def _semente(*partes) -> int:
    return int(hashlib.md5("|".join(map(str, partes)).encode()).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Geocodificação
# ---------------------------------------------------------------------------
def geocodificar(cidade: str) -> dict:
    if USE_DUMMY:
        from viagens.dummy import normalizar

        chave = normalizar(cidade).strip()
        if chave not in _COORDS_DUMMY:
            raise APIError(f"Cidade '{cidade}' não encontrada (base simulada).")
        nome, pais, lat, lon = _COORDS_DUMMY[chave]
        return {"nome": nome, "pais": pais, "lat": lat, "lon": lon, "simulado": True}

    try:
        r = requests.get(GEO_URL, params={"name": cidade, "count": 1, "language": "pt", "format": "json"},
                         timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise APIError(f"Falha ao consultar o geocodificador Open-Meteo: {e}") from e
    res = r.json().get("results") or []
    if not res:
        raise APIError(f"Cidade '{cidade}' não encontrada no geocodificador Open-Meteo.")
    g = res[0]
    return {"nome": g["name"], "pais": g.get("country"), "lat": g["latitude"], "lon": g["longitude"]}


# ---------------------------------------------------------------------------
# Previsão do tempo
# ---------------------------------------------------------------------------
def previsao_tempo(cidade: str, data_inicio: Optional[str] = None, dias: int = 3) -> dict:
    """Previsão diária para `dias` dias a partir de `data_inicio` (YYYY-MM-DD, padrão hoje).

    Datas além de 15 dias usam a referência histórica do ano anterior.
    """
    dias = max(1, min(int(dias), 7))
    hoje = date.today()
    inicio = date.fromisoformat(data_inicio) if data_inicio else hoje
    if inicio < hoje:
        # não troca a data em silêncio: o agente precisa saber que pediu um período errado
        raise APIError(f"A data {inicio.isoformat()} já passou (hoje é {hoje.isoformat()}). "
                       "Informe a data FUTURA da viagem, por exemplo o mesmo dia no próximo ano.")
    fim = inicio + timedelta(days=dias - 1)
    geo = geocodificar(cidade)

    if (fim - hoje).days <= 15:
        tipo, ini_q, fim_q = "previsao", inicio, fim
    else:
        tipo = "referencia_historica"
        # recua anos inteiros até o período estar no passado (o arquivo tem ~5 dias de atraso)
        anos = 1
        while fim - timedelta(days=round(365.25 * anos)) > hoje - timedelta(days=7):
            anos += 1
        ini_q = inicio - timedelta(days=round(365.25 * anos))
        fim_q = ini_q + timedelta(days=dias - 1)

    saida = {"cidade": geo["nome"], "pais": geo["pais"], "tipo": tipo, "fonte": "Open-Meteo",
             "periodo_consultado": [ini_q.isoformat(), fim_q.isoformat()], "dias": []}
    if tipo == "referencia_historica":
        saida["observacao"] = ("A data está além do horizonte de previsão (16 dias); "
                               "valores do mesmo período em ano anterior, como referência de clima típico.")

    if USE_DUMMY:
        base = 12 + 14 * (1 - abs(geo["lat"]) / 60)  # mais quente perto do equador
        mes = inicio.month
        sazonal = 6 * (1 if (mes in (6, 7, 8)) == (geo["lat"] > 0) else -1)
        for i in range(dias):
            d = ini_q + timedelta(days=i)
            s = _semente(geo["nome"], d)
            tmax = round(base + sazonal + (s % 50) / 10, 1)
            chuva = round((s % 70) / 10 if s % 3 == 0 else 0.0, 1)
            saida["dias"].append({"data": d.isoformat(), "temp_max": tmax, "temp_min": round(tmax - 7 - s % 4, 1),
                                  "chuva_mm": chuva, "condicao": "chuva fraca" if chuva else "parcialmente nublado"})
        saida["simulado"] = True
        return saida

    params = {"latitude": geo["lat"], "longitude": geo["lon"], "timezone": "auto",
              "start_date": ini_q.isoformat(), "end_date": fim_q.isoformat(),
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"}
    url = FORECAST_URL if tipo == "previsao" else ARCHIVE_URL
    if tipo == "previsao":
        params["daily"] += ",precipitation_probability_max"
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise APIError(f"Falha ao consultar Open-Meteo: {e}") from e
    d = r.json()["daily"]
    for i, dia in enumerate(d["time"]):
        item = {"data": dia, "temp_max": d["temperature_2m_max"][i], "temp_min": d["temperature_2m_min"][i],
                "chuva_mm": d["precipitation_sum"][i],
                "condicao": CODIGOS_TEMPO.get(d["weather_code"][i], f"código {d['weather_code'][i]}")}
        if "precipitation_probability_max" in d:
            item["prob_chuva_%"] = d["precipitation_probability_max"][i]
        saida["dias"].append(item)
    return saida


# ---------------------------------------------------------------------------
# Câmbio
# ---------------------------------------------------------------------------
def converter_moeda(valor: float, de: str, para: str) -> dict:
    de, para = de.upper().strip(), para.upper().strip()
    if valor <= 0:
        # observado na avaliação: o agente chamou o câmbio EM PARALELO com a busca do preço e mandou 0
        raise APIError("O valor a converter precisa ser positivo. Obtenha o valor (ex.: o preço do voo) antes "
                       "de chamar a conversão.")
    if de == para:
        return {"valor": valor, "de": de, "para": para, "convertido": valor, "taxa": 1.0, "fonte": "identidade"}

    if USE_DUMMY:
        if de not in _TAXAS_DUMMY or para not in _TAXAS_DUMMY:
            raise APIError(f"Moeda não suportada: {de if de not in _TAXAS_DUMMY else para} "
                           "(a Frankfurter só cobre moedas cotadas pelo BCE).")
        taxa = _TAXAS_DUMMY[para] / _TAXAS_DUMMY[de]
        return {"valor": valor, "de": de, "para": para, "taxa": round(taxa, 6),
                "convertido": round(valor * taxa, 2), "data_cotacao": date.today().isoformat(),
                "fonte": "Frankfurter (BCE)", "simulado": True}

    try:
        r = requests.get(FX_URL, params={"amount": valor, "from": de, "to": para}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise APIError(f"Falha ao consultar Frankfurter: {e}") from e
    if r.status_code == 404 or (r.ok and para not in r.json().get("rates", {})):
        raise APIError(f"Moeda não suportada pela Frankfurter: {de}->{para} "
                       "(a API só cobre moedas cotadas pelo Banco Central Europeu; ARS e CLP, por exemplo, não).")
    if not r.ok:
        raise APIError(f"A Frankfurter recusou a conversão ({r.status_code}): {r.text[:200]}")
    j = r.json()
    convertido = j["rates"][para]
    return {"valor": valor, "de": de, "para": para, "convertido": round(convertido, 2),
            "taxa": round(convertido / valor, 6) if valor else None, "data_cotacao": j["date"],
            "fonte": "Frankfurter (BCE)"}
