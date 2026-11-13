"""Função SERVERLESS: estimativa de custo total da viagem.

Segue o contrato de uma função FaaS (AWS Lambda / Google Cloud Functions):
`handler(event, context) -> {"statusCode": int, "body": str(JSON)}`, sem
estado entre chamadas. Em produção seria implantada na nuvem e chamada via
HTTP ou SDK. Aqui, `invocar()` faz o papel do gateway: serializa o evento,
mede a duração, simula o cold start da primeira chamada e valida o contrato.

Regra de negócio (fixa): passagem + hospedagem + diárias de gastos locais,
mais o IOF de 3,5% sobre os gastos no exterior e uma margem cambial de 10%.
"""
from __future__ import annotations

import json
import time

IOF = 0.035
MARGEM_CAMBIAL = 0.10
GASTO_DIARIO_PADRAO_BRL = {"economico": 300, "moderado": 550, "confortavel": 1000}


def handler(event: dict, context=None) -> dict:
    try:
        dias = int(event["dias"])
        pessoas = int(event.get("pessoas", 1))
        passagem = float(event.get("passagem_por_pessoa_brl", 0))
        diaria_hotel = float(event.get("diaria_hotel_brl", 0))
        perfil = event.get("perfil_gastos", "moderado")
        gasto_diario = float(event.get("gasto_diario_por_pessoa_brl", GASTO_DIARIO_PADRAO_BRL[perfil]))
    except (KeyError, ValueError) as e:
        return {"statusCode": 400, "body": json.dumps({"erro": f"evento inválido: {e}"}, ensure_ascii=False)}
    if dias <= 0 or pessoas <= 0:
        return {"statusCode": 422, "body": json.dumps({"erro": "dias e pessoas devem ser positivos"})}

    passagens = passagem * pessoas
    hospedagem = diaria_hotel * dias
    gastos_locais = gasto_diario * pessoas * dias
    exterior = hospedagem + gastos_locais            # pago em moeda estrangeira
    iof = exterior * IOF
    margem = exterior * MARGEM_CAMBIAL
    total = passagens + exterior + iof + margem
    corpo = {"itens_brl": {"passagens": round(passagens, 2), "hospedagem": round(hospedagem, 2),
                           "gastos_locais": round(gastos_locais, 2), "iof_3_5%": round(iof, 2),
                           "margem_cambial_10%": round(margem, 2)},
             "total_brl": round(total, 2), "por_pessoa_brl": round(total / pessoas, 2),
             "premissas": f"IOF {IOF:.1%} e margem cambial {MARGEM_CAMBIAL:.0%} sobre gastos no exterior; "
                          f"gasto diário de R$ {gasto_diario:.0f} por pessoa ({perfil})."}
    return {"statusCode": 200, "body": json.dumps(corpo, ensure_ascii=False)}


_aquecida = False


def invocar(event: dict) -> dict:
    """Simula o gateway FaaS: cold start na 1ª chamada, evento serializado, contrato validado."""
    global _aquecida
    inicio = time.perf_counter()
    cold = not _aquecida
    if cold:
        time.sleep(0.05)  # inicialização do runtime (simulada)
        _aquecida = True
    resposta = handler(json.loads(json.dumps(event)))  # o evento atravessa a rede como JSON
    assert {"statusCode", "body"} <= resposta.keys(), "contrato FaaS violado"
    return {"statusCode": resposta["statusCode"], "body": json.loads(resposta["body"]),
            "cold_start": cold, "duracao_ms": round((time.perf_counter() - inicio) * 1000, 1)}
