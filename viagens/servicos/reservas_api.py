"""Microsserviço INTERNO de reservas (API HTTP/JSON).

Representa um sistema interno da agência, separado do agente: o agente não
acessa a tabela de reservas para escrever, ele chama esta API, que aplica as
regras de negócio (validação, status inicial 'pendente', cálculo do valor).

Endpoints:
  GET  /saude                         -> {"status": "ok"}
  GET  /reservas?usuario_id=...       -> lista de reservas do usuário
  POST /reservas  {usuario_id, voo_id, pessoas}  -> cria reserva de voo 'pendente'

Roda em uma thread local (http.server, sem dependências extras), então funciona
no VS Code e no Colab. Em produção seria um serviço próprio (FastAPI, container).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from viagens.banco import DB_PATH, garantir_banco

MAX_PESSOAS = 9


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silencia o log padrão do http.server
        pass

    def _responder(self, status: int, corpo: dict):
        dados = json.dumps(corpo, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/saude":
            return self._responder(200, {"status": "ok"})
        if url.path == "/reservas":
            usuario = parse_qs(url.query).get("usuario_id", [None])[0]
            if not usuario:
                return self._responder(400, {"erro": "parâmetro usuario_id é obrigatório"})
            con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            linhas = con.execute("""
                SELECT r.id, r.tipo, r.status, r.data_inicio, r.data_fim, r.pessoas, ROUND(r.valor_total_brl, 2) AS valor_total_brl,
                       v.numero AS voo, v.origem, v.destino, h.nome AS hotel, h.cidade AS cidade_hotel
                FROM reservas r LEFT JOIN voos v ON v.id = r.voo_id LEFT JOIN hoteis h ON h.id = r.hotel_id
                WHERE r.usuario_id = ? ORDER BY r.data_inicio""", (usuario,)).fetchall()
            con.close()
            return self._responder(200, {"usuario_id": usuario, "reservas": [dict(l) for l in linhas]})
        return self._responder(404, {"erro": f"rota {url.path} não existe"})

    def do_POST(self):
        if urlparse(self.path).path != "/reservas":
            return self._responder(404, {"erro": "rota não existe"})
        try:
            corpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            usuario, voo_id, pessoas = corpo["usuario_id"], int(corpo["voo_id"]), int(corpo.get("pessoas", 1))
        except (KeyError, ValueError, json.JSONDecodeError):
            return self._responder(400, {"erro": "corpo inválido: informe usuario_id, voo_id e pessoas"})
        if not 1 <= pessoas <= MAX_PESSOAS:
            return self._responder(422, {"erro": f"pessoas deve estar entre 1 e {MAX_PESSOAS}"})
        con = sqlite3.connect(DB_PATH)
        try:
            voo = con.execute("SELECT data_partida, preco_brl, assentos_disponiveis, numero FROM voos WHERE id = ?",
                              (voo_id,)).fetchone()
            if not voo:
                return self._responder(404, {"erro": f"voo {voo_id} não existe"})
            if voo[2] < pessoas:
                return self._responder(409, {"erro": f"voo {voo[3]} tem só {voo[2]} assento(s) disponível(is)"})
            cur = con.execute("""INSERT INTO reservas (usuario_id, tipo, voo_id, data_inicio, pessoas, valor_total_brl,
                                 status, criada_em) VALUES (?, 'voo', ?, ?, ?, ?, 'pendente', ?)""",
                              (usuario, voo_id, voo[0], pessoas, round(voo[1] * pessoas, 2),
                               datetime.now().isoformat(timespec="seconds")))
            con.commit()
            return self._responder(201, {"reserva_id": cur.lastrowid, "status": "pendente", "voo": voo[3],
                                         "data": voo[0], "pessoas": pessoas,
                                         "valor_total_brl": round(voo[1] * pessoas, 2),
                                         "observacao": "Reserva pendente: o pagamento é confirmado por um atendente."})
        finally:
            con.close()


_servidor = None


def iniciar_servico(porta: int = 0) -> str:
    """Sobe o microsserviço em segundo plano e devolve a URL base. Idempotente."""
    global _servidor
    if _servidor is None:
        garantir_banco()
        _servidor = ThreadingHTTPServer(("127.0.0.1", porta), _Handler)
        threading.Thread(target=_servidor.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{_servidor.server_address[1]}"


def parar_servico() -> None:
    global _servidor
    if _servidor is not None:
        _servidor.shutdown()
        _servidor = None
