"""Banco SQLite do projeto: voos, hotéis e reservas (dados FICTÍCIOS e determinísticos).

A companhia AeroSul e os hotéis são inventados; preços e horários são gerados
com semente fixa, então o banco é idêntico em qualquer máquina (Colab incluso).
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from viagens.config import ROOT

DB_PATH = ROOT / "data" / "viagens.db"

AEROPORTOS = {
    "GRU": ("São Paulo", "Brasil"), "GIG": ("Rio de Janeiro", "Brasil"),
    "LIS": ("Lisboa", "Portugal"), "OPO": ("Porto", "Portugal"), "CDG": ("Paris", "França"),
    "FCO": ("Roma", "Itália"), "BCN": ("Barcelona", "Espanha"), "MAD": ("Madri", "Espanha"),
    "LHR": ("Londres", "Reino Unido"), "EZE": ("Buenos Aires", "Argentina"),
}
# preço base (BRL, só ida, econômica) e duração aproximada a partir do Brasil
ROTAS = {"LIS": (3200, 600), "OPO": (3400, 620), "CDG": (3900, 690), "FCO": (4100, 720),
         "BCN": (3800, 700), "MAD": (3600, 650), "LHR": (4300, 700), "EZE": (1200, 170)}

HOTEIS = {
    "Lisboa": [("Hotel Alfama Azul", "Alfama", 3), ("Lisboa Tejo Palace", "Baixa", 5), ("Casa Bairro Alto", "Bairro Alto", 4),
               ("Belém Garden Inn", "Belém", 3)],
    "Porto": [("Ribeira Douro Hotel", "Ribeira", 4), ("Porto Azulejo", "Baixa", 3)],
    "Paris": [("Hôtel Petit Marais", "Marais", 3), ("Le Grand Opéra", "Opéra", 5), ("Montmartre Lumière", "Montmartre", 4),
              ("Saint-Germain Rive", "Saint-Germain", 4)],
    "Roma": [("Albergo Trastevere", "Trastevere", 3), ("Roma Colosseo Suites", "Monti", 4), ("Palazzo Navona", "Centro Storico", 5)],
    "Barcelona": [("Hotel Gòtic Blau", "Bairro Gótico", 3), ("Barceloneta Mar", "Barceloneta", 4), ("Eixample Modern", "Eixample", 4)],
    "Madri": [("Hostal Sol Madrid", "Sol", 2), ("Gran Vía Plaza", "Gran Vía", 4)],
    "Londres": [("Camden Lodge", "Camden", 3), ("Westminster Royal", "Westminster", 5)],
    "Buenos Aires": [("Palermo Soho Hotel", "Palermo", 4), ("San Telmo Tango", "San Telmo", 3), ("Recoleta Grand", "Recoleta", 5)],
}


def criar_banco(caminho: Path = DB_PATH, seed: int = 42) -> Path:
    rnd = random.Random(seed)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        caminho.unlink()
    con = sqlite3.connect(caminho)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE aeroportos (
        iata TEXT PRIMARY KEY,          -- código IATA, ex.: 'GRU'
        cidade TEXT NOT NULL,
        pais TEXT NOT NULL
    );
    CREATE TABLE voos (
        id INTEGER PRIMARY KEY,
        numero TEXT NOT NULL,           -- ex.: 'AS1234'
        origem TEXT NOT NULL REFERENCES aeroportos(iata),
        destino TEXT NOT NULL REFERENCES aeroportos(iata),
        data_partida TEXT NOT NULL,     -- 'YYYY-MM-DD'
        hora_partida TEXT NOT NULL,     -- 'HH:MM'
        duracao_min INTEGER NOT NULL,
        escalas INTEGER NOT NULL,       -- 0 = voo direto
        classe TEXT NOT NULL,           -- 'economica' | 'executiva'
        preco_brl REAL NOT NULL,        -- preço por pessoa, só ida
        assentos_disponiveis INTEGER NOT NULL
    );
    CREATE TABLE hoteis (
        id INTEGER PRIMARY KEY,
        nome TEXT NOT NULL,
        cidade TEXT NOT NULL,
        bairro TEXT,
        estrelas INTEGER NOT NULL,
        diaria_brl REAL NOT NULL,       -- diária do quarto duplo
        nota REAL NOT NULL,             -- avaliação de 0 a 10
        cafe_incluso INTEGER NOT NULL,  -- 1 = sim
        opcao_vegana INTEGER NOT NULL   -- 1 = restaurante com opção vegana
    );
    CREATE TABLE reservas (
        id INTEGER PRIMARY KEY,
        usuario_id TEXT NOT NULL,
        tipo TEXT NOT NULL,             -- 'voo' | 'hotel'
        voo_id INTEGER REFERENCES voos(id),
        hotel_id INTEGER REFERENCES hoteis(id),
        data_inicio TEXT NOT NULL,
        data_fim TEXT,
        pessoas INTEGER NOT NULL,
        valor_total_brl REAL NOT NULL,
        status TEXT NOT NULL,           -- 'confirmada' | 'pendente' | 'cancelada'
        criada_em TEXT NOT NULL
    );
    """)
    cur.executemany("INSERT INTO aeroportos VALUES (?,?,?)", [(k, c, p) for k, (c, p) in AEROPORTOS.items()])

    inicio = date(2026, 12, 1)
    voos = []
    vid = 0
    for destino, (base, dur) in ROTAS.items():
        for origem in ("GRU", "GIG"):
            for dia in range(0, 210, 3):   # dez/2026 a jun/2027
                d = inicio + timedelta(days=dia)
                alta = d.month in (12, 1, 7)
                for _ in range(rnd.choice([1, 1, 2])):
                    for ida in (True, False):
                        vid += 1
                        escalas = rnd.choice([0, 0, 1, 1, 2]) if destino != "EZE" else rnd.choice([0, 0, 0, 1])
                        classe = "executiva" if rnd.random() < 0.15 else "economica"
                        preco = base * (1.35 if alta else 1.0) * (0.85 if escalas else 1.1) * (3.2 if classe == "executiva" else 1)
                        preco *= rnd.uniform(0.85, 1.25)
                        o, de = (origem, destino) if ida else (destino, origem)
                        voos.append((vid, f"AS{1000 + vid}", o, de, d.isoformat(),
                                     f"{rnd.choice([7, 9, 11, 14, 17, 19, 21, 22, 23]):02d}:{rnd.choice(['00', '15', '30', '45'])}",
                                     dur + escalas * rnd.randint(90, 240), escalas, classe, round(preco, 2),
                                     rnd.randint(0, 60)))
    cur.executemany("INSERT INTO voos VALUES (?,?,?,?,?,?,?,?,?,?,?)", voos)

    hoteis = []
    hid = 0
    for cidade, lista in HOTEIS.items():
        for nome, bairro, estrelas in lista:
            hid += 1
            diaria = {2: 280, 3: 450, 4: 750, 5: 1500}[estrelas] * rnd.uniform(0.8, 1.3)
            if cidade == "Buenos Aires":
                diaria *= 0.6
            hoteis.append((hid, nome, cidade, bairro, estrelas, round(diaria, 2), round(rnd.uniform(7.2, 9.6), 1),
                           int(rnd.random() < 0.6), int(rnd.random() < 0.5)))
    cur.executemany("INSERT INTO hoteis VALUES (?,?,?,?,?,?,?,?,?)", hoteis)

    reservas = [
        (1, "viajante_demo", "voo", 15, None, "2026-12-07", None, 2, 0, "confirmada", "2026-10-02"),
        (2, "viajante_demo", "hotel", None, 2, "2026-12-07", "2026-12-14", 2, 0, "confirmada", "2026-10-02"),
        (3, "viajante_demo", "voo", 640, None, "2027-03-04", None, 1, 0, "pendente", "2026-11-05"),
        (4, "outro_usuario", "hotel", None, 8, "2027-01-10", "2027-01-15", 2, 0, "cancelada", "2026-09-20"),
    ]
    cur.executemany("INSERT INTO reservas VALUES (?,?,?,?,?,?,?,?,?,?,?)", reservas)
    # valores coerentes com os preços gerados
    cur.execute("""UPDATE reservas SET valor_total_brl = (SELECT preco_brl FROM voos WHERE voos.id = reservas.voo_id) * pessoas,
                   data_inicio = (SELECT data_partida FROM voos WHERE voos.id = reservas.voo_id) WHERE tipo = 'voo'""")
    # o hotel da reserva 2 cobre a mesma viagem do voo da reserva 1 (7 noites a partir do voo)
    cur.execute("""UPDATE reservas SET data_inicio = (SELECT data_inicio FROM reservas WHERE id = 1),
                   data_fim = date((SELECT data_inicio FROM reservas WHERE id = 1), '+7 days') WHERE id = 2""")
    cur.execute("""UPDATE reservas SET valor_total_brl = (SELECT diaria_brl FROM hoteis WHERE hoteis.id = reservas.hotel_id)
                   * (julianday(data_fim) - julianday(data_inicio)) WHERE tipo = 'hotel'""")
    con.commit()
    con.close()
    return caminho


def garantir_banco() -> Path:
    if not DB_PATH.exists():
        criar_banco()
    return DB_PATH


def conectar_somente_leitura() -> sqlite3.Connection:
    """Conexão read-only (modo URI 'ro'): mesmo um SQL malicioso não consegue escrever."""
    garantir_banco()
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


if __name__ == "__main__":
    criar_banco()
    con = conectar_somente_leitura()
    for t in ("aeroportos", "voos", "hoteis", "reservas"):
        print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
