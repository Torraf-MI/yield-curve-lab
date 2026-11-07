"""Configurazione dell'analisi: portafoglio del cliente, strumenti di copertura e obiettivo da immunizzare.

Formato JSON (tutti i campi tranne ``portafoglio`` sono facoltativi)::

    {
      "cliente": "Famiglia Rossi",
      "valutazione": "2026-10-30",
      "titoli": "titoli.csv",
      "storia": "storia_curve.csv",
      "portafoglio": [{"codice": "BTP-2033-03", "nominale": 90000}, ...],
      "coperture": ["BUND-2028-02", "BUND-2031-08", ...],
      "obiettivo": {"descrizione": "Università dei figli", "importo": 250000, "data": "2034-09-01"},
      "livello_var": 0.99
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .copertura import Obiettivo

COPERTURE_STANDARD = ("BUND-2028-02", "BUND-2031-08", "BUND-2034-02", "BUND-2036-02", "BUND-2044-07",
                      "BUND-2053-08")


@dataclass
class Configurazione:
    portafoglio: dict[str, float]  # codice → nominale in euro
    cliente: str = "Cliente"
    valutazione: date = date(2026, 10, 30)
    titoli: str | None = None  # CSV dei titoli e dei prezzi; se assente si usano i dati dimostrativi
    storia: str | None = None  # CSV della storia mensile delle curve; se assente si usa quella simulata
    coperture: tuple[str, ...] = COPERTURE_STANDARD
    obiettivo: Obiettivo | None = None
    livello_var: float = 0.99


def da_dizionario(d: dict, base: Path | None = None) -> Configurazione:
    if not d.get("portafoglio"):
        raise ValueError("La configurazione deve contenere almeno una posizione in 'portafoglio'")
    port: dict[str, float] = {}
    for p in d["portafoglio"]:
        if float(p["nominale"]) <= 0:
            raise ValueError(f"Nominale non positivo per {p['codice']}")
        port[p["codice"]] = port.get(p["codice"], 0.0) + float(p["nominale"])
    ob = None
    if d.get("obiettivo"):
        o = d["obiettivo"]
        ob = Obiettivo(o.get("descrizione", "Obiettivo"), float(o["importo"]), date.fromisoformat(o["data"]))

    def percorso(chiave):
        p = d.get(chiave)
        if p and base is not None and not Path(p).is_absolute():
            p = str((base / p).resolve())
        return p

    titoli, storia = percorso("titoli"), percorso("storia")
    livello = float(d.get("livello_var", 0.99))
    if not 0.5 < livello < 1:
        raise ValueError("livello_var deve essere tra 0,5 e 1 (per esempio 0.99)")
    return Configurazione(
        portafoglio=port, cliente=d.get("cliente", "Cliente"),
        valutazione=date.fromisoformat(d["valutazione"]) if d.get("valutazione") else date(2026, 10, 30),
        titoli=titoli, storia=storia, coperture=tuple(d.get("coperture", COPERTURE_STANDARD)), obiettivo=ob,
        livello_var=livello,
    )


def carica(percorso: str | Path) -> Configurazione:
    percorso = Path(percorso)
    return da_dizionario(json.loads(percorso.read_text(encoding="utf-8")), percorso.parent)


def esempio() -> Configurazione:
    """Portafoglio dimostrativo: 500.000 € di nominale, quasi tutto in BTP lungo la curva."""
    return Configurazione(
        cliente="Famiglia Rossi",
        portafoglio={"BTP-2028-03": 80000, "BTP-2030-10": 70000, "BTP-2033-03": 90000, "BTP-2035-07": 80000,
                     "BTP-2039-08": 60000, "BTP-2044-09": 50000, "BTP-2053-09": 40000, "BUND-2032-02": 30000},
        obiettivo=Obiettivo("Università dei figli", 250000, date(2034, 9, 1)),
    )
