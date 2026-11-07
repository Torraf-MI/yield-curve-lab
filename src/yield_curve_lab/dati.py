"""Dati dimostrativi: titoli BTP e Bund con i loro prezzi, storia mensile delle curve dal 2008.

Tutto è **simulato**: le curve «vere» di oggi sono curve di Svensson con livelli realistici, i prezzi sono quelli
che quelle curve implicano più piccoli scarti specifici (alcuni titoli sono volutamente ricchi o a sconto), la
storia segue per date e ordine di grandezza gli episodi reali (crisi del 2008, debito sovrano 2011, tassi negativi,
spread del 2018, pandemia, rialzo del 2022). Codici e prezzi non sono quelli di titoli reali.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .curve import CurvaSpostata, Svensson, parallelo
from .obbligazioni import Obbligazione

VALUTAZIONE = date(2026, 10, 30)
NODI_STORIA = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)

# Curve «vere» del giorno di valutazione: Bund 2a 2,00%, 10a 2,60%, 30a 3,10%; BTP 2a 2,35%, 10a 3,50%, 30a 4,40%
BUND_VERA = Svensson(0.03865, -0.01845, -0.02047, -0.02409, 1.5, 8.0)
BTP_VERA = Svensson(0.05771, -0.03562, -0.03453, -0.04307, 1.5, 8.0)

# (scadenza, cedola) – BTP semestrali, Bund annuali
_BTP = [
    ("2027-02-01", 0.0085), ("2027-08-01", 0.0205), ("2028-03-01", 0.0315), ("2028-12-01", 0.0280),
    ("2029-06-01", 0.0270), ("2030-02-01", 0.0350), ("2030-10-01", 0.0315), ("2031-08-01", 0.0060),
    ("2032-06-01", 0.0345), ("2033-03-01", 0.0440), ("2033-12-01", 0.0420), ("2034-08-01", 0.0500),
    ("2035-07-01", 0.0385), ("2036-03-01", 0.0225), ("2037-02-01", 0.0400), ("2038-09-01", 0.0350),
    ("2039-08-01", 0.0500), ("2040-10-01", 0.0310), ("2042-09-01", 0.0450), ("2044-09-01", 0.0475),
    ("2046-03-01", 0.0325), ("2048-09-01", 0.0345), ("2051-09-01", 0.0170), ("2053-09-01", 0.0450),
    ("2056-10-01", 0.0420),
]
_BUND = [
    ("2027-02-15", 0.0025), ("2028-02-15", 0.0050), ("2029-08-15", 0.0210), ("2030-08-15", 0.0000),
    ("2031-08-15", 0.0000), ("2032-02-15", 0.0170), ("2033-02-15", 0.0230), ("2034-02-15", 0.0220),
    ("2035-02-15", 0.0250), ("2036-02-15", 0.0260), ("2039-05-15", 0.0400), ("2044-07-04", 0.0250),
    ("2048-08-15", 0.0125), ("2053-08-15", 0.0180), ("2056-08-15", 0.0290),
]
# Scarti specifici di rendimento in punti base (positivo = titolo a sconto, rende più della curva)
_SCARTI = {"BTP-2033-03": 11.0, "BTP-2039-08": 14.0, "BTP-2046-03": 9.0, "BTP-2031-08": -10.0,
           "BTP-2051-09": -12.0, "BUND-2034-02": -6.0}


def _codice(emittente: str, scadenza: str) -> str:
    return f"{emittente.upper()}-{scadenza[:7]}"


def universo() -> list[Obbligazione]:
    titoli = [Obbligazione(_codice("BTP", s), "BTP", c, date.fromisoformat(s), 2) for s, c in _BTP]
    titoli += [Obbligazione(_codice("Bund", s), "Bund", c, date.fromisoformat(s), 1) for s, c in _BUND]
    return titoli


def prezzi_dimostrativi(valutazione: date = VALUTAZIONE, seed: int = 11) -> pd.Series:
    """Prezzi secchi: valore sulla curva vera con uno scarto specifico per titolo, arrotondati al millesimo."""
    rng = np.random.default_rng(seed)
    prezzi = {}
    for t in universo():
        if t.scadenza <= valutazione:
            continue
        bp = _SCARTI.get(t.codice, float(rng.normal(0, 1.5)))
        curva = CurvaSpostata(BTP_VERA if t.emittente == "BTP" else BUND_VERA, parallelo(bp))
        prezzi[t.codice] = round(t.prezzo(curva, valutazione), 3)
    return pd.Series(prezzi, name="Prezzo")


# --------------------------------------------------------------------------- #
# Storia delle curve
# --------------------------------------------------------------------------- #
# data: (Bund 2a, Bund 10a, Bund 30a, spread BTP-Bund a 10a, inclinazione dello spread)
# inclinazione < 0: spread brevi più bassi dei lunghi (normale); > 0: spread brevi più alti (crisi).
_ANCORE = {
    "2008-01": (0.0390, 0.0410, 0.0440, 0.0045, -0.4),
    "2008-10": (0.0290, 0.0400, 0.0430, 0.0110, -0.2),
    "2009-03": (0.0130, 0.0300, 0.0390, 0.0140, -0.3),
    "2010-06": (0.0060, 0.0260, 0.0320, 0.0160, -0.3),
    "2011-04": (0.0170, 0.0340, 0.0380, 0.0140, -0.3),
    "2011-11": (0.0040, 0.0190, 0.0250, 0.0520, 0.5),
    "2012-07": (-0.0005, 0.0130, 0.0210, 0.0500, 0.3),
    "2013-06": (0.0015, 0.0160, 0.0250, 0.0280, -0.2),
    "2014-12": (-0.0010, 0.0055, 0.0140, 0.0135, -0.4),
    "2016-07": (-0.0065, -0.0015, 0.0040, 0.0130, -0.5),
    "2018-01": (-0.0060, 0.0050, 0.0120, 0.0150, -0.5),
    "2018-10": (-0.0060, 0.0045, 0.0110, 0.0310, -0.1),
    "2019-08": (-0.0090, -0.0070, -0.0030, 0.0190, -0.4),
    "2020-02": (-0.0065, -0.0040, 0.0010, 0.0135, -0.4),
    "2020-03": (-0.0070, -0.0030, 0.0020, 0.0260, 0.0),
    "2021-02": (-0.0070, -0.0040, 0.0020, 0.0095, -0.5),
    "2021-12": (-0.0065, -0.0020, 0.0010, 0.0135, -0.5),
    "2022-06": (0.0090, 0.0150, 0.0160, 0.0220, -0.3),
    "2022-10": (0.0200, 0.0215, 0.0200, 0.0240, -0.3),
    "2023-10": (0.0315, 0.0285, 0.0310, 0.0195, -0.4),
    "2024-09": (0.0215, 0.0215, 0.0250, 0.0140, -0.5),
    "2025-06": (0.0185, 0.0255, 0.0300, 0.0100, -0.5),
    "2026-10": (0.0200, 0.0260, 0.0310, 0.0090, -0.5),
}


def _ns_da_tre(y2: float, y10: float, y30: float, tau: float = 2.0) -> Svensson:
    """Curva di Nelson-Siegel che passa esattamente per i tassi a 2, 10 e 30 anni."""
    from .curve import _carico

    pend, curv = _carico(np.array([2.0, 10.0, 30.0]), tau)
    a = np.column_stack([np.ones(3), pend, curv])
    beta = np.linalg.solve(a, np.array([y2, y10, y30]))
    return Svensson(beta[0], beta[1], beta[2], 0.0, tau, 10.0)


def storia_curve(seed: int = 5) -> pd.DataFrame:
    """Tassi zero mensili di Bund e BTP sui nodi di ``NODI_STORIA``. Colonne a due livelli: (emittente, nodo)."""
    date_mensili = pd.date_range("2008-01-31", "2026-10-31", freq="ME")
    anc = pd.DataFrame(_ANCORE, index=["y2", "y10", "y30", "s10", "incl"]).T
    anc.index = pd.to_datetime(anc.index) + pd.offsets.MonthEnd(0)
    anc = anc.reindex(anc.index.union(date_mensili)).interpolate(method="time").loc[date_mensili]
    rng = np.random.default_rng(seed)
    n = len(date_mensili)

    def ar(sigma, phi=0.6):
        x = np.zeros(n)
        e = rng.normal(0, sigma, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        x[-1] = 0.0  # l'ultimo mese coincide con le ancore di oggi
        return x

    livello, pend, curv = ar(0.0012), ar(0.0008), ar(0.0006)
    spread_r = ar(0.0012, 0.5)
    nodi = np.array(NODI_STORIA)
    righe = []
    for i, d in enumerate(date_mensili):
        a = anc.loc[d]
        bund = _ns_da_tre(a.y2 + livello[i] - pend[i], a.y10 + livello[i] + curv[i], a.y30 + livello[i] + pend[i])
        z_bund = bund.zero(nodi)
        s10 = max(a.s10 + spread_r[i], 0.002)
        forma = 1 + a.incl * (1 - np.minimum(nodi, 10) / 10) + 0.012 * np.maximum(nodi - 10, 0)
        z_btp = z_bund + s10 * np.clip(forma, 0.2, None)
        righe.append(np.r_[z_bund, z_btp])
    # raccordo sugli ultimi 12 mesi: la storia termina esattamente sulle curve vere di oggi
    righe = np.array(righe)
    oggi = np.r_[BUND_VERA.zero(nodi), BTP_VERA.zero(nodi)]
    rampa = np.clip(1 - (n - 1 - np.arange(n)) / 12, 0, 1)
    righe += rampa[:, None] * (oggi - righe[-1])
    colonne = pd.MultiIndex.from_product([["Bund", "BTP"], [f"{x:g}a" for x in nodi]])
    return pd.DataFrame(righe, index=date_mensili, columns=colonne)


def salva_storia(storia: pd.DataFrame, percorso: str | Path) -> None:
    piatta = storia.copy()
    piatta.columns = [f"{a} {b}" for a, b in storia.columns]
    piatta.round(6).to_csv(percorso, index_label="data")


def carica_storia(percorso: str | Path) -> pd.DataFrame:
    """Storia mensile dei tassi zero (in decimali) da CSV: colonna ``data`` e colonne «Bund 10a», «BTP 10a», …
    con gli stessi nodi per le due curve."""
    df = pd.read_csv(percorso, index_col="data", parse_dates=True)
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split(" ", 1)) for c in df.columns])
    if set(df.columns.get_level_values(0)) != {"Bund", "BTP"} or not df["Bund"].columns.equals(df["BTP"].columns):
        raise ValueError("La storia deve avere colonne «Bund <nodo>» e «BTP <nodo>» con gli stessi nodi")
    return df[["Bund", "BTP"]].sort_index()


def carica_titoli(percorso: str | Path) -> tuple[list[Obbligazione], pd.Series]:
    """Titoli e prezzi da CSV con colonne: codice, emittente, cedola (in %), scadenza (AAAA-MM-GG), frequenza,
    prezzo."""
    df = pd.read_csv(percorso)
    titoli = [Obbligazione(r.codice, r.emittente, float(r.cedola) / 100, date.fromisoformat(str(r.scadenza)),
                           int(r.frequenza)) for r in df.itertuples()]
    return titoli, pd.Series(df["prezzo"].to_numpy(dtype=float), index=df["codice"], name="Prezzo")


def salva_titoli(titoli: list[Obbligazione], prezzi: pd.Series, percorso: str | Path) -> None:
    pd.DataFrame([{"codice": t.codice, "emittente": t.emittente, "cedola": round(t.cedola * 100, 4),
                   "scadenza": t.scadenza.isoformat(), "frequenza": t.frequenza, "prezzo": prezzi[t.codice]}
                  for t in titoli if t.codice in prezzi.index]).to_csv(percorso, index=False)
