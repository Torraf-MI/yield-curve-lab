"""Titoli a cedola fissa: calendario delle cedole, rateo, prezzo da una curva, rendimento e misure di rischio.

Convenzioni: rateo ACT/ACT (ICMA) come per BTP e Bund, tempi dei flussi in anni ACT/365, rendimento a
scadenza composto con la frequenza delle cedole. Prezzi per 100 di nominale; il regolamento coincide con la data
di valutazione.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .curve import CurvaSpostata, parallelo, triangolo

NODI_KRD = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)


def _anni(da: date, a: date) -> float:
    return (pd.Timestamp(a) - pd.Timestamp(da)).days / 365.0


@dataclass(frozen=True)
class Obbligazione:
    codice: str
    emittente: str  # "BTP" o "Bund"
    cedola: float  # annua, in decimali (0,035 = 3,5%)
    scadenza: date
    frequenza: int = 2
    nome: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.nome:
            object.__setattr__(self, "nome", f"{self.emittente} {self.cedola * 100:.2f}% {self.scadenza:%m/%Y}"
                               .replace(".", ","))

    # ------------------------------------------------------------------ #
    def date_cedole(self, valutazione: date) -> tuple[date, list[date]]:
        """Ultima cedola pagata (o data di godimento teorica) e cedole future, ricostruite dalla scadenza."""
        mesi = 12 // self.frequenza
        scad = pd.Timestamp(self.scadenza)
        val = pd.Timestamp(valutazione)
        if scad <= val:
            raise ValueError(f"{self.nome} è già scaduto alla data di valutazione")
        future = []
        k = 0
        while True:
            d = scad - pd.DateOffset(months=mesi * k)
            if d <= val:
                return d.date(), [x.date() for x in reversed(future)]
            future.append(d)
            k += 1

    def rateo(self, valutazione: date) -> float:
        precedente, future = self.date_cedole(valutazione)
        prossima = future[0]
        frazione = (pd.Timestamp(valutazione) - pd.Timestamp(precedente)).days / \
            (pd.Timestamp(prossima) - pd.Timestamp(precedente)).days
        return 100 * self.cedola / self.frequenza * frazione

    def flussi(self, valutazione: date) -> tuple[np.ndarray, np.ndarray]:
        """Tempi in anni e importi per 100 di nominale."""
        _, future = self.date_cedole(valutazione)
        tempi = np.array([_anni(valutazione, d) for d in future])
        importi = np.full(len(future), 100 * self.cedola / self.frequenza)
        importi[-1] += 100.0
        return tempi, importi

    def vita_residua(self, valutazione: date) -> float:
        return _anni(valutazione, self.scadenza)

    # ------------------------------------------------------------------ #
    def prezzo_tel_quel(self, curva, valutazione: date) -> float:
        t, c = self.flussi(valutazione)
        return float(c @ curva.sconto(t))

    def prezzo(self, curva, valutazione: date) -> float:
        """Prezzo secco (quotato) dalla curva."""
        return self.prezzo_tel_quel(curva, valutazione) - self.rateo(valutazione)

    def _da_rendimento(self, y: float, valutazione: date) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        t, c = self.flussi(valutazione)
        f = self.frequenza
        df = (1 + y / f) ** (-f * t)
        return float(c @ df), t, c, df

    def rendimento(self, prezzo_secco: float, valutazione: date) -> float:
        """Rendimento a scadenza (composto con la frequenza delle cedole) dato il prezzo secco."""
        tel_quel = prezzo_secco + self.rateo(valutazione)
        return brentq(lambda y: self._da_rendimento(y, valutazione)[0] - tel_quel, -0.05, 0.5, xtol=1e-12)

    def misure(self, rendimento: float, valutazione: date) -> dict[str, float]:
        """Duration di Macaulay e modificata, convessità e DV01 (per 100 di nominale) dal rendimento."""
        f = self.frequenza
        p, t, c, df = self._da_rendimento(rendimento, valutazione)
        pv = c * df
        macaulay = float((t * pv).sum() / p)
        modificata = macaulay / (1 + rendimento / f)
        convessita = float((pv * t * (t + 1 / f)).sum() / (p * (1 + rendimento / f) ** 2))
        return {"Duration di Macaulay": macaulay, "Duration modificata": modificata, "Convessità": convessita,
                "DV01": modificata * p / 1e4}


# --------------------------------------------------------------------------- #
# Misure sulla curva (non sul rendimento): funzionano anche per shock non paralleli
# --------------------------------------------------------------------------- #
def duration_effettiva(titolo: Obbligazione, curva, valutazione: date, bp: float = 1.0) -> tuple[float, float]:
    """Duration e convessità effettive: rivalutazione con la curva spostata di ±``bp`` punti base."""
    p0 = titolo.prezzo_tel_quel(curva, valutazione)
    su = titolo.prezzo_tel_quel(CurvaSpostata(curva, parallelo(bp)), valutazione)
    giu = titolo.prezzo_tel_quel(CurvaSpostata(curva, parallelo(-bp)), valutazione)
    h = bp / 1e4
    return (giu - su) / (2 * p0 * h), (giu + su - 2 * p0) / (p0 * h**2)


def shock_krd(nodi: tuple[float, ...] = NODI_KRD, bp: float = 1.0) -> list:
    return [triangolo(n, nodi[i - 1] if i else None, nodi[i + 1] if i < len(nodi) - 1 else None, bp)
            for i, n in enumerate(nodi)]


def key_rate_duration(titolo: Obbligazione, curva, valutazione: date, nodi: tuple[float, ...] = NODI_KRD,
                      bp: float = 1.0) -> pd.Series:
    """Key rate duration: sensibilità del prezzo a uno shock a triangolo su ciascun nodo. Sommano alla duration."""
    p0 = titolo.prezzo_tel_quel(curva, valutazione)
    h = bp / 1e4
    valori = []
    for i in range(len(nodi)):
        su = titolo.prezzo_tel_quel(CurvaSpostata(curva, shock_krd(nodi, bp)[i]), valutazione)
        giu = titolo.prezzo_tel_quel(CurvaSpostata(curva, shock_krd(nodi, -bp)[i]), valutazione)
        valori.append((giu - su) / (2 * p0 * h))
    return pd.Series(valori, index=[f"{n:g}a" for n in nodi], name=titolo.nome)
