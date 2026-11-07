"""Curve dei tassi: Nelson-Siegel e Svensson, tassi zero, fattori di sconto e tassi forward.

Convenzione: tassi zero a capitalizzazione continua, tempo in anni (ACT/365). Il modello di Svensson (1994)
estende quello di Nelson e Siegel (1987) con una seconda «gobba»; è il modello che la Bundesbank e la BCE usano
per pubblicare le curve dei titoli di Stato.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_T_MINIMO = 1e-6


def _carico(t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Carichi di pendenza e curvatura: (1 − e^{−x})/x e (1 − e^{−x})/x − e^{−x}, con x = t/τ."""
    x = np.maximum(np.asarray(t, dtype=float), _T_MINIMO) / tau
    pendenza = -np.expm1(-x) / x
    return pendenza, pendenza - np.exp(-x)


@dataclass(frozen=True)
class Svensson:
    """Curva di Svensson; con ``beta3 = 0`` è una curva di Nelson-Siegel.

    - ``beta0``: livello di lungo periodo (tasso a scadenza infinita);
    - ``beta1``: pendenza (il tasso istantaneo è ``beta0 + beta1``);
    - ``beta2``, ``beta3``: gobbe con posizione ``tau1`` e ``tau2``.
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    tau1: float = 1.5
    tau2: float = 8.0

    @property
    def parametri(self) -> np.ndarray:
        return np.array([self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2])

    def zero(self, t) -> np.ndarray:
        """Tasso zero a capitalizzazione continua."""
        p1, c1 = _carico(t, self.tau1)
        _, c2 = _carico(t, self.tau2)
        return self.beta0 + self.beta1 * p1 + self.beta2 * c1 + self.beta3 * c2

    def forward(self, t) -> np.ndarray:
        """Tasso forward istantaneo."""
        t = np.asarray(t, dtype=float)
        e1, e2 = np.exp(-t / self.tau1), np.exp(-t / self.tau2)
        return (self.beta0 + self.beta1 * e1 + self.beta2 * (t / self.tau1) * e1
                + self.beta3 * (t / self.tau2) * e2)

    def sconto(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)

    def par(self, t: float, frequenza: int = 1) -> float:
        """Rendimento alla pari di un titolo a cedola fissa con scadenza ``t`` e ``frequenza`` cedole l'anno."""
        n = max(int(round(t * frequenza)), 1)
        tempi = np.arange(1, n + 1) / frequenza
        df = self.sconto(tempi)
        return float(frequenza * (1 - df[-1]) / df.sum())


class CurvaSpostata:
    """Una curva base più uno shock sui tassi zero, funzione della scadenza: s(t)."""

    def __init__(self, base, shock: Callable[[np.ndarray], np.ndarray]):
        self.base = base
        self.shock = shock

    def zero(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.base.zero(t) + self.shock(t)

    def sconto(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)


def parallelo(bp: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda t: np.full_like(np.asarray(t, dtype=float), bp / 1e4)


def triangolo(centro: float, sinistra: float | None, destra: float | None, bp: float = 1.0
              ) -> Callable[[np.ndarray], np.ndarray]:
    """Shock a triangolo sul nodo ``centro``, nullo sui nodi vicini: la base delle key rate duration.

    Il primo nodo resta costante a sinistra, l'ultimo a destra, così la somma degli shock è uno shock parallelo.
    """

    def f(t):
        t = np.asarray(t, dtype=float)
        s = np.zeros_like(t)
        if sinistra is None:
            s = np.where(t <= centro, 1.0, s)
        else:
            m = (t > sinistra) & (t <= centro)
            s = np.where(m, (t - sinistra) / (centro - sinistra), s)
        if destra is None:
            s = np.where(t >= centro, 1.0, s)
        else:
            m = (t > centro) & (t < destra)
            s = np.where(m, (destra - t) / (destra - centro), s)
        return s * bp / 1e4

    return f


def interpolato(nodi: np.ndarray, bp: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Shock definito sui nodi (in punti base) e interpolato linearmente, costante oltre gli estremi."""
    nodi, bp = np.asarray(nodi, dtype=float), np.asarray(bp, dtype=float)
    return lambda t: np.interp(np.asarray(t, dtype=float), nodi, bp) / 1e4
