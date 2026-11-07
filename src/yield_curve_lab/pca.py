"""Analisi delle componenti principali delle variazioni della curva: livello, pendenza e curvatura.

Litterman e Scheinkman (1991) hanno mostrato che tre fattori spiegano quasi tutta la variabilità dei tassi.
Qui la PCA è fatta sulle variazioni mensili dei tassi zero (in punti base) di una curva o dello spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NOMI_FATTORI = ("Livello", "Pendenza", "Curvatura")


@dataclass
class RisultatoPCA:
    carichi: pd.DataFrame  # nodi × fattori: spostamento in pb per una deviazione standard del fattore
    varianza_spiegata: pd.Series  # quota della varianza totale
    fattori: pd.DataFrame  # serie storica dei fattori (deviazioni standard)


def variazioni(storia: pd.DataFrame, curva: str, mesi: int = 1) -> pd.DataFrame:
    """Variazioni dei tassi zero in punti base su ``mesi`` mesi. ``curva``: "Bund", "BTP" o "Spread"."""
    if curva == "Spread":
        livelli = storia["BTP"] - storia["Bund"]
    else:
        livelli = storia[curva]
    return (livelli.diff(mesi).dropna() * 1e4)


def componenti_principali(storia: pd.DataFrame, curva: str = "Bund", n: int = 3, mesi: int = 1) -> RisultatoPCA:
    dz = variazioni(storia, curva, mesi)
    cov = np.cov(dz.to_numpy().T)
    autovalori, autovettori = np.linalg.eigh(cov)
    ordine = np.argsort(autovalori)[::-1]
    autovalori, autovettori = autovalori[ordine], autovettori[:, ordine]
    # segni convenzionali: livello positivo, pendenza crescente con la scadenza, curvatura positiva al centro
    for k in range(min(n, 3)):
        v = autovettori[:, k]
        riferimento = (v.sum(), v[-1] - v[0], v[len(v) // 2] - (v[0] + v[-1]) / 2)[k]
        if riferimento < 0:
            autovettori[:, k] = -v
    nomi = [NOMI_FATTORI[k] if k < 3 else f"Fattore {k + 1}" for k in range(n)]
    carichi = pd.DataFrame(autovettori[:, :n] * np.sqrt(autovalori[:n]), index=dz.columns, columns=nomi)
    spiegata = pd.Series(autovalori[:n] / autovalori.sum(), index=nomi, name="Varianza spiegata")
    centrate = dz - dz.mean()
    fattori = pd.DataFrame(centrate.to_numpy() @ autovettori[:, :n] / np.sqrt(autovalori[:n]), index=dz.index,
                           columns=nomi)
    return RisultatoPCA(carichi, spiegata, fattori)
