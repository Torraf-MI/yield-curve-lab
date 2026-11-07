"""Scenari di stress: spostamenti delle curve Bund e BTP definiti sui nodi e interpolati.

Ogni scenario indica lo shock della curva Bund (i tassi «privi di rischio» dell'area euro) e lo shock della curva
BTP, che è lo shock dei tassi più quello dello spread. Gli scenari storici riapplicano alle curve di oggi le
variazioni osservate in un periodo, nodo per nodo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .curve import interpolato

Shock = Callable[[np.ndarray], np.ndarray]
NODI = (0.5, 2.0, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class Scenario:
    nome: str
    descrizione: str
    bund: Shock
    btp: Shock

    def shock(self, emittente: str) -> Shock:
        return self.btp if emittente == "BTP" else self.bund


def _su_nodi(nodi, bp) -> Shock:
    return interpolato(np.asarray(nodi, dtype=float), np.asarray(bp, dtype=float))


def tassi_e_spread(nome: str, descrizione: str, tassi_bp, spread_bp=(0, 0, 0, 0, 0), nodi=NODI) -> Scenario:
    """Scenario da uno shock dei tassi (entrambe le curve) e uno shock dello spread (solo BTP), sui nodi dati."""
    tassi_bp, spread_bp = np.asarray(tassi_bp, dtype=float), np.asarray(spread_bp, dtype=float)
    return Scenario(nome, descrizione, _su_nodi(nodi, tassi_bp), _su_nodi(nodi, tassi_bp + spread_bp))


def storico(storia: pd.DataFrame, inizio: str, fine: str, nome: str | None = None,
            descrizione: str = "") -> Scenario:
    """Variazioni osservate tra due mesi (formato AAAA-MM), riapplicate alle curve di oggi."""
    a = storia.loc[inizio].iloc[-1] if isinstance(storia.loc[inizio], pd.DataFrame) else storia.loc[inizio]
    b = storia.loc[fine].iloc[-1] if isinstance(storia.loc[fine], pd.DataFrame) else storia.loc[fine]
    d = (b - a) * 1e4
    nodi = np.array([float(c[:-1]) for c in storia["Bund"].columns])
    return Scenario(nome or f"Storico {inizio} → {fine}", descrizione, _su_nodi(nodi, d["Bund"].to_numpy()),
                    _su_nodi(nodi, d["BTP"].to_numpy()))


def da_variazione(nodi: np.ndarray, d_bund_bp: np.ndarray, d_btp_bp: np.ndarray, nome: str = "") -> Scenario:
    return Scenario(nome, "", _su_nodi(nodi, d_bund_bp), _su_nodi(nodi, d_btp_bp))


def scenari_standard(storia: pd.DataFrame | None = None) -> list[Scenario]:
    s = [
        tassi_e_spread("Tassi +100 pb", "Rialzo parallelo di 100 pb di tutte le curve", [100] * 5),
        tassi_e_spread("Tassi −100 pb", "Ribasso parallelo di 100 pb di tutte le curve", [-100] * 5),
        tassi_e_spread("Irripidimento", "Brevi −50 pb, lunghi +50 pb", [-50, -50, -20, 10, 50]),
        tassi_e_spread("Appiattimento", "Brevi +50 pb, lunghi −50 pb", [50, 50, 20, -10, -50]),
        tassi_e_spread("Spread BTP-Bund +100 pb", "Solo i BTP: spread +100 pb a 10 anni, Bund invariati",
                       [0] * 5, [60, 80, 95, 100, 105]),
        tassi_e_spread("Spread BTP-Bund −50 pb", "Solo i BTP: spread −50 pb a 10 anni, Bund invariati",
                       [0] * 5, [-30, -40, -48, -50, -52]),
        tassi_e_spread("Fuga verso la qualità", "Bund −50 pb, spread BTP +150 pb", [-50] * 5,
                       [150, 150, 150, 150, 150]),
    ]
    if storia is not None:
        s += [
            storico(storia, "2011-06", "2011-11", "Crisi del debito 2011",
                    "Giugno → novembre 2011: lo spread a 10 anni sale fino a oltre 500 pb"),
            storico(storia, "2021-12", "2022-10", "Rialzo dei tassi 2022",
                    "Dicembre 2021 → ottobre 2022: la BCE esce dai tassi negativi"),
        ]
    return s
