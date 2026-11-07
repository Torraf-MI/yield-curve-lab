"""Copertura del rischio tasso con Bund e immunizzazione di un obiettivo del cliente.

**Copertura.** Si cercano i nominali di alcuni Bund (venduti allo scoperto o, in pratica, con i future Schatz,
Bobl, Bund e Buxl) che annullano le key rate duration in euro del portafoglio, ai minimi quadrati. Il rischio
tasso sparisce, quello di spread dei BTP resta: è la scomposizione che un gestore usa per decidere quale dei due
rischi vuole tenere.

**Immunizzazione.** Un obiettivo futuro (una spesa certa a una data) è una passività: la si copre con due titoli
che hanno lo stesso valore attuale e la stessa duration (Redington, 1952). Se la convessità del portafoglio
supera quella della passività, uno spostamento parallelo dei tassi genera sempre un surplus; uno spostamento non
parallelo invece può far perdere, ed è il rischio che resta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .curve import CurvaSpostata, parallelo
from .obbligazioni import NODI_KRD, Obbligazione
from .portafoglio import Portafoglio, Posizione


@dataclass
class Copertura:
    posizioni: list[Posizione]  # le posizioni di copertura (nominali negativi = vendite)
    krd_prima: pd.Series
    krd_dopo: pd.Series
    coperto: Portafoglio


def copertura_krd(portafoglio: Portafoglio, strumenti: list[Obbligazione], nodi: tuple[float, ...] = NODI_KRD,
                  lotto: float = 1000.0) -> Copertura:
    """Nominali degli ``strumenti`` che minimizzano la somma dei quadrati delle KRD in euro residue."""
    prima = portafoglio.krd(nodi)
    unitari = np.column_stack([Portafoglio([Posizione(s, 100.0)], portafoglio.curve, portafoglio.valutazione)
                               .krd(nodi).to_numpy() / 100.0 for s in strumenti])
    # una riga in più, con peso alto, chiede che anche il DV01 totale sia nullo
    a = np.vstack([unitari, 10 * unitari.sum(axis=0)])
    b = -np.r_[prima.to_numpy(), 10 * prima.sum()]
    n, *_ = np.linalg.lstsq(a, b, rcond=None)
    n = np.round(n / lotto) * lotto
    posizioni = [Posizione(s, float(x)) for s, x in zip(strumenti, n)]
    coperto = portafoglio.con(posizioni)
    return Copertura(posizioni, prima, coperto.krd(nodi), coperto)


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Obiettivo:
    descrizione: str
    importo: float
    data: date


def valore_obiettivo(ob: Obiettivo, curva, valutazione: date) -> float:
    t = (pd.Timestamp(ob.data) - pd.Timestamp(valutazione)).days / 365.0
    return float(ob.importo * curva.sconto(np.array([t]))[0])


@dataclass
class Immunizzazione:
    obiettivo: Obiettivo
    portafoglio: Portafoglio
    valore_passivita: float
    duration_passivita: float
    convessita_passivita: float
    duration_portafoglio: float
    convessita_portafoglio: float


def _misure_effettive(valore, bp: float = 10.0) -> tuple[float, float, float]:
    """Valore, duration e convessità effettive di una funzione ``valore(shock)`` della curva."""
    v0, su, giu = valore(parallelo(0)), valore(parallelo(bp)), valore(parallelo(-bp))
    h = bp / 1e4
    return v0, (giu - su) / (2 * v0 * h), (giu + su - 2 * v0) / (v0 * h**2)


def immunizza(ob: Obiettivo, titoli: tuple[Obbligazione, Obbligazione], curve: dict[str, object],
              valutazione: date) -> Immunizzazione:
    """Due titoli con valore e duration (effettive, sulla curva) pari a quelli dell'obiettivo."""
    curva = curve[titoli[0].emittente]

    def passivita(shock):
        return valore_obiettivo(ob, CurvaSpostata(curva, shock), valutazione)

    vl, dl, cl = _misure_effettive(passivita)
    misure = []
    for t in titoli:
        v, d, c = _misure_effettive(lambda s, t=t: t.prezzo_tel_quel(CurvaSpostata(curve[t.emittente], s),
                                                                      valutazione))
        misure.append((v, d))
    (v1, d1), (v2, d2) = misure
    # n1·v1 + n2·v2 = VL  e  n1·v1·d1 + n2·v2·d2 = VL·DL  (nominali per 100)
    a = np.array([[v1, v2], [v1 * d1, v2 * d2]])
    n = np.linalg.solve(a, np.array([vl, vl * dl]))
    port = Portafoglio([Posizione(t, float(x) * 100) for t, x in zip(titoli, n)], curve, valutazione)

    def attivo(shock):
        return port.valore(port.curve_spostate(shock, shock))

    _, dp, cp = _misure_effettive(attivo)
    return Immunizzazione(ob, port, vl, dl, cl, dp, cp)


def surplus(port: Portafoglio, ob: Obiettivo, shock, emittente: str = "BTP") -> float:
    """Attivo − valore dell'obiettivo dopo lo ``shock`` (stesso shock su tutte le curve). L'obiettivo si sconta
    sulla curva dell'``emittente`` dei titoli che lo coprono."""
    attivo = port.valore(port.curve_spostate(shock, shock))
    return attivo - valore_obiettivo(ob, CurvaSpostata(port.curve[emittente], shock), port.valutazione)


def titolo_singolo(ob: Obiettivo, titolo: Obbligazione, curve: dict[str, object], valutazione: date) -> Portafoglio:
    """Tutto su un solo titolo, con lo stesso valore attuale dell'obiettivo (senza pareggiare la duration)."""
    vl = valore_obiettivo(ob, curve[titolo.emittente], valutazione)
    nominale = vl / titolo.prezzo_tel_quel(curve[titolo.emittente], valutazione) * 100
    return Portafoglio([Posizione(titolo, nominale)], curve, valutazione)


def scegli_coppia(ob: Obiettivo, titoli: list[Obbligazione], curve: dict[str, object], valutazione: date,
                  emittente: str = "BTP", distanza: float = 4.0) -> tuple[Obbligazione, Obbligazione]:
    """Un titolo che scade circa ``distanza`` anni prima dell'obiettivo e uno circa ``distanza`` anni dopo."""
    t_ob = (pd.Timestamp(ob.data) - pd.Timestamp(valutazione)).days / 365.0
    candidati = [t for t in titoli if t.emittente == emittente and t.scadenza > valutazione]
    prima = min((t for t in candidati if t.vita_residua(valutazione) < t_ob),
                key=lambda t: abs(t.vita_residua(valutazione) - (t_ob - distanza)))
    dopo = min((t for t in candidati if t.vita_residua(valutazione) > t_ob),
               key=lambda t: abs(t.vita_residua(valutazione) - (t_ob + distanza)))
    return prima, dopo
