"""Stima delle curve sui prezzi dei titoli e analisi ricco/caro.

Si minimizza la somma degli errori di prezzo pesati con l'inverso della duration, che equivale con buona
approssimazione a minimizzare gli errori di rendimento (Bliss, 1997): un errore di 10 centesimi su un titolo a 2
anni pesa come uno di 1 euro su un trentennale. I parametri non lineari (τ) rendono il problema non convesso, per
questo si parte da più punti iniziali e si tiene il migliore.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .curve import Svensson
from .obbligazioni import Obbligazione


@dataclass
class CurvaStimata:
    curva: Svensson
    modello: str
    tabella: pd.DataFrame  # per titolo: prezzo, rendimento di mercato e del modello, scarto in punti base
    rmse_bp: float  # errore quadratico medio sui rendimenti in punti base (nella stima robusta, esclusi gli anomali)

    @property
    def ricco_caro(self) -> pd.Series:
        """Scarto di rendimento mercato − modello in punti base: positivo = titolo «a sconto» (rende più della
        curva)."""
        return self.tabella["Scarto (pb)"]


def stima_curva(titoli: list[Obbligazione], prezzi: pd.Series, valutazione: date, modello: str = "svensson",
                pesi: np.ndarray | None = None) -> CurvaStimata:
    """Stima Nelson-Siegel (``modello="nelson-siegel"``) o Svensson sui prezzi secchi ``prezzi`` (indice: codice)."""
    flussi = [t.flussi(valutazione) for t in titoli]
    tel_quel = np.array([prezzi[t.codice] + t.rateo(valutazione) for t in titoli])
    rend = np.array([t.rendimento(prezzi[t.codice], valutazione) for t in titoli])
    dur = np.array([t.misure(r, valutazione)["Duration modificata"] for t, r in zip(titoli, rend)])
    w = 1 / np.maximum(dur, 0.1) if pesi is None else np.asarray(pesi, dtype=float)
    svensson = modello == "svensson"
    if modello not in ("svensson", "nelson-siegel"):
        raise ValueError(f"Modello sconosciuto: {modello}")

    def curva_da(x):
        if svensson:
            return Svensson(x[0], x[1], x[2], x[3], x[4], x[5])
        return Svensson(x[0], x[1], x[2], 0.0, x[3], 10.0)

    def residui(x):
        c = curva_da(x)
        return np.array([cf @ c.sconto(t) for t, cf in flussi]) - tel_quel

    lungo = float(np.mean(rend[np.argsort(-dur)[:3]]))
    breve = float(np.mean(rend[np.argsort(dur)[:3]]))
    migliore = None
    for tau1, tau2 in product((0.7, 2.0, 5.0), (6.0, 12.0) if svensson else (None,)):
        if svensson:
            x0 = [lungo, breve - lungo, 0.0, 0.0, tau1, tau2]
            lo, hi = [-0.05, -0.2, -0.3, -0.3, 0.05, 2.0], [0.2, 0.2, 0.3, 0.3, 10.0, 40.0]
        else:
            x0 = [lungo, breve - lungo, 0.0, tau1]
            lo, hi = [-0.05, -0.2, -0.3, 0.05], [0.2, 0.2, 0.3, 30.0]
        ris = least_squares(lambda x: w * residui(x), x0, bounds=(lo, hi), method="trf", x_scale="jac",
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
        if migliore is None or ris.cost < migliore.cost:
            migliore = ris
    curva = curva_da(migliore.x)
    modello_px = np.array([t.prezzo(curva, valutazione) for t in titoli])
    modello_y = np.array([t.rendimento(p, valutazione) for t, p in zip(titoli, modello_px)])
    tabella = pd.DataFrame({
        "Titolo": [t.nome for t in titoli],
        "Vita residua": [t.vita_residua(valutazione) for t in titoli],
        "Cedola": [t.cedola for t in titoli],
        "Prezzo": [prezzi[t.codice] for t in titoli],
        "Prezzo modello": modello_px,
        "Rendimento": rend,
        "Rendimento modello": modello_y,
        "Duration": dur,
    }, index=[t.codice for t in titoli])
    tabella["Scarto (pb)"] = (tabella["Rendimento"] - tabella["Rendimento modello"]) * 1e4
    rmse = float(np.sqrt(np.mean(tabella["Scarto (pb)"] ** 2)))
    return CurvaStimata(curva, "Svensson" if svensson else "Nelson-Siegel", tabella, rmse)


def stima_robusta(titoli: list[Obbligazione], prezzi: pd.Series, valutazione: date, soglia_bp: float = 3.0,
                  modello: str = "svensson") -> CurvaStimata:
    """Stima in due passi: i titoli molto lontani dalla prima curva (oltre 3 deviazioni robuste e ``soglia_bp``)
    pesano poco nella seconda, così pochi titoli anomali non deformano la curva su cui li si giudica."""
    prima = stima_curva(titoli, prezzi, valutazione, modello)
    scarti = prima.ricco_caro.to_numpy()
    mad = 1.4826 * np.median(np.abs(scarti - np.median(scarti)))
    limite = max(3 * mad, soglia_bp)
    dur = prima.tabella["Duration"].to_numpy()
    anomali = np.abs(scarti) > limite
    pesi = np.where(anomali, 0.05, 1.0) / np.maximum(dur, 0.1)
    seconda = stima_curva(titoli, prezzi, valutazione, modello, pesi=pesi)
    seconda.tabella["Anomalo"] = anomali
    normali = seconda.tabella.loc[~anomali, "Scarto (pb)"]
    seconda.rmse_bp = float(np.sqrt(np.mean(normali**2)))
    return seconda
