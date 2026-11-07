"""Portafoglio obbligazionario del cliente: controvalore, sensibilità, scenari e VaR storico.

Le sensibilità sono espresse in euro: il DV01 è la perdita per un rialzo di 1 punto base, le key rate
duration in euro (KRD €) sono i DV01 per nodo della curva e sommano al DV01, la spread duration in euro
misura la perdita se si allarga di 1 punto base solo lo spread dei BTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .curve import CurvaSpostata, parallelo
from .obbligazioni import NODI_KRD, Obbligazione, shock_krd
from .pca import variazioni
from .scenari import Scenario, da_variazione


@dataclass(frozen=True)
class Posizione:
    titolo: Obbligazione
    nominale: float  # in euro; negativo = posizione corta (per esempio una copertura)


class Portafoglio:
    def __init__(self, posizioni: list[Posizione], curve: dict[str, object], valutazione: date):
        self.posizioni = [p for p in posizioni if p.nominale != 0]
        self.curve = curve
        self.valutazione = valutazione

    # ------------------------------------------------------------------ #
    def _valori(self, curve: dict[str, object] | None = None) -> np.ndarray:
        curve = curve or self.curve
        return np.array([p.nominale / 100 * p.titolo.prezzo_tel_quel(curve[p.titolo.emittente], self.valutazione)
                         for p in self.posizioni])

    def valore(self, curve: dict[str, object] | None = None) -> float:
        return float(self._valori(curve).sum())

    def curve_spostate(self, shock_bund, shock_btp) -> dict[str, object]:
        return {"Bund": CurvaSpostata(self.curve["Bund"], shock_bund),
                "BTP": CurvaSpostata(self.curve["BTP"], shock_btp)}

    def con(self, altre: list[Posizione]) -> Portafoglio:
        return Portafoglio(self.posizioni + list(altre), self.curve, self.valutazione)

    # ------------------------------------------------------------------ #
    def tabella(self) -> pd.DataFrame:
        righe = []
        valori = self._valori()
        for p, v in zip(self.posizioni, valori):
            t, curva = p.titolo, self.curve[p.titolo.emittente]
            prezzo = t.prezzo(curva, self.valutazione)
            y = t.rendimento(prezzo, self.valutazione)
            m = t.misure(y, self.valutazione)
            righe.append({"Codice": t.codice, "Titolo": t.nome, "Nominale": p.nominale, "Prezzo": prezzo,
                          "Rendimento": y, "Controvalore": v, "Duration modificata": m["Duration modificata"],
                          "Convessità": m["Convessità"], "DV01 (€)": m["DV01"] * p.nominale / 100})
        df = pd.DataFrame(righe).set_index("Codice")
        df["Peso"] = df["Controvalore"] / df["Controvalore"].sum()
        return df

    def sintesi(self) -> dict[str, float]:
        tab = self.tabella()
        v = tab["Controvalore"].sum()
        return {"Controvalore": v, "Nominale": tab["Nominale"].sum(),
                "Rendimento medio": float((tab["Rendimento"] * tab["Controvalore"]).sum() / v),
                "Duration modificata": float((tab["Duration modificata"] * tab["Controvalore"]).sum() / v),
                "DV01 (€)": self.dv01(), "Spread DV01 (€)": self.spread_dv01()}

    def dv01(self) -> float:
        return self.perdita(parallelo(1), parallelo(1), centrale=True)

    def spread_dv01(self) -> float:
        return self.perdita(parallelo(0), parallelo(1), centrale=True)

    def perdita(self, shock_bund, shock_btp, centrale: bool = False) -> float:
        """Perdita (positiva se il valore scende) per uno shock; ``centrale`` usa la differenza simmetrica."""
        v0 = self.valore()
        su = self.valore(self.curve_spostate(shock_bund, shock_btp))
        if not centrale:
            return v0 - su
        giu = self.valore(self.curve_spostate(lambda t: -shock_bund(t), lambda t: -shock_btp(t)))
        return (giu - su) / 2

    def krd(self, nodi: tuple[float, ...] = NODI_KRD, solo: str | None = None) -> pd.Series:
        """KRD in euro per nodo: tassi di entrambe le curve (``solo=None``) o di una sola."""
        zero = parallelo(0)
        valori = []
        for s in shock_krd(nodi):
            valori.append(self.perdita(s if solo in (None, "Bund") else zero, s if solo in (None, "BTP") else zero,
                                       centrale=True))
        return pd.Series(valori, index=[f"{n:g}a" for n in nodi], name="KRD (€ per pb)")

    def krd_per_posizione(self, nodi: tuple[float, ...] = NODI_KRD) -> pd.DataFrame:
        return pd.DataFrame({p.titolo.codice: Portafoglio([p], self.curve, self.valutazione).krd(nodi)
                             for p in self.posizioni}).T

    # ------------------------------------------------------------------ #
    def scenario(self, s: Scenario, krd: dict[str, pd.Series] | None = None) -> dict[str, float]:
        """Utile o perdita dello scenario con rivalutazione completa, confrontata con due approssimazioni:
        duration e convessità di ciascun titolo (lo shock letto alla sua duration) e, se si passano le KRD in euro
        per curva (``krd``), la somma delle KRD per lo shock sui nodi."""
        v0 = self.valore()
        pieno = self.valore(self.curve_spostate(s.bund, s.btp)) - v0
        approssimato = 0.0
        tab = self.tabella()
        for p, (_, r) in zip(self.posizioni, tab.iterrows()):
            t, dm = p.titolo, r["Duration modificata"]
            # variazione del rendimento letta sullo shock alla duration di Macaulay del titolo
            dy = float(s.shock(t.emittente)(np.array([dm * (1 + r["Rendimento"] / t.frequenza)]))[0])
            approssimato += r["Controvalore"] * (-dm * dy + 0.5 * r["Convessità"] * dy**2)
        con_krd = None
        if krd is not None:
            nodi = np.array([float(c[:-1]) for c in krd["Bund"].index])
            con_krd = -float(krd["Bund"].to_numpy() @ (s.bund(nodi) * 1e4)
                             + krd["BTP"].to_numpy() @ (s.btp(nodi) * 1e4))
        return {"Scenario": s.nome, "Utile/perdita": pieno, "Utile/perdita %": pieno / v0,
                "Duration e convessità": approssimato, "Key rate duration": con_krd}

    def scenari(self, elenco: list[Scenario]) -> pd.DataFrame:
        krd = {"Bund": self.krd(solo="Bund"), "BTP": self.krd(solo="BTP")}
        return pd.DataFrame([self.scenario(s, krd) for s in elenco]).set_index("Scenario")

    # ------------------------------------------------------------------ #
    def pnl_storici(self, storia: pd.DataFrame, mesi: int = 1) -> pd.Series:
        """Utile o perdita di oggi se le curve si muovessero come in ciascun periodo di ``mesi`` della storia."""
        d_bund, d_btp = variazioni(storia, "Bund", mesi), variazioni(storia, "BTP", mesi)
        nodi = np.array([float(c[:-1]) for c in d_bund.columns])
        v0 = self.valore()
        pnl = [self.valore(self.curve_spostate(s.bund, s.btp)) - v0
               for s in (da_variazione(nodi, a, b) for a, b in zip(d_bund.to_numpy(), d_btp.to_numpy()))]
        return pd.Series(pnl, index=d_bund.index, name=f"Utile/perdita a {mesi} mesi")

    def var_storico(self, storia: pd.DataFrame, livello: float = 0.99, mesi: int = 1) -> dict[str, float]:
        pnl = self.pnl_storici(storia, mesi)
        soglia = np.quantile(pnl, 1 - livello)
        return {"VaR": float(-soglia), "CVaR": float(-pnl[pnl <= soglia].mean()), "Peggiore": float(-pnl.min()),
                "Data peggiore": pnl.idxmin(), "Osservazioni": len(pnl)}


def peggior_periodo(portafoglio: Portafoglio, storia: pd.DataFrame, mesi: int = 12) -> Scenario:
    """Lo scenario storico di ``mesi`` mesi peggiore per questo portafoglio."""
    pnl = portafoglio.pnl_storici(storia, mesi)
    fine = pnl.idxmin()
    inizio = storia.index[storia.index.get_loc(fine) - mesi]
    d = (storia.loc[fine] - storia.loc[inizio]) * 1e4
    nodi = np.array([float(c[:-1]) for c in storia["Bund"].columns])
    s = da_variazione(nodi, d["Bund"].to_numpy(), d["BTP"].to_numpy())
    return Scenario(f"Peggiori {mesi} mesi ({inizio:%m/%Y} → {fine:%m/%Y})",
                    "Il periodo storico che avrebbe fatto perdere di più a questo portafoglio", s.bund, s.btp)
