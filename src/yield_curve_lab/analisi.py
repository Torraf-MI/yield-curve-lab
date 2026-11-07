"""Analisi completa: curve stimate, ricco/caro, PCA, rischio del portafoglio, copertura e immunizzazione."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import dati
from .configurazione import Configurazione
from .copertura import Copertura, Immunizzazione, copertura_krd, immunizza, scegli_coppia, surplus, titolo_singolo
from .curve import interpolato, parallelo
from .obbligazioni import Obbligazione
from .pca import RisultatoPCA, componenti_principali
from .portafoglio import Portafoglio, Posizione, peggior_periodo
from .scenari import Scenario, scenari_standard
from .stima import CurvaStimata, stima_curva, stima_robusta

SPOSTAMENTI_BP = np.arange(-300, 301, 25)


@dataclass
class Analisi:
    config: Configurazione
    titoli: list[Obbligazione]
    prezzi: pd.Series
    stime: dict[str, CurvaStimata]  # Svensson robusta per emittente
    nelson_siegel: dict[str, CurvaStimata]
    storia: pd.DataFrame
    pca: dict[str, RisultatoPCA]
    portafoglio: Portafoglio
    tabella: pd.DataFrame
    sintesi: dict[str, float]
    krd: pd.DataFrame  # posizioni × nodi, in euro per punto base
    scenari: list[Scenario]
    stress: pd.DataFrame
    var: dict[str, dict]
    pnl_mensili: pd.Series
    copertura: Copertura
    stress_coperto: pd.DataFrame
    var_coperto: dict[str, float]
    immunizzazione: Immunizzazione | None = None
    alternative: dict[str, Portafoglio] = field(default_factory=dict)
    surplus: pd.DataFrame = field(default_factory=pd.DataFrame)  # spostamento parallelo × strategia
    surplus_torsioni: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def curve(self) -> dict[str, object]:
        return {k: v.curva for k, v in self.stime.items()}


def analizza(cfg: Configurazione, titoli: list[Obbligazione] | None = None, prezzi: pd.Series | None = None,
             storia: pd.DataFrame | None = None) -> Analisi:
    if titoli is None:
        if cfg.titoli:
            titoli, prezzi = dati.carica_titoli(cfg.titoli)
        else:
            titoli, prezzi = dati.universo(), dati.prezzi_dimostrativi(cfg.valutazione)
    titoli = [t for t in titoli if t.scadenza > cfg.valutazione and t.codice in prezzi.index]
    per_codice = {t.codice: t for t in titoli}
    mancanti = [c for c in list(cfg.portafoglio) + list(cfg.coperture) if c not in per_codice]
    if mancanti:
        raise ValueError(f"Titoli senza prezzo o sconosciuti: {', '.join(mancanti)}")
    if storia is None:
        storia = dati.carica_storia(cfg.storia) if cfg.storia else dati.storia_curve()

    stime, ns = {}, {}
    for em in ("BTP", "Bund"):
        gruppo = [t for t in titoli if t.emittente == em]
        stime[em] = stima_robusta(gruppo, prezzi, cfg.valutazione)
        ns[em] = stima_curva(gruppo, prezzi, cfg.valutazione, "nelson-siegel")
    curve = {k: v.curva for k, v in stime.items()}

    port = Portafoglio([Posizione(per_codice[c], n) for c, n in cfg.portafoglio.items()], curve, cfg.valutazione)
    elenco = scenari_standard(storia) + [peggior_periodo(port, storia, 12)]
    lv = cfg.livello_var
    var = {"1 mese": port.var_storico(storia, lv, 1), "12 mesi": port.var_storico(storia, lv, 12)}
    cop = copertura_krd(port, [per_codice[c] for c in cfg.coperture])

    imm, alt, sur, tor = None, {}, pd.DataFrame(), pd.DataFrame()
    if cfg.obiettivo is not None:
        ob = cfg.obiettivo
        coppia = scegli_coppia(ob, titoli, curve, cfg.valutazione)
        imm = immunizza(ob, coppia, curve, cfg.valutazione)
        btp = [t for t in titoli if t.emittente == "BTP"]
        vicino = min(btp, key=lambda t: abs((pd.Timestamp(t.scadenza) - pd.Timestamp(ob.data)).days))
        lungo = max(btp, key=lambda t: t.vita_residua(cfg.valutazione))
        breve = min(btp, key=lambda t: t.vita_residua(cfg.valutazione))
        alt = {
            f"Immunizzato: {coppia[0].nome} + {coppia[1].nome}": imm.portafoglio,
            f"Scadenza vicina: {vicino.nome}": titolo_singolo(ob, vicino, curve, cfg.valutazione),
            f"Breve (da rinnovare): {breve.nome}": titolo_singolo(ob, breve, curve, cfg.valutazione),
            f"Lungo: {lungo.nome}": titolo_singolo(ob, lungo, curve, cfg.valutazione),
        }
        sur = pd.DataFrame({n: [surplus(p, ob, parallelo(bp)) for bp in SPOSTAMENTI_BP] for n, p in alt.items()},
                           index=SPOSTAMENTI_BP)
        torsioni = {
            "Tassi +100 pb": parallelo(100), "Tassi −100 pb": parallelo(-100),
            "Irripidimento (2a −50, 30a +50)": interpolato([2, 30], [-50, 50]),
            "Appiattimento (2a +50, 30a −50)": interpolato([2, 30], [50, -50]),
            "Gobba (10a +50)": interpolato([2, 10, 30], [0, 50, 0]),
        }
        tor = pd.DataFrame({n: [surplus(p, ob, s) for s in torsioni.values()] for n, p in alt.items()},
                           index=list(torsioni))

    return Analisi(
        config=cfg, titoli=titoli, prezzi=prezzi, stime=stime, nelson_siegel=ns, storia=storia,
        pca={k: componenti_principali(storia, k) for k in ("Bund", "BTP", "Spread")},
        portafoglio=port, tabella=port.tabella(), sintesi=port.sintesi(), krd=port.krd_per_posizione(),
        scenari=elenco, stress=port.scenari(elenco), var=var, pnl_mensili=port.pnl_storici(storia, 1),
        copertura=cop, stress_coperto=cop.coperto.scenari(elenco), var_coperto=cop.coperto.var_storico(storia, lv, 1),
        immunizzazione=imm, alternative=alt, surplus=sur, surplus_torsioni=tor,
    )
