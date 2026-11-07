from datetime import date

import numpy as np
import pytest

from yield_curve_lab import dati
from yield_curve_lab.copertura import Obiettivo, immunizza, surplus, valore_obiettivo
from yield_curve_lab.curve import interpolato, parallelo
from yield_curve_lab.portafoglio import Portafoglio, Posizione
from yield_curve_lab.scenari import scenari_standard, storico, tassi_e_spread

V = dati.VALUTAZIONE
CURVE = {"Bund": dati.BUND_VERA, "BTP": dati.BTP_VERA}
TITOLI = {t.codice: t for t in dati.universo()}


def _port(**nominali):
    return Portafoglio([Posizione(TITOLI[c.replace("_", "-")], n) for c, n in nominali.items()], CURVE, V)


def test_valore_lineare_nei_nominali():
    a, b = _port(BTP_2033_03=100000), _port(BTP_2033_03=200000)
    assert b.valore() == pytest.approx(2 * a.valore())
    assert a.valore() == pytest.approx(1000 * TITOLI["BTP-2033-03"].prezzo_tel_quel(dati.BTP_VERA, V))


def test_krd_sommano_al_dv01_e_si_scompongono(analisi_demo):
    p = analisi_demo.portafoglio
    assert p.krd().sum() == pytest.approx(p.dv01(), rel=1e-6)
    assert (p.krd(solo="Bund") + p.krd(solo="BTP")).sum() == pytest.approx(p.dv01(), rel=1e-6)
    assert analisi_demo.krd.sum().sum() == pytest.approx(p.dv01(), rel=1e-6)
    # tutto BTP tranne un Bund: lo spread DV01 è il DV01 della parte BTP
    solo_bund = Portafoglio([x for x in p.posizioni if x.titolo.emittente == "Bund"], p.curve, V)
    assert p.spread_dv01() == pytest.approx(p.dv01() - solo_bund.dv01(), rel=1e-6)


def test_scenari_segni_e_approssimazioni(analisi_demo):
    s = analisi_demo.stress
    assert s.loc["Tassi +100 pb", "Utile/perdita"] < 0 < s.loc["Tassi −100 pb", "Utile/perdita"]
    # convessità positiva: si guadagna più di quanto si perde
    assert s.loc["Tassi −100 pb", "Utile/perdita"] > -s.loc["Tassi +100 pb", "Utile/perdita"]
    errore = (s["Duration e convessità"] - s["Utile/perdita"]).abs()
    assert errore.loc["Tassi +100 pb"] < 0.03 * abs(s.loc["Tassi +100 pb", "Utile/perdita"])
    krd = s.loc["Tassi +100 pb", "Key rate duration"]
    assert krd == pytest.approx(-100 * analisi_demo.sintesi["DV01 (€)"], rel=1e-6)


def test_scenario_spread_non_tocca_i_bund():
    p = _port(BUND_2032_02=100000)
    sc = tassi_e_spread("s", "", [0] * 5, [100] * 5)
    assert p.scenario(sc)["Utile/perdita"] == pytest.approx(0, abs=1e-8)


def test_scenari_storici(storia):
    sc = storico(storia, "2021-12", "2022-10")
    d = (storia.loc["2022-10"].iloc[-1] - storia.loc["2021-12"].iloc[-1]) * 1e4
    assert sc.bund(np.array([10.0]))[0] * 1e4 == pytest.approx(d[("Bund", "10a")])
    assert len(scenari_standard(storia)) == len(scenari_standard()) + 2


def test_var_storico(analisi_demo, storia):
    v = analisi_demo.var["1 mese"]
    assert 0 < v["VaR"] <= v["CVaR"] <= v["Peggiore"]
    assert v["Osservazioni"] == len(storia) - 1
    assert analisi_demo.var["12 mesi"]["VaR"] > v["VaR"]
    pnl = analisi_demo.pnl_mensili
    assert -pnl.min() == pytest.approx(v["Peggiore"])


def test_copertura_annulla_il_rischio_tasso_non_lo_spread(analisi_demo):
    c = analisi_demo.copertura
    assert abs(c.krd_dopo.sum()) < 0.02 * abs(c.krd_prima.sum())
    assert np.abs(c.krd_dopo).max() < 0.3 * np.abs(c.krd_prima).max()
    assert all(p.nominale < 0 and p.titolo.emittente == "Bund" for p in c.posizioni)
    prima, dopo = analisi_demo.stress, analisi_demo.stress_coperto
    assert abs(dopo.loc["Tassi +100 pb", "Utile/perdita"]) < 0.02 * abs(prima.loc["Tassi +100 pb", "Utile/perdita"])
    assert dopo.loc["Spread BTP-Bund +100 pb", "Utile/perdita"] == pytest.approx(
        prima.loc["Spread BTP-Bund +100 pb", "Utile/perdita"])
    # nella fuga verso la qualità i Bund venduti salgono: la copertura peggiora la perdita
    assert dopo.loc["Fuga verso la qualità", "Utile/perdita"] < prima.loc["Fuga verso la qualità", "Utile/perdita"]


def test_immunizzazione_redington():
    ob = Obiettivo("Casa", 300000, date(2033, 6, 30))
    imm = immunizza(ob, (TITOLI["BTP-2029-06"], TITOLI["BTP-2039-08"]), CURVE, V)
    assert imm.portafoglio.valore() == pytest.approx(valore_obiettivo(ob, dati.BTP_VERA, V), rel=1e-9)
    assert imm.duration_portafoglio == pytest.approx(imm.duration_passivita, rel=1e-6)
    assert imm.convessita_portafoglio > imm.convessita_passivita
    for bp in (-200, -50, 50, 200):
        assert surplus(imm.portafoglio, ob, parallelo(bp)) > 0
    # le torsioni invece possono far perdere
    assert surplus(imm.portafoglio, ob, interpolato([2, 30], [-50, 50])) < 0


def test_alternative_all_immunizzazione(analisi_demo):
    sur = analisi_demo.surplus
    assert (sur.iloc[:, 0] >= -1e-6).all()
    assert (sur.iloc[:, 1:].min() < -1000).all()
    assert sur.loc[0].abs().max() < 1e-6
