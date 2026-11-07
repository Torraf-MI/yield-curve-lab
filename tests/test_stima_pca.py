import numpy as np
import pandas as pd
import pytest

from yield_curve_lab import dati
from yield_curve_lab.pca import componenti_principali
from yield_curve_lab.stima import stima_curva, stima_robusta

V = dati.VALUTAZIONE


@pytest.fixture(scope="module")
def btp():
    return [t for t in dati.universo() if t.emittente == "BTP"]


def test_stima_su_prezzi_esatti_ritrova_la_curva(btp):
    prezzi = pd.Series({t.codice: t.prezzo(dati.BTP_VERA, V) for t in btp})
    s = stima_curva(btp, prezzi, V)
    assert s.rmse_bp < 0.05
    t = np.array([1, 5, 10, 20, 30])
    assert np.abs(s.curva.zero(t) - dati.BTP_VERA.zero(t)).max() < 2e-5


def test_svensson_meglio_di_nelson_siegel(btp):
    prezzi = pd.Series({t.codice: t.prezzo(dati.BTP_VERA, V) for t in btp})
    assert stima_curva(btp, prezzi, V).rmse_bp < stima_curva(btp, prezzi, V, "nelson-siegel").rmse_bp
    with pytest.raises(ValueError):
        stima_curva(btp, prezzi, V, "spline")


def test_stima_robusta_trova_i_titoli_anomali(btp):
    prezzi = dati.prezzi_dimostrativi()
    s = stima_robusta(btp, prezzi, V)
    anomali = set(s.tabella.index[s.tabella["Anomalo"]])
    attesi = {c for c in dati._SCARTI if c.startswith("BTP")}
    assert anomali == attesi
    for c in attesi:  # lo scarto stimato è vicino a quello simulato
        assert s.ricco_caro[c] == pytest.approx(dati._SCARTI[c], abs=2.0)
    assert s.rmse_bp < 2.0
    # la stima semplice è deformata dai titoli anomali
    assert stima_curva(btp, prezzi, V).rmse_bp > 3 * s.rmse_bp


def test_storia_finisce_sulle_curve_di_oggi(storia):
    nodi = np.array(dati.NODI_STORIA)
    assert np.allclose(storia["Bund"].iloc[-1], dati.BUND_VERA.zero(nodi))
    assert np.allclose(storia["BTP"].iloc[-1], dati.BTP_VERA.zero(nodi))
    spread = (storia[("BTP", "10a")] - storia[("Bund", "10a")]) * 1e4
    assert spread.loc["2011"].max() > 450  # crisi del debito
    assert storia[("Bund", "10a")].loc["2019":"2021"].min() < 0  # tassi negativi


def test_pca_fattori_ortogonali_e_ordinati(storia):
    r = componenti_principali(storia, "BTP")
    assert list(r.varianza_spiegata.index) == ["Livello", "Pendenza", "Curvatura"]
    assert r.varianza_spiegata.is_monotonic_decreasing and r.varianza_spiegata.sum() > 0.9
    assert (r.carichi["Livello"] > 0).all()
    assert r.carichi["Pendenza"].iloc[-1] > r.carichi["Pendenza"].iloc[0]
    c = np.corrcoef(r.fattori.to_numpy().T)
    assert np.allclose(c, np.eye(3), atol=1e-8)
    assert np.allclose(r.fattori.std(ddof=1), 1.0)
