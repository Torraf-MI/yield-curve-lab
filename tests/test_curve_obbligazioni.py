from datetime import date

import numpy as np
import pytest

from yield_curve_lab import dati
from yield_curve_lab.curve import CurvaSpostata, Svensson, interpolato, parallelo, triangolo
from yield_curve_lab.obbligazioni import NODI_KRD, Obbligazione, duration_effettiva, key_rate_duration, shock_krd

V = date(2026, 10, 30)
PIATTA = Svensson(0.03, 0.0, 0.0)


def test_curva_piatta_e_limiti_svensson():
    t = np.array([0.1, 1, 10, 50])
    assert np.allclose(PIATTA.zero(t), 0.03)
    c = dati.BTP_VERA
    assert c.zero(1e-6) == pytest.approx(c.beta0 + c.beta1, abs=1e-6)
    assert c.zero(1e4) == pytest.approx(c.beta0, abs=1e-3)
    assert c.forward(1e-9) == pytest.approx(c.beta0 + c.beta1)


def test_forward_coerente_con_lo_zero():
    # f(t) = d/dt [t·z(t)]
    c, t, h = dati.BTP_VERA, 7.0, 1e-5
    derivata = ((t + h) * c.zero(t + h) - (t - h) * c.zero(t - h)) / (2 * h)
    assert c.forward(t) == pytest.approx(derivata, abs=1e-8)


def test_titolo_alla_pari_vale_cento():
    # par() usa anni esatti, i flussi i giorni effettivi (ACT/365, con i bisestili): differenza minima
    c = dati.BUND_VERA
    titolo = Obbligazione("X", "Bund", c.par(10, 1), date(2036, 10, 30), 1)
    assert titolo.prezzo(c, V) == pytest.approx(100.0, abs=0.05)
    assert PIATTA.par(5, 1) == pytest.approx(np.expm1(0.03))


def test_calendario_e_rateo():
    t = Obbligazione("BTP-X", "BTP", 0.04, date(2030, 3, 1), 2)
    precedente, future = t.date_cedole(V)
    assert precedente == date(2026, 9, 1) and future[0] == date(2027, 3, 1) and future[-1] == date(2030, 3, 1)
    assert len(future) == 7
    assert t.rateo(V) == pytest.approx(2.0 * 59 / 181)
    with pytest.raises(ValueError):
        t.date_cedole(date(2031, 1, 1))


def test_rendimento_e_prezzo_si_invertono():
    t = Obbligazione("BTP-X", "BTP", 0.035, date(2035, 7, 1), 2)
    for y in (-0.005, 0.01, 0.045, 0.08):
        tel_quel = t._da_rendimento(y, V)[0]
        assert t.rendimento(tel_quel - t.rateo(V), V) == pytest.approx(y, abs=1e-10)


def test_duration_e_convessita_approssimano_il_prezzo():
    t = Obbligazione("BTP-X", "BTP", 0.045, date(2044, 9, 1), 2)
    y = 0.038
    m = t.misure(y, V)
    p0 = t._da_rendimento(y, V)[0]
    for dy in (-0.005, 0.005):
        esatto = t._da_rendimento(y + dy, V)[0] / p0 - 1
        stima = -m["Duration modificata"] * dy + 0.5 * m["Convessità"] * dy**2
        assert stima == pytest.approx(esatto, abs=1e-4)  # resta solo il termine di terzo ordine
        assert abs(-m["Duration modificata"] * dy - esatto) > 5 * abs(stima - esatto)
    assert m["DV01"] == pytest.approx(m["Duration modificata"] * p0 / 1e4)


def test_zero_coupon_duration_uguale_alla_vita():
    t = Obbligazione("BUND-0", "Bund", 0.0, date(2031, 10, 30), 1)
    m = t.misure(0.025, V)
    assert m["Duration di Macaulay"] == pytest.approx(t.vita_residua(V), rel=1e-12)


def test_key_rate_duration_sommano_alla_duration_effettiva():
    t = Obbligazione("BTP-X", "BTP", 0.0385, date(2035, 7, 1), 2)
    krd = key_rate_duration(t, dati.BTP_VERA, V)
    d, conv = duration_effettiva(t, dati.BTP_VERA, V)
    assert krd.sum() == pytest.approx(d, rel=1e-6)
    assert conv > 0
    assert krd["20a"] == pytest.approx(0) and krd["30a"] == pytest.approx(0)  # nessun flusso oltre 9 anni


def test_shock_triangolari_sommano_a_un_parallelo():
    t = np.linspace(0, 40, 401)
    totale = sum(s(t) for s in shock_krd(NODI_KRD))
    assert np.allclose(totale, 1e-4)
    tri = triangolo(5, 3, 7)
    assert tri(np.array([3, 4, 5, 6, 7]))[2] == pytest.approx(1e-4)
    assert tri(np.array([4.0]))[0] == pytest.approx(0.5e-4)


def test_curva_spostata_e_interpolazione():
    c = CurvaSpostata(PIATTA, parallelo(50))
    assert np.allclose(c.zero([1, 10]), 0.035)
    s = interpolato([2, 10], [-20, 40])
    assert s(np.array([0.5, 6, 30])) * 1e4 == pytest.approx([-20, 10, 40])
