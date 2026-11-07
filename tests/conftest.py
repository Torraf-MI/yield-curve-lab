import pytest

from yield_curve_lab import configurazione, dati
from yield_curve_lab.analisi import analizza


@pytest.fixture(scope="session")
def storia():
    return dati.storia_curve()


@pytest.fixture(scope="session")
def analisi_demo(storia):
    return analizza(configurazione.esempio(), storia=storia)
