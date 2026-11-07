import json

import pytest

from yield_curve_lab import cli, configurazione, dati
from yield_curve_lab.analisi import analizza


def test_configurazione_da_json(tmp_path):
    titoli = dati.universo()
    dati.salva_titoli(titoli, dati.prezzi_dimostrativi(), tmp_path / "titoli.csv")
    cfg = {"cliente": "Prova", "titoli": "titoli.csv",
           "portafoglio": [{"codice": "BTP-2035-07", "nominale": 50000}, {"codice": "BTP-2035-07", "nominale": 10000}],
           "obiettivo": {"descrizione": "Casa", "importo": 100000, "data": "2032-01-31"}, "livello_var": 0.95}
    (tmp_path / "cliente.json").write_text(json.dumps(cfg))
    c = configurazione.carica(tmp_path / "cliente.json")
    assert c.portafoglio == {"BTP-2035-07": 60000.0}
    assert c.titoli == str((tmp_path / "titoli.csv").resolve())
    t, p = dati.carica_titoli(c.titoli)
    assert len(t) == len(titoli) and p["BTP-2035-07"] == dati.prezzi_dimostrativi()["BTP-2035-07"]


def test_storia_da_csv(tmp_path, storia):
    dati.salva_storia(storia, tmp_path / "storia.csv")
    letta = dati.carica_storia(tmp_path / "storia.csv")
    assert letta.shape == storia.shape and abs(letta - storia).max().max() < 1e-6
    (tmp_path / "rotta.csv").write_text("data,Bund 1a,BTP 2a\n2020-01-31,0.01,0.02\n")
    with pytest.raises(ValueError):
        dati.carica_storia(tmp_path / "rotta.csv")


@pytest.mark.parametrize("cattiva", [{}, {"portafoglio": [{"codice": "X", "nominale": -5}]},
                                     {"portafoglio": [{"codice": "X", "nominale": 5}], "livello_var": 99}])
def test_configurazioni_non_valide(cattiva):
    with pytest.raises(ValueError):
        configurazione.da_dizionario(cattiva)


def test_titolo_sconosciuto(storia):
    cfg = configurazione.esempio()
    cfg.portafoglio = {"BTP-1999-01": 1000}
    with pytest.raises(ValueError, match="BTP-1999-01"):
        analizza(cfg, storia=storia)


def test_cli_produce_il_report(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert cli.main(["--demo", "--cliente", "Famiglia Verdi", "--output", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "Famiglia Verdi" in html and "Immunizzare un obiettivo" in html and "plotly" in html
    assert html.count('class="plotly-graph-div"') == 9
    assert "Report salvato" in capsys.readouterr().out
