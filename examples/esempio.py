"""Uso della libreria in poche righe: un BTP, la curva stimata sui prezzi e il rischio di un portafoglio."""

from datetime import date

from yield_curve_lab import Obbligazione, Portafoglio, Posizione, dati, key_rate_duration, stima_robusta
from yield_curve_lab.scenari import tassi_e_spread

v = dati.VALUTAZIONE
titoli, prezzi = dati.universo(), dati.prezzi_dimostrativi()

# 1. Un titolo: rendimento, duration, convessità e key rate duration
btp = next(t for t in titoli if t.codice == "BTP-2035-07")
y = btp.rendimento(prezzi[btp.codice], v)
print(f"{btp.nome}: prezzo {prezzi[btp.codice]:.3f}, rateo {btp.rateo(v):.3f}, rendimento {y:.3%}")
print({k: round(x, 2) for k, x in btp.misure(y, v).items()})

# 2. Curve BTP e Bund stimate sui prezzi, con i titoli ricchi e a sconto
curve = {}
for em in ("BTP", "Bund"):
    s = stima_robusta([t for t in titoli if t.emittente == em], prezzi, v)
    curve[em] = s.curva
    print(f"\n{em}: errore {s.rmse_bp:.2f} pb, zero a 10 anni {s.curva.zero(10.0):.2%}")
    print(s.ricco_caro[s.tabella["Anomalo"]].round(1).to_string())
print("\nKRD:", key_rate_duration(btp, curve["BTP"], v).round(2).to_dict())

# 3. Un portafoglio e uno scenario di allargamento dello spread
port = Portafoglio([Posizione(btp, 100_000), Posizione(Obbligazione("BTP-X", "BTP", 0.045, date(2044, 9, 1)),
                                                       50_000)], curve, v)
spread = tassi_e_spread("Spread +100", "", [0] * 5, [100] * 5)
print(f"\nValore {port.valore():,.0f} €, DV01 {port.dv01():,.0f} €, spread +100 pb: "
      f"{port.scenario(spread)['Utile/perdita']:,.0f} €")
