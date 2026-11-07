"""Riga di comando: curve stimate, ricco/caro, rischio del portafoglio, copertura, immunizzazione e report.

Esempi::

    yield-curve-lab --demo
    yield-curve-lab --config esempi/cliente.json --output output/report.html
    yield-curve-lab --demo --offline --grafici docs/img
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import configurazione
from .analisi import analizza
from .report import _eur, build_report, salva_grafici

NOTA_DEMO = ("Analisi svolta su dati simulati: curve, prezzi e storia ricalcano per livelli, date e ordine di "
             "grandezza il mercato dei titoli di Stato italiani e tedeschi, ma i titoli e i numeri non sono reali.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="yield-curve-lab", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="File JSON con portafoglio, coperture e obiettivo")
    src.add_argument("--demo", action="store_true", help="Usa titoli, prezzi e portafoglio dimostrativi")
    p.add_argument("--cliente", help="Nome del cliente nel report")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG (richiede kaleido)")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = configurazione.esempio() if args.demo else configurazione.carica(args.config)
    if args.demo or not (cfg.titoli or cfg.storia):
        nota = NOTA_DEMO
    elif not cfg.storia:
        nota = "La storia delle curve usata per PCA, VaR e scenari storici è simulata."
    elif not cfg.titoli:
        nota = "Titoli e prezzi sono simulati."
    else:
        nota = ""
    if args.cliente:
        cfg.cliente = args.cliente
    a = analizza(cfg)
    path = build_report(a, args.output, data_note=nota, offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    pd.set_option("display.width", 200)
    for em, s in a.stime.items():
        z = [f"{s.curva.zero(t) * 100:.2f}%" for t in (2.0, 10.0, 30.0)]
        print(f"Curva {em}: zero 2a {z[0]}, 10a {z[1]}, 30a {z[2]} · errore {s.rmse_bp:.1f} pb")
    s = a.sintesi
    print(f"\nPortafoglio: {_eur(s['Controvalore'])}, duration {s['Duration modificata']:.2f}, "
          f"DV01 {_eur(s['DV01 (€)'])}, spread DV01 {_eur(s['Spread DV01 (€)'])}")
    v = a.var["1 mese"]
    print(f"VaR {a.config.livello_var:.0%} a 1 mese {_eur(v['VaR'])}, CVaR {_eur(v['CVaR'])}")
    print("\nScenari (€)")
    tab = pd.DataFrame({"Senza copertura": a.stress["Utile/perdita"],
                        "Con copertura": a.stress_coperto["Utile/perdita"]}).round(0)
    print(tab.to_string())
    if a.immunizzazione is not None:
        i = a.immunizzazione
        print(f"\nImmunizzazione di «{cfg.obiettivo.descrizione}»: "
              + " + ".join(f"{_eur(p.nominale)} {p.titolo.nome}" for p in i.portafoglio.posizioni))
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
