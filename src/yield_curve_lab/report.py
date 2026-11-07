"""Grafici interattivi (plotly) e report HTML per il cliente."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from .analisi import Analisi

BLU, ARANCIO, ACQUA, VIOLA = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLU_SCURO, ROSSO, GIALLO, GRIGIO = "#1c5cab", "#e34948", "#e8a317", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORE = {"BTP": BLU_SCURO, "Bund": ARANCIO, "Spread": VIOLA}
COLORI_POSIZIONI = ["#1c5cab", "#2a78d6", "#7fb0ea", "#8a5cd1", "#1baf7a", "#78cfae", "#e8a317", "#eb6834",
                    "#c9a227", "#8c8b86"]
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def _eur(x: float, dec: int = 0) -> str:
    s = f"{abs(x):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{'−' if x < -0.5 * 10**-dec else ''}{s} €"


def _pct(x: float, dec: int = 1) -> str:
    return f"{x * 100:.{dec}f}%".replace(".", ",").replace("-", "−")


def _num(x: float, dec: int = 2) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def _layout(fig: go.Figure, titolo: str, altezza: int = 430) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza, template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 70, "r": 20, "t": 60, "b": 50}, separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO}, legend={"orientation": "h", "y": -0.2, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def _pari(curva, t: float, f: int) -> float:
    """Rendimento alla pari per una vita residua qualunque (prima cedola anche a meno di un periodo)."""
    tempi = t - np.arange(0, int(np.ceil(t * f - 1e-9))) / f
    tempi = tempi[tempi > 0]
    rateo = 1 - tempi.min() * f  # frazione di cedola già maturata
    return float(f * (1 - curva.sconto(np.array([t]))[0]) / (curva.sconto(tempi).sum() - rateo))


def grafico_curve(a: Analisi) -> go.Figure:
    fig = go.Figure()
    t = np.linspace(0.25, 31, 200)
    for em, s in a.stime.items():
        f = 2 if em == "BTP" else 1
        tab = s.tabella
        fig.add_scatter(x=t, y=[_pari(s.curva, x, f) for x in t], name=f"{em}: curva alla pari (Svensson)",
                        line={"color": COLORE[em], "width": 2.5},
                        hovertemplate="%{x:.1f} anni: %{y:.2%}<extra>" + em + "</extra>")
        fig.add_scatter(x=t, y=s.curva.forward(t), name=f"{em}: forward istantaneo",
                        line={"color": COLORE[em], "width": 1.2, "dash": "dot"}, hoverinfo="skip")
        fig.add_scatter(x=tab["Vita residua"], y=tab["Rendimento"], mode="markers", name=f"{em}: titoli",
                        marker={"color": COLORE[em], "size": 7, "line": {"color": "white", "width": 1}},
                        text=tab["Titolo"], customdata=tab["Scarto (pb)"],
                        hovertemplate="%{text}<br>rendimento %{y:.3%}<br>scarto %{customdata:+.1f} pb<extra></extra>")
    fig.update_xaxes(title="Vita residua (anni)", range=[0, 31])
    fig.update_yaxes(tickformat=".1%", title="Rendimento")
    return _layout(fig, "Curve BTP e Bund stimate sui prezzi dei titoli", 480)


def grafico_ricco_caro(a: Analisi) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=("BTP", "Bund"))
    for col, em in enumerate(("BTP", "Bund"), start=1):
        tab = a.stime[em].tabella.sort_values("Vita residua")
        an = tab.get("Anomalo", pd.Series(False, index=tab.index))
        colori = [ROSSO if (x > 0 and m) else ACQUA if (x < 0 and m) else "#b5b3ad"
                  for x, m in zip(tab["Scarto (pb)"], an)]
        fig.add_bar(x=tab["Titolo"].str.replace(f"{em} ", ""), y=tab["Scarto (pb)"], marker_color=colori,
                    showlegend=False, row=1, col=col, hovertemplate="%{x}: %{y:+.1f} pb<extra>" + em + "</extra>")
    fig.update_yaxes(title_text="Rendimento mercato − curva (pb)", row=1, col=1)
    fig.update_xaxes(tickangle=-60, tickfont={"size": 10})
    _layout(fig, "Ricco o a sconto: scarto di ogni titolo dalla curva (rosso = a sconto, verde = caro)", 500)
    fig.update_layout(margin={"b": 120})
    return fig


def grafico_storia(a: Analisi) -> go.Figure:
    h = a.storia
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
    for em in ("BTP", "Bund"):
        fig.add_scatter(x=h.index, y=h[(em, "10a")], name=f"{em} 10 anni", line={"color": COLORE[em], "width": 2},
                        hovertemplate="%{x|%m/%Y}: %{y:.2%}<extra>" + em + " 10a</extra>", row=1, col=1)
    spread = (h[("BTP", "10a")] - h[("Bund", "10a")]) * 1e4
    fig.add_scatter(x=h.index, y=spread, name="Spread BTP-Bund 10 anni (pb)", fill="tozeroy",
                    line={"color": VIOLA, "width": 1.5}, fillcolor="rgba(138,92,209,0.18)",
                    hovertemplate="%{x|%m/%Y}: %{y:.0f} pb<extra>spread</extra>", row=2, col=1)
    for x, testo in (("2011-11-30", "Debito sovrano"), ("2018-10-31", "Spread 2018"), ("2020-03-31", "Pandemia"),
                     ("2022-10-31", "Rialzo BCE")):
        fig.add_annotation(x=x, y=float(spread.loc[x]), text=testo, showarrow=True, arrowhead=0, ay=-28,
                           font={"size": 10}, row=2, col=1)
    fig.update_yaxes(tickformat=".0%", title_text="Tasso zero", row=1, col=1)
    fig.update_yaxes(title_text="pb", row=2, col=1)
    return _layout(fig, "Tassi a 10 anni e spread BTP-Bund dal 2008", 540)


def grafico_pca(a: Analisi) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Curva BTP",
        f"Spread BTP-Bund ({_pct(a.pca['Spread'].varianza_spiegata.iloc[0], 0)} dal primo fattore)"],
        horizontal_spacing=0.1)
    colori = (BLU_SCURO, ARANCIO, ACQUA)
    for col, chiave in enumerate(("BTP", "Spread"), start=1):
        r = a.pca[chiave]
        x = [float(c[:-1]) for c in r.carichi.index]
        for k, nome in enumerate(r.carichi.columns):
            if chiave == "Spread" and k > 1:
                continue
            fig.add_scatter(x=x, y=r.carichi[nome], mode="lines+markers", line={"color": colori[k]},
                            name=f"{nome} ({_pct(r.varianza_spiegata[nome], 0)})", legendgroup=chiave,
                            legendgrouptitle_text="BTP" if col == 1 and k == 0 else ("Spread" if k == 0 else None),
                            hovertemplate="%{x}a: %{y:.1f} pb<extra>" + nome + "</extra>", row=1, col=col)
    fig.update_xaxes(type="log", tickvals=[0.5, 1, 2, 5, 10, 20, 30], title_text="Scadenza (anni)")
    fig.update_yaxes(title_text="pb per una deviazione standard mensile", row=1, col=1)
    _layout(fig, "Componenti principali delle variazioni mensili", 460)
    fig.update_layout(legend={"y": -0.28})
    return fig


def grafico_krd(a: Analisi) -> go.Figure:
    krd = a.krd
    nomi = a.tabella["Titolo"]
    fig = go.Figure([go.Bar(name=nomi[c], x=krd.columns, y=krd.loc[c],
                            marker_color=COLORI_POSIZIONI[i % len(COLORI_POSIZIONI)],
                            hovertemplate="%{x}: %{y:.1f} € per pb<extra>" + nomi[c] + "</extra>")
                     for i, c in enumerate(krd.index)])
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Nodo della curva")
    fig.update_yaxes(title="Perdita per +1 pb sul nodo (€)")
    _layout(fig, f"Key rate duration in euro: dove si concentra il rischio (totale {_eur(a.sintesi['DV01 (€)'])}"
            " per pb)", 470)
    fig.update_layout(legend={"y": -0.25})
    return fig


def grafico_stress(a: Analisi) -> go.Figure:
    s = a.stress.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=s.index, x=s["Utile/perdita"], orientation="h", name="Rivalutazione completa",
                marker_color=[ROSSO if x < 0 else ACQUA for x in s["Utile/perdita"]],
                hovertemplate="%{y}: %{x:,.0f} €<extra></extra>")
    fig.add_scatter(y=s.index, x=s["Duration e convessità"], mode="markers", name="Stima con duration e convessità",
                    marker={"symbol": "diamond", "size": 9, "color": TESTO}, hovertemplate="%{x:,.0f} €<extra></extra>")
    fig.add_scatter(y=s.index, x=s["Key rate duration"], mode="markers", name="Stima con key rate duration",
                    marker={"symbol": "line-ns-open", "size": 14, "color": GIALLO, "line": {"width": 3}},
                    hovertemplate="%{x:,.0f} €<extra></extra>")
    fig.update_xaxes(title="Utile o perdita (€)", tickformat=",.0f")
    _layout(fig, "Scenari di stress sul portafoglio", 520)
    fig.update_layout(margin={"l": 260})
    return fig


def grafico_var(a: Analisi) -> go.Figure:
    pnl = a.pnl_mensili
    v = a.var["1 mese"]
    lv = _pct(a.config.livello_var, 0)
    fig = go.Figure()
    fig.add_histogram(x=pnl, nbinsx=45, marker_color=BLU, opacity=0.85, name="Mesi della storia",
                      hovertemplate="%{x:,.0f} €: %{y} mesi<extra></extra>")
    for x, nome, c in ((-v["VaR"], f"VaR {lv}", GIALLO), (-v["CVaR"], f"CVaR {lv}", ROSSO)):
        fig.add_vline(x=x, line={"color": c, "width": 2, "dash": "dash"})
        fig.add_annotation(x=x, y=1, yref="paper", text=f"{nome}: {_eur(-x)}", showarrow=False, xanchor="right",
                           font={"color": c}, bgcolor="white")
    fig.update_xaxes(title="Utile o perdita in un mese (€)", tickformat=",.0f")
    fig.update_yaxes(title="Numero di mesi")
    return _layout(fig, f"Simulazione storica: il portafoglio di oggi in {len(pnl)} mesi dal {pnl.index[0]:%Y}", 430)


def grafico_copertura(a: Analisi) -> go.Figure:
    c = a.copertura
    fig = make_subplots(rows=2, cols=1, row_heights=[0.38, 0.62], vertical_spacing=0.12,
                        subplot_titles=("KRD in euro per nodo", "Scenari: prima e dopo la copertura"))
    fig.add_bar(x=c.krd_prima.index, y=c.krd_prima, name="Senza copertura", marker_color="#b5b3ad", row=1, col=1,
                hovertemplate="%{x}: %{y:.0f} €/pb<extra>prima</extra>")
    fig.add_bar(x=c.krd_dopo.index, y=c.krd_dopo, name="Con copertura", marker_color=BLU_SCURO, row=1, col=1,
                hovertemplate="%{x}: %{y:.0f} €/pb<extra>dopo</extra>")
    s, sc = a.stress.iloc[::-1], a.stress_coperto.iloc[::-1]
    fig.add_bar(y=s.index, x=s["Utile/perdita"], orientation="h", marker_color="#b5b3ad", showlegend=False,
                row=2, col=1, hovertemplate="%{y}: %{x:,.0f} €<extra>prima</extra>")
    fig.add_bar(y=sc.index, x=sc["Utile/perdita"], orientation="h", marker_color=BLU_SCURO, showlegend=False,
                row=2, col=1, hovertemplate="%{y}: %{x:,.0f} €<extra>dopo</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="€ per pb", row=1, col=1)
    fig.update_xaxes(tickformat=",.0f", title_text="Utile o perdita (€)", row=2, col=1)
    _layout(fig, "Copertura del rischio tasso con Bund: il rischio spread resta", 820)
    fig.update_layout(margin={"l": 260}, legend={"y": -0.08})
    return fig


def grafico_immunizzazione(a: Analisi) -> go.Figure:
    sur = a.surplus
    colori = (BLU_SCURO, ACQUA, GIALLO, ROSSO)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.1,
                        subplot_titles=("Quattro modi di investire lo stesso capitale",
                                        "Ingrandimento: la soluzione immunizzata"))
    for i, n in enumerate(sur.columns):
        fig.add_scatter(x=sur.index, y=sur[n], name=n, mode="lines", row=1, col=1,
                        line={"color": colori[i % 4], "width": 3 if i == 0 else 2},
                        hovertemplate="%{x:+} pb: %{y:,.0f} €<extra></extra>")
    fig.add_scatter(x=sur.index, y=sur.iloc[:, 0], mode="lines", showlegend=False, row=1, col=2,
                    line={"color": colori[0], "width": 3}, hovertemplate="%{x:+} pb: %{y:,.0f} €<extra></extra>")
    for col in (1, 2):
        fig.add_hline(y=0, line={"color": "#bdbcb6"}, row=1, col=col)
    fig.update_xaxes(title_text="Spostamento parallelo dei tassi (pb)")
    fig.update_yaxes(tickformat=",.0f")
    fig.update_yaxes(title_text="Surplus: titoli − valore dell'obiettivo (€)", row=1, col=1)
    ob = a.config.obiettivo
    _layout(fig, f"Immunizzazione: «{ob.descrizione}», {_eur(ob.importo)} al {ob.data:%m/%Y}", 500)
    fig.update_layout(legend={"y": -0.25})
    return fig


def _disponibili(a: Analisi) -> dict:
    g = {"curve": grafico_curve, "ricco_caro": grafico_ricco_caro, "storia": grafico_storia, "pca": grafico_pca,
         "krd": grafico_krd, "stress": grafico_stress, "var": grafico_var, "copertura": grafico_copertura}
    if a.immunizzazione is not None:
        g["immunizzazione"] = grafico_immunizzazione
    return g


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede kaleido)."""
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, f in _disponibili(a).items():
        p = cartella / f"{nome}.png"
        f(a).write_image(p, width=1000, scale=scala)
        percorsi.append(p)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _segno(x: float) -> str:
    return "pos" if x > 0.5 else "neg" if x < -0.5 else ""


def build_report(a: Analisi, output: str | Path, data_note: str = "", offline: bool = False) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    c, s = a.config, a.sintesi
    lv = _pct(c.livello_var, 0)
    config_plotly = {"displaylogo": False, "responsive": True}
    grafici = {n: f(a).to_html(full_html=False, include_plotlyjs=False, config=config_plotly)
               for n, f in _disponibili(a).items()}
    v1, v12 = a.var["1 mese"], a.var["12 mesi"]
    kpi = [
        ("Controvalore", _eur(s["Controvalore"]), f"nominale {_eur(s['Nominale'])}"),
        ("Rendimento a scadenza", _pct(s["Rendimento medio"], 2), "medio ponderato"),
        ("Duration modificata", _num(s["Duration modificata"], 1), "anni"),
        ("DV01", _eur(s["DV01 (€)"]), "perdita per +1 pb su tutte le curve"),
        ("Spread DV01", _eur(s["Spread DV01 (€)"]), "perdita per +1 pb di spread BTP"),
        (f"VaR {lv} a 1 mese", _eur(v1["VaR"]), f"CVaR {_eur(v1['CVaR'])}"),
    ]
    posizioni = [{"titolo": r["Titolo"], "nominale": _eur(r["Nominale"]), "prezzo": _num(r["Prezzo"], 3),
                  "rend": _pct(r["Rendimento"], 2), "cv": _eur(r["Controvalore"]), "peso": _pct(r["Peso"]),
                  "dur": _num(r["Duration modificata"], 2), "conv": _num(r["Convessità"], 1),
                  "dv01": _eur(r["DV01 (€)"], 1), "rc": _num(a.stime[t.emittente].ricco_caro[cod], 1)}
                 for cod, r in a.tabella.iterrows() for t in [next(x for x in a.titoli if x.codice == cod)]]
    stress = [{"nome": n, "descr": sc.descrizione, "pnl": _eur(r["Utile/perdita"]), "pct": _pct(r["Utile/perdita %"]),
               "dc": _eur(r["Duration e convessità"]), "krd": _eur(r["Key rate duration"]),
               "dopo": _eur(a.stress_coperto.loc[n, "Utile/perdita"]),
               "segno": _segno(r["Utile/perdita"]), "segno_dopo": _segno(a.stress_coperto.loc[n, "Utile/perdita"])}
              for (n, r), sc in zip(a.stress.iterrows(), a.scenari)]
    curve = []
    for em in ("BTP", "Bund"):
        st, ns = a.stime[em], a.nelson_siegel[em]
        p = st.curva
        anomali = st.tabella.index[st.tabella["Anomalo"]].tolist()
        curve.append({"em": em, "n": len(st.tabella), "rmse": _num(st.rmse_bp, 1), "rmse_ns": _num(ns.rmse_bp, 1),
                      "z2": _pct(p.zero(2.0), 2), "z10": _pct(p.zero(10.0), 2), "z30": _pct(p.zero(30.0), 2),
                      "parametri": ", ".join(f"{k} {_num(v, 4) if i < 4 else _num(v, 2)}" for i, (k, v) in enumerate(
                          zip(("β₀", "β₁", "β₂", "β₃", "τ₁", "τ₂"), p.parametri))),
                      "anomali": [f"{st.tabella.loc[x, 'Titolo']} ({_num(st.tabella.loc[x, 'Scarto (pb)'], 1)} pb)"
                                  for x in anomali]})
    spread_oggi = (a.stime["BTP"].curva.zero(10.0) - a.stime["Bund"].curva.zero(10.0)) * 1e4
    coperture = [{"titolo": p.titolo.nome, "nominale": _eur(p.nominale)} for p in a.copertura.posizioni]
    cop_dv01 = float(a.copertura.krd_dopo.sum())
    imm = None
    if a.immunizzazione is not None:
        i = a.immunizzazione
        nomi = list(a.surplus.columns)
        imm = {
            "descr": c.obiettivo.descrizione, "importo": _eur(c.obiettivo.importo),
            "data": f"{c.obiettivo.data:%d/%m/%Y}",
            "valore": _eur(i.valore_passivita), "dur_l": _num(i.duration_passivita, 2),
            "dur_p": _num(i.duration_portafoglio, 2), "conv_l": _num(i.convessita_passivita, 1),
            "conv_p": _num(i.convessita_portafoglio, 1),
            "posizioni": [{"titolo": p.titolo.nome, "nominale": _eur(p.nominale),
                           "valore": _eur(p.nominale / 100 * p.titolo.prezzo_tel_quel(a.curve["BTP"], c.valutazione))}
                          for p in i.portafoglio.posizioni],
            "colonne": nomi,
            "righe": [{"nome": n, "valori": [_eur(a.surplus_torsioni.loc[n, k]) for k in nomi],
                       "segni": [_segno(a.surplus_torsioni.loc[n, k]) for k in nomi]}
                      for n in a.surplus_torsioni.index],
            "peggiore_alt": _eur(a.surplus[nomi[1:]].min().min()), "peggiore_imm": _eur(a.surplus[nomi[0]].min()),
        }
    pca = a.pca
    env = Environment(loader=PackageLoader("yield_curve_lab", "templates"), autoescape=select_autoescape())
    html = env.get_template("report.html.j2").render(
        cliente=c.cliente, valutazione=f"{c.valutazione:%d/%m/%Y}", oggi=date.today().strftime("%d/%m/%Y"),
        kpi=kpi, posizioni=posizioni, stress=stress, curve=curve, grafici=grafici, nota=data_note, lv=lv,
        spread_oggi=_num(spread_oggi, 0), var1=v1, var12=v12, eur=_eur, num=_num, pct=_pct,
        peggiore_data=f"{v1['Data peggiore']:%m/%Y}", n_mesi=v1["Osservazioni"],
        pca_btp=[_pct(x, 0) for x in pca["BTP"].varianza_spiegata], pca_spread=_pct(pca["Spread"].varianza_spiegata
                                                                                    .iloc[0], 0),
        pca_bund=[_pct(x, 0) for x in pca["Bund"].varianza_spiegata],
        quota_spread=_pct(s["Spread DV01 (€)"] / s["DV01 (€)"], 0), coperture=coperture,
        cop_dv01=_eur(cop_dv01, 0), cop_nominale=_eur(-sum(p.nominale for p in a.copertura.posizioni)),
        var_coperto=_eur(a.var_coperto["VaR"]), cvar_coperto=_eur(a.var_coperto["CVaR"]),
        crisi_peggio=bool("Crisi del debito 2011" in a.stress.index and a.stress_coperto.loc[
            "Crisi del debito 2011", "Utile/perdita"] < a.stress.loc["Crisi del debito 2011", "Utile/perdita"]),
        imm=imm, peggiore_periodo=a.stress.index[-1].split("(")[-1].rstrip(")"),
        plotly_inline=get_plotlyjs() if offline else None, plotly_cdn=PLOTLY_CDN,
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
