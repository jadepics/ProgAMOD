# salbp/plots_heuristic.py
from __future__ import annotations
import os, csv
import matplotlib.pyplot as plt

def _read_trace(trace_csv_path: str):
    rows = []
    try:
        import pandas as pd
        df = pd.read_csv(trace_csv_path)
        for _, r in df.iterrows():
            rows.append({
                "step": int(r.get("step", len(rows))),
                "t": float(r.get("t", 0.0)),
                "C": float(r.get("C", float("nan"))),
            })
    except Exception:
        with open(trace_csv_path, "r", newline="", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for r in rr:
                try:
                    rows.append({
                        "step": int(r.get("step", len(rows))),
                        "t": float(r.get("t", 0.0)),
                        "C": float(r.get("C", "")),
                    })
                except Exception:
                    pass
    # tieni solo righe valide con C numerico
    rows = [r for r in rows if r.get("C") == r.get("C")]  # na check
    # ordina per step crescente
    rows.sort(key=lambda x: (x["step"], x["t"]))
    return rows

def plot_h0_progress(trace_csv_path: str, out_png: str, by: str = "step", title: str | None = None):
    """
    H0 – Progress C vs step/time per euristiche iterative (VND, 1-move…).
    - by="step": asse x = step di miglioramento (0,1,2,…); curva a tacche (plt.step).
    - by="time": asse x = tempo (s); curva continua (plt.plot).
    Richiede un CSV con colonne: step, t, C (come prodotto dal logger VND).
    """
    rows = _read_trace(trace_csv_path)

    if not rows:
        plt.figure(figsize=(7,4))
        plt.text(0.5, 0.5, "Trace vuota o assente", ha="center", va="center")
        plt.axis("off")
        if title: plt.title(title)
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        return

    if by == "time":
        xs = [r["t"] for r in rows]
        xlabel = "Tempo (s)"
        draw_step = False
    else:
        xs = [r["step"] for r in rows]
        xlabel = "Passi di miglioramento (step)"
        draw_step = True

    ys = [r["C"] for r in rows]

    plt.figure(figsize=(8,5))
    if draw_step and len(rows) > 1:
        plt.step(xs, ys, where="post", linewidth=2)
    else:
        plt.plot(xs, ys, linewidth=2, marker="o", markersize=3)

    plt.gca().invert_yaxis()  # C più basso è meglio
    plt.xlabel(xlabel)
    plt.ylabel("C (makespan)")
    if title: plt.title(title)
    else: plt.title("Progresso euristica (C)")

    # annotazioni “start” e “final”
    try:
        plt.scatter([xs[0]], [ys[0]], s=28)
        plt.annotate("start", (xs[0], ys[0]), xytext=(5,5), textcoords="offset points")
        plt.scatter([xs[-1]], [ys[-1]], s=28)
        plt.annotate("final", (xs[-1], ys[-1]), xytext=(5,-10), textcoords="offset points")
    except Exception:
        pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

# --- H3: contributo delle mosse per metrica (C / range / var) ---
from pathlib import Path
import math

def plot_h3_move_contrib(trace_src, out_png, metric="C", title=None):
    """
    Legge una trace (CSV o lista di dict) e somma il 'contributo' dei miglioramenti
    per tipo di mossa: 1move / swap / eject (ignora 'init'/'final').
    metric: 'C' (scende), 'range' (scende), 'var' (scende).

    Output: bar chart orizzontale con una barra per fase presente.
    """
    # --- 1) carica/normalizza trace ---
    rows = []
    # path/str -> CSV
    if isinstance(trace_src, (str, Path)):
        path = str(trace_src)
        try:
            import pandas as _pd
            df = _pd.read_csv(path)
            rows = df.to_dict(orient="records")
        except Exception:
            import csv
            with open(path, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                rows = list(r)
    # lista di dict già pronta
    elif isinstance(trace_src, list):
        rows = list(trace_src)
    else:
        raise ValueError("trace_src deve essere path CSV o lista di dizionari")

    if not rows:
        # fallback: grafico vuoto “nessun miglioramento”
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(["(nessun dato)"], [0.0])
        ax.set_xlabel(f"Contributo {metric}↓")
        ax.set_title(title or "Contributo delle mosse")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    # normalizza tipi/chiavi
    norm = []
    for r in rows:
        rr = dict(r)
        # chiavi canoniche
        rr["phase"] = (rr.get("phase") or "1move").strip().lower()
        # tempo e step (se mancano, li stimiamo)
        rr["t"] = float(rr.get("t", rr.get("time", 0.0) or 0.0))
        # step può essere assente nelle trace LS -> costruiscilo con l'indice
        if "step" in rr:
            try:
                rr["step"] = int(rr["step"])
            except Exception:
                rr["step"] = 0
        norm.append(rr)

    # ordina cronologicamente (per tempo se disponibile, altrimenti per indice)
    if all(("t" in r for r in norm)):
        norm.sort(key=lambda x: x["t"])
    else:
        # fallback: lascia l’ordine dato
        pass

    # se la metrica non c'è (LS vecchie), prova a ricavarla
    def _num(v, default=math.nan):
        try:
            return float(v)
        except Exception:
            return default

    # --- 2) somma contributi per fase ---
    from collections import defaultdict
    contrib = defaultdict(float)
    counts   = defaultdict(int)

    # scegli colonna da usare
    metric = metric.lower()
    col = {"c": "C", "range": "range", "var": "var"}.get(metric, "C")

    # inizializza prev ai primi valori disponibili
    def _get_value(row, key):
        if key in ("C", "c"):
            return _num(row.get("C", row.get("c", math.nan)))
        return _num(row.get(key, math.nan))

    prev = norm[0]
    prev_val = _get_value(prev, col)

    for r in norm[1:]:
        phase = r.get("phase", "1move")
        if phase in ("init", "final") or phase is None:
            prev = r
            prev_val = _get_value(prev, col)
            continue

        cur_val = _get_value(r, col)

        # contributo = miglioramento (positivizzato): prev - cur  (per range/var idem)
        if not math.isnan(prev_val) and not math.isnan(cur_val):
            delta = prev_val - cur_val
            if delta > 0:
                contrib[phase] += float(delta)
                counts[phase]  += 1

        prev = r
        prev_val = cur_val

    # --- 3) disegna il bar chart ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not contrib:
        labels = ["(nessun miglioramento)"]
        vals   = [0.0]
    else:
        # ordina per contributo decrescente
        items = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        labels = [f"{ph} (n={counts[ph]})" for ph, _ in items]
        vals   = [v for _, v in items]

    ax.barh(labels, vals)
    ax.invert_yaxis()  # prima barra in alto
    met_name = {"C": "C (↓)", "range": "range (↓)", "var": "var (↓)"}[col]
    ax.set_xlabel(f"Contributo totale su {met_name}")
    ax.set_title(title or "Contributo delle mosse")

    # etichette numeriche a destra
    for y, v in enumerate(vals):
        ax.text(v, y, f"  {v:.3g}", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_h4_timeline(csv_path: str,
                     out_png: str,
                     metric: str = "C",
                     by: str = "time",
                     title: str | None = None) -> None:
    """
    H4 – Timeline: linea della metrica scelta (C/range/var) vs step/tempo
    con marker delle mosse (1move/swap/eject). Funziona sia su trace VND sia su 1-move.

    csv atteso: colonne step,t,C,dC,phase,move,range,var (come vnd_trace.csv o ls_trace.csv)
    - metric: "C" | "range" | "var"
    - by: "time" (usa colonna 't') oppure "step" (usa colonna 'step')
    """
    import csv as _csv
    import math as _math
    import matplotlib.pyplot as _plt

    # 1) leggi CSV (robusto senza pandas)
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rd = _csv.DictReader(f)
            for r in rd:
                rows.append(r)
    except Exception as e:
        print(f"[H4] impossibile leggere {csv_path}: {e}")
        rows = []

    if not rows:
        # crea un grafico vuoto "parlante"
        _plt.figure(figsize=(8, 4.5))
        _plt.title(title or f"H4 – Timeline (no data)")
        _plt.text(0.5, 0.5, "Nessuna trace disponibile", ha="center", va="center", transform=_plt.gca().transAxes)
        _plt.tight_layout()
        _plt.savefig(out_png, dpi=150)
        _plt.close()
        return

    xcol = "t" if by == "time" else "step"
    # 2) estrai serie
    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    X = [_to_float(r.get(xcol, 0.0)) for r in rows]
    Y = [_to_float(r.get(metric, _math.nan)) for r in rows]
    phases = [str(r.get("phase", "")) for r in rows]
    dC = [_to_float(r.get("dC", 0.0)) for r in rows]

    # 3) plot
    _plt.figure(figsize=(8, 4.5))
    # linea metrica
    if len(X) >= 1:
        _plt.plot(X, Y, linewidth=1.5, label=metric)

    # marker per mosse (niente colori “forzati”: lasciamo default)
    markers = {"1move": "o", "swap": "s", "eject": "^"}
    for ph in ("1move", "swap", "eject"):
        idx = [i for i, p in enumerate(phases) if p == ph]
        if not idx:
            continue
        xs = [X[i] for i in idx]
        ys = [Y[i] for i in idx]
        _plt.scatter(xs, ys, marker=markers.get(ph, "o"), s=28, label=ph)

    # evidenzia i punti dove C migliora (se stiamo plottando C)
    if metric.lower() == "c":
        imp = [i for i, val in enumerate(dC) if val > 0]
        if imp:
            _plt.scatter([X[i] for i in imp], [Y[i] for i in imp],
                         s=60, facecolors="none", edgecolors="black", linewidths=1.2,
                         label="improvement (ΔC>0)")

    _plt.xlabel("time (s)" if by == "time" else "step")
    _plt.ylabel(metric)
    if title:
        _plt.title(title)
    _plt.legend(loc="best")
    _plt.tight_layout()
    _plt.savefig(out_png, dpi=150)
    _plt.close()
