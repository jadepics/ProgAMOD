from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict  # ⬅️ AGGIUNTO

# pandas opzionale (se non c'è, usiamo csv)
try:
    import pandas as pd  # noqa
except Exception:
    pd = None

def plot_q1_apx_box(metrics_source, out_png: str):
    """
    Q1 – Boxplot qualità (APX = C / LB) aggregando CSV di metriche.
    Robusto: accetta run_metrics_*.csv, metrics*.csv, *summary*.csv, qualsiasi CSV che contenga
    almeno 'obj_C' e ('bound' o 'LB1') o direttamente 'apx'.
    """
    import csv
    import math
    from pathlib import Path
    import statistics as stats
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import re

    base = Path(metrics_source)

    # 🔍 Cerca più pattern, ricorsivamente
    patterns = ["run_metrics_*.csv", "metrics*.csv", "*summary*.csv", "*.metrics.csv", "*.csv"]
    cand = []
    for pat in patterns:
        cand.extend(base.rglob(pat))
    # rimuovi duplicati mantenendo l'ordine
    seen = set()
    csv_files = []
    for p in cand:
        if p.suffix.lower() == ".csv" and p not in seen:
            seen.add(p)
            csv_files.append(p)

    if not csv_files:
        raise RuntimeError(f"Nessun CSV di metriche trovato in {base}")

    def _to_float(v):
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def _guess_label(row_dict, filepath: Path):
        # 1) campo esplicito
        for k in ("algorithm", "formulation", "tag", "algo"):
            if row_dict.get(k):
                return str(row_dict[k])
        # 2) dal filename tipo run_metrics_<tag>.csv
        m = re.search(r"run_metrics_([A-Za-z0-9+\-_.]+)\.csv$", filepath.name)
        if m:
            return m.group(1)
        # 3) dal path: cartelle note
        parts = [x.lower() for x in filepath.parts]
        for candid in ("y", "prefix", "heuristic", "heuristic+vnd", "heuristic+1move", "heuristic+1move+vnd"):
            if candid in parts:
                return candid
        # fallback: nome file senza estensione
        return filepath.stem

    series = defaultdict(list)
    used_files = 0

    for f in csv_files:
        try:
            with open(f, "r", newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                has_any = False
                for r in rd:
                    has_any = True
                    label = _guess_label(r, f)

                    apx = _to_float(r.get("apx"))
                    if apx is None:
                        C = _to_float(r.get("obj_C")) or _to_float(r.get("C"))
                        bound = _to_float(r.get("bound")) or _to_float(r.get("best_bound"))
                        LB1 = _to_float(r.get("LB1"))
                        # priorità: bound, poi LB1
                        if apx is None and C is not None and bound is not None and bound > 0:
                            apx = C / bound
                        if apx is None and C is not None and LB1 is not None and LB1 > 0:
                            apx = C / LB1

                    if apx is not None and math.isfinite(apx):
                        series[label].append(apx)

                if has_any:
                    used_files += 1
        except Exception:
            # file non leggibile; ignora e continua
            continue

    if not series:
        raise RuntimeError(
            "Nessun dato valido per Q1. Verifica che i CSV contengano 'obj_C' e "
            "'bound' o 'LB1' oppure un campo 'apx'."
        )

    # Ordina le etichette per mediana (più "basso è meglio")
    order = sorted(series.keys(), key=lambda k: stats.median(series[k]))
    data = [series[k] for k in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, showmeans=True, meanline=True, widths=0.5, patch_artist=False)
    plt.title("Q1 – Qualità (APX = C / LB)")
    plt.xlabel("Algoritmo / Formulazione")
    plt.ylabel("APX (minore è meglio)")
    plt.xticks(range(1, len(order) + 1), order, rotation=0)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# --- Q2: ECDF dell'APX per algoritmo -----------------------------------------

def _read_first_row(csv_path: Path):
    """Legge la prima riga di run_metrics_*.csv come dict.
       Usa pandas se disponibile, altrimenti csv.DictReader."""
    row = None
    if pd is not None:
        try:
            df = pd.read_csv(csv_path)
            if df is not None and len(df) > 0:
                row = df.iloc[0].to_dict()
        except Exception:
            row = None
    if row is None:
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rd = csv.DictReader(f)
                row = next(rd, None)
        except Exception:
            row = None
    return row or {}

def _as_float(x):
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def plot_q2_apx_ecdf(metrics_source, out_png, include="all"):
    """
    ECDF dell'APX raggruppata per 'formulation' (y, prefix, heuristic, heuristic+vnd, ...)

    include:
      - "all"  -> PLI + euristiche
      - "pli"  -> solo y, prefix
      - iterable esplicito, es: {"heuristic","heuristic+vnd"}
    """
    src = Path(metrics_source)
    groups = defaultdict(list)

    for csv_file in src.rglob("run_metrics_*.csv"):
        row = _read_first_row(csv_file)
        if not row:
            continue

        # nome del gruppo
        g = str(row.get("formulation") or row.get("tag") or csv_file.parent.name)

        # APX: preferisci 'apx', poi 'apx_bound', poi 'apx_lb1'
        apx = _as_float(row.get("apx"))
        if apx is None:
            apx = _as_float(row.get("apx_bound"))
        if apx is None:
            apx = _as_float(row.get("apx_lb1"))
        if apx is None:
            continue

        groups[g].append(apx)

    # filtro include
    pli_set = {"y", "prefix"}
    if include == "pli":
        groups = {g: v for g, v in groups.items() if g in pli_set}
    elif isinstance(include, (set, list, tuple)):
        groups = {g: v for g, v in groups.items() if g in include}
    # else: "all" -> nessun filtro

    plt.figure(figsize=(12, 6))
    if not groups:
        plt.text(0.5, 0.5, "Nessun dato trovato per Q2", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    from itertools import cycle  # palette marker

    _palette = {
        "prefix": "tab:blue",
        "y": "tab:orange",
        "heuristic": "tab:green",
        "heuristic+1move": "tab:purple",
        "heuristic+vnd": "tab:red",
        "heuristic+1move+vnd": "tab:brown",
    }
    _markers = cycle(["o", "s", "D", "^", "v", "P", "X"])

    # disegna ECDF per ogni gruppo, gestendo il caso "un solo punto"
    x_all = []
    for g, vals in sorted(groups.items()):
        x = np.sort(np.asarray(vals, dtype=float))
        x_all.extend(x.tolist())

        color = _palette.get(g)
        line_kwargs = {"color": color, "linewidth": 2} if color is not None else {"linewidth": 2}
        sc_kwargs = {"color": color, "edgecolors": "white", "linewidths": 0.7} if color is not None else {
            "edgecolors": "white", "linewidths": 0.7}
        marker = next(_markers)

        if len(x) == 1:
            # linea verticale + marker con colore coerente
            plt.vlines(x[0], 0.0, 1.0, linestyles="-", label=g, **line_kwargs)
            plt.scatter([x[0]], [1.0], s=35, marker=marker, zorder=3, **sc_kwargs)
        else:
            y = np.arange(1, len(x) + 1) / len(x)
            xs = np.r_[x[0], x]  # parte da 0 per mostrare il salto
            ys = np.r_[0.0, y]
            plt.step(xs, ys, where="post", label=g, **line_kwargs)
            plt.scatter(x, y, s=25, marker=marker, zorder=3, **sc_kwargs)

    # assi chiari
    if x_all:
        xmin, xmax = float(min(x_all)), float(max(x_all))
        pad = max(1e-4, 0.01 * (xmax - xmin))
        plt.xlim(xmin - pad, xmax + pad)

    plt.ylim(0.0, 1.0)
    plt.xlabel("APX (minore è meglio)")
    plt.ylabel("F(APX ≤ x)")
    plt.title("Q2 – ECDF dell'APX per algoritmo")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend(loc="lower right")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def plot_q3_success_rate(metrics_source, out_png, tau=1.01, include="all"):
    """
    Q3 – Success rate: quota di run con APX ≤ tau per ciascun algoritmo/gruppo.
    - tau: soglia (default 1.01)
    - include: "all" (default) / "pli" / insieme esplicito di etichette
    """
    from collections import defaultdict
    src = Path(metrics_source)

    counts = defaultdict(int)
    good   = defaultdict(int)

    for csv_file in src.rglob("run_metrics_*.csv"):
        row = _read_first_row(csv_file)
        if not row:
            continue

        g = str(row.get("formulation") or row.get("tag") or csv_file.parent.name)

        apx = _as_float(row.get("apx"))
        if apx is None:
            apx = _as_float(row.get("apx_bound"))
        if apx is None:
            apx = _as_float(row.get("apx_lb1"))
        if apx is None:
            continue

        counts[g] += 1
        if apx <= float(tau):
            good[g] += 1

    # filtro include
    pli_set = {"y", "prefix"}
    if include == "pli":
        keys = [g for g in counts if g in pli_set]
    elif isinstance(include, (set, list, tuple)):
        keys = [g for g in counts if g in include]
    else:
        keys = list(counts.keys())

    keys = sorted(keys)
    if not keys:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "Nessun dato trovato per Q3", ha="center", va="center")
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    rates = [(good[k] / counts[k]) if counts[k] else 0.0 for k in keys]

    # palette coerente con Q2
    _palette = {
        "prefix": "tab:blue",
        "y": "tab:orange",
        "heuristic": "tab:green",
        "heuristic+1move": "tab:purple",
        "heuristic+vnd": "tab:red",
        "heuristic+1move+vnd": "tab:brown",
    }
    colors = [_palette.get(k, None) for k in keys]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(keys, rates, color=colors, edgecolor="black", linewidth=0.6)

    # etichette “g/b (n)” sopra ogni barra
    for i, b in enumerate(bars):
        txt = f"{good[keys[i]]}/{counts[keys[i]]} (n)"
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, txt,
                 ha="center", va="bottom", fontsize=9)

    plt.ylim(0.0, 1.05)
    plt.ylabel(f"Quota con APX ≤ {tau}")
    plt.xlabel("Algoritmo / Formulazione")
    plt.title("Q3 – Success rate entro soglia APX")
    plt.grid(True, ls=":", alpha=0.4, axis="y")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

