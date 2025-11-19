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


def _iter_metric_rows(csv_path: Path):
    """Itera TUTTE le righe di un CSV metriche come dict (pandas se presente, altrimenti csv.DictReader)."""
    # pandas
    if pd is not None:
        try:
            df = pd.read_csv(csv_path)
            # normalizza NaN -> None
            for _, r in (df if df is not None else []).iterrows():
                d = {}
                for k, v in r.to_dict().items():
                    try:
                        # mantieni stringhe così come sono; NaN -> None
                        if v is None or (hasattr(v, "is_nan") and v.is_nan()) or (isinstance(v, float) and np.isnan(v)):
                            d[k] = None
                        else:
                            d[k] = v
                    except Exception:
                        d[k] = v
                yield d
            return
        except Exception:
            pass
    # csv stdlib
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                yield r
    except Exception:
        return

def plot_q4_runtime_ecdf(metrics_source, out_png, include="all", cap_sec=None, logx=False):
    """
    Q4 – ECDF dei tempi di esecuzione (runtime) per algoritmo/formulazione.

    Robusta:
      - scansiona: run_metrics_*.csv, metrics*.csv, *summary*.csv, *.metrics.csv (ricorsivo)
      - legge TUTTE le righe
      - accetta alias per 'runtime' (runtime, runtime_total, time, elapsed, wall_time, solver_runtime, model_runtime,
        e perfino 'Runtime (s)')
      - scrive un file di debug <out_png>.debug.txt se non trova dati
    """
    from itertools import cycle
    base = Path(metrics_source)
    groups = defaultdict(list)

    # 1) trova CSV candidati
    patterns = ["run_metrics_*.csv", "metrics*.csv", "*summary*.csv", "*.metrics.csv"]
    cand = []
    for pat in patterns:
        cand.extend(base.rglob(pat))
    # dedup mantendendo l'ordine
    seen = set()
    csv_files = []
    for p in cand:
        if p.suffix.lower() == ".csv" and p not in seen:
            seen.add(p)
            csv_files.append(p)

    # 2) raccogli runtime da tutte le righe
    aliases = {
        "runtime", "runtime_total", "time", "elapsed", "wall_time",
        "solver_runtime", "model_runtime", "runtime(s)", "runtime_sec", "seconds"
    }

    scanned = []  # per debug

    for csv_path in csv_files:
        # leggi tutte le righe
        for row in _iter_metric_rows(csv_path):
            # normalizza chiavi
            norm = {}
            for k, v in (row or {}).items():
                kk = (k or "").strip().lower()
                kk = kk.replace(" ", "").replace("\t", "").replace("(s)", "s")
                norm[kk] = v

            # gruppo: preferisci 'formulation'/'tag', altrimenti nome cartella
            g = (row.get("formulation") or row.get("tag") or csv_path.parent.name)
            g = str(g)

            # runtime: usa alias
            rt = None
            for key in aliases:
                if key in norm:
                    rt = _as_float(norm.get(key))
                    if rt is not None:
                        break

            # se non trovato, prova proprio 'runtime' non normalizzato
            if rt is None:
                rt = _as_float(row.get("runtime"))

            # cap al time-limit (se dato)
            if rt is not None and cap_sec is not None:
                try:
                    rt = min(rt, float(cap_sec))
                except Exception:
                    pass

            if rt is not None:
                groups[g].append(rt)

        scanned.append((csv_path, list((row or {}).keys())))

    # 3) filtro include
    pli_set = {"y", "prefix"}
    if include == "pli":
        groups = {g: v for g, v in groups.items() if g in pli_set}
    elif isinstance(include, (set, list, tuple)):
        groups = {g: v for g, v in groups.items() if g in include}

    # 4) plot (o debug)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    if not groups:
        # scrivi un file di debug accanto al PNG
        dbg = Path(out_png).with_suffix(".debug.txt")
        try:
            with open(dbg, "w", encoding="utf-8") as f:
                f.write(f"Q4 DEBUG – Nessun runtime trovato in: {base}\n")
                f.write("CSV ispezionati e intestazioni viste:\n")
                for p, cols in scanned:
                    f.write(f"- {p} -> {cols}\n")
        except Exception:
            pass

        plt.figure(figsize=(9, 4))
        plt.text(0.5, 0.6, "Nessun runtime disponibile per Q4", ha="center", va="center", fontsize=14)
        plt.text(0.5, 0.35, f"Vedi file di debug:\n{dbg.name}", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    _palette = {
        "prefix": "tab:blue",
        "y": "tab:orange",
        "heuristic": "tab:green",
        "heuristic+1move": "tab:purple",
        "heuristic+vnd": "tab:red",
        "heuristic+1move+vnd": "tab:brown",
    }
    _markers = cycle(["o", "s", "D", "^", "v", "P", "X"])

    order = sorted(groups.keys(), key=lambda k: np.median(groups[k]))
    x_all = []

    plt.figure(figsize=(12, 6))
    for g in order:
        vals = groups[g]
        x = np.sort(np.asarray(vals, dtype=float))
        x_all.extend(x.tolist())
        color = _palette.get(g, None)
        marker = next(_markers)

        if len(x) == 1:
            plt.vlines(x[0], 0.0, 1.0, linestyles="-", colors=color, label=g, linewidth=2)
            plt.scatter([x[0]], [1.0], s=35, color=color, edgecolors="white",
                        linewidths=0.7, marker=marker, zorder=3)
        else:
            y = np.arange(1, len(x) + 1) / len(x)
            xs = np.r_[x[0], x]
            ys = np.r_[0.0, y]
            plt.step(xs, ys, where="post", label=g, color=color, linewidth=2)
            plt.scatter(x, y, s=25, color=color, edgecolors="white",
                        linewidths=0.5, marker=marker, zorder=3)

    if logx:
        plt.xscale("log")

    if x_all:
        xmin, xmax = float(min(x_all)), float(max(x_all))
        pad = max(1e-3, 0.01 * (xmax - xmin))
        plt.xlim(max(1e-6, xmin - pad), xmax + pad)

    plt.ylim(0.0, 1.0)
    plt.xlabel("Tempo di esecuzione (s)")
    plt.ylabel("Quota con runtime ≤ t")
    plt.title("Q4 – ECDF dei runtime per algoritmo")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# === sostituisci TUTTO il corpo di plot_q5_apx_vs_runtime con questo ===
def plot_q5_apx_vs_runtime(metrics_source,
                           out_png,
                           include="all",
                           cap_sec=None,
                           logx=False,
                           annotate=False):
    import os, glob, re, csv, math
    import matplotlib.pyplot as plt

    def _list_csvs(root):
        pats = [
            os.path.join(str(root), "run_metrics_*.csv"),
            os.path.join(str(root), "*", "run_metrics_*.csv"),
        ]
        files = []
        for p in pats:
            files += glob.glob(p)
        return sorted(set(files))

    def _get(row, *names):
        # lookup case-insensitive su più alias
        low = {k.lower(): k for k in row.keys()}
        for nm in names:
            k = low.get(nm.lower())
            if k is not None:
                return row.get(k)
        return None

    def _to_float(x):
        try:
            if x is None or str(x).strip() == "":
                return float("nan")
            return float(str(x).replace(",", "."))
        except Exception:
            return float("nan")

    files = _list_csvs(metrics_source)
    print(f"[Q5] metrics_source={metrics_source}  trovati {len(files)} file:")
    for f in files:
        try:
            sz = os.path.getsize(f)
        except Exception:
            sz = -1
        print(f"[Q5]   - {f}  size={sz}")

    points = []  # (algo, instance, runtime, apx)

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                rows = list(rd)
        except Exception as e:
            print(f"[Q5] skip {fp}: {e}")
            continue

        # algoritmo dal CSV o dal filename
        m = re.search(r"run_metrics_([^/\\]+)\.csv$", fp)
        algo_from_file = m.group(1) if m else "unknown"

        for r in rows:
            algo = (_get(r, "formulation") or algo_from_file).strip()

            instance = _get(r, "instance_name", "instance")
            instance = str(instance).strip() if instance is not None else "?"

            # runtime (vari alias)
            runtime = None
            for nm in ("runtime", "runtime_sec", "time", "secs", "seconds"):
                v = _get(r, nm)
                if v is not None:
                    runtime = _to_float(v)
                    break
            if runtime is None:
                runtime = float("nan")
            if cap_sec is not None and runtime == runtime:  # not NaN
                runtime = min(runtime, float(cap_sec))

            # APX (apx|apx_bound|apx_lb1) oppure ricalcolo C/LB
            apx = None
            for nm in ("apx", "apx_bound", "apx_lb1"):
                v = _get(r, nm)
                if v is not None:
                    apx = _to_float(v)
                    break
            if apx is None or not (apx == apx):  # NaN
                C = _to_float(_get(r, "obj_C", "C", "obj_c"))
                best_bound = _to_float(_get(r, "best_bound"))
                LB1 = _to_float(_get(r, "LB1", "lb1"))
                denom = None
                if best_bound == best_bound and best_bound > 0:
                    denom = best_bound
                elif LB1 == LB1 and LB1 > 0:
                    denom = LB1
                apx = (C / denom) if (denom and C == C) else float("nan")

            points.append((algo, instance, runtime, apx))

    # filtro include
    keep_sets = {
        "pli": {"y", "prefix"},
        "heur": {"heuristic", "heuristic+1move", "heuristic+vnd", "heuristic+1move+vnd"},
        "all": None,
    }
    keep = keep_sets.get(include, None)
    if keep is not None:
        points = [p for p in points if p[0] in keep]

    # pulizia e validazione
    pts = [(a, i, rt, ax) for (a, i, rt, ax) in points
           if (rt == rt and rt > 0) and (ax == ax and ax >= 1.0)]

    print(f"[Q5] righe lette={len(points)}, righe valide={len(pts)}, punti={len(pts)}")

    if not pts:
        _render_no_data(out_png, "Nessun dato disponibile per Q5")
        return

    # rimuovi duplicati identici
    dedup = {}
    for a, i, rt, ax in pts:
        dedup[(a, i, rt, ax)] = (a, i, rt, ax)
    pts = list(dedup.values())

    # scatter
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5), dpi=150)

    # raggruppa per algoritmo
    by_algo = {}
    for a, i, rt, ax in pts:
        by_algo.setdefault(a, []).append((rt, ax, i))

    for a, lst in by_algo.items():
        xs = [x for (x, _, __) in lst]
        ys = [y for (_, y, __) in lst]
        plt.scatter(xs, ys, label=a, alpha=0.9)
        if annotate:
            for x, y, inst in lst:
                plt.annotate(inst, (x, y), fontsize=7, alpha=.75)

    plt.axhline(1.0, ls="--", lw=1)
    plt.xlabel("Runtime (s)" + (f" (cap={cap_sec}s)" if cap_sec else ""))
    plt.ylabel("Approssimazione (C / LB)")
    if logx:
        plt.xscale("log")
    plt.title("Q5 – APX vs Runtime")
    plt.legend(title="Algoritmo")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# se non c'è già nel file, aggiungi/lascia questa helper:
def _render_no_data(out_png, msg):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4), dpi=150)
    plt.text(0.5, 0.5, msg, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
