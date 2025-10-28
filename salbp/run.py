from __future__ import annotations
import argparse, os
from pathlib import Path
import argparse
import time
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

from salbp.instance import Instance
from salbp.model import SALBPMinMaxModel
from salbp.metrics import balance_metrics

#si chiama monitor per gestire la costruzione dei grafici
try:
    from salbp.monitor import ProgressLogger, plot_progress, plot_gap, plot_station_loads
    HAVE_MONITOR = True
except Exception:
    HAVE_MONITOR = False
    ProgressLogger = None


def plot_station_loads_fallback(loads_dict, outpath):
    """
    Fallback minimale per generare station_loads.png anche se la funzione
    plot_station_loads non è disponibile dal modulo monitor.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib non disponibile: salto il plot ({e})")
        return

    stations = sorted(loads_dict.keys())
    loads = [loads_dict[s] for s in stations]

    plt.figure(figsize=(8, 4))
    plt.bar([str(s) for s in stations], loads)
    plt.xlabel("Station")
    plt.ylabel("Load")
    plt.title("Station Loads")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()



def plot_progress(times, best_series, bound_series, final_best,
                  out_incumbent_bound, out_gap, scale_mode: str = "raw"):
    """
    Grafici di progresso per MIP in *unità reali* (default "raw") oppure "normalized".
    - Ordina e deduplica per timestamp.
    - Traccia *a gradini* (step) con marker sui punti veri.
    - In "raw": aggiunge una linea orizzontale a y = final_best.
    - Gap in percentuale su asse log.
    """
    try:
        import matplotlib.pyplot as plt
        import math
        import numpy as np
    except Exception as e:
        print(f"[INFO] matplotlib non disponibile: salto progress ({e})")
        return

    # Dati minimi se manca tutto
    if not times or len(times) != len(best_series) or len(times) != len(bound_series):
        times = [0.0]
        best_series = [final_best]
        bound_series = [final_best]

    # --- pulizia + ordinamento ---
    def fnum(x):
        try:
            x = float(x)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    recs = []
    for t, b, bd in zip(times, best_series, bound_series):
        t = fnum(t)
        if t is None:
            continue
        recs.append((t, fnum(b), fnum(bd)))
    if not recs:
        recs = [(0.0, fnum(final_best), fnum(final_best))]
    recs.sort(key=lambda r: r[0])

    # --- deduplica timestamp quasi uguali ---
    eps = 1e-9
    cleaned = []
    for t, b, bd in recs:
        if cleaned and abs(t - cleaned[-1][0]) < eps:
            lt, lb, lbd = cleaned[-1]
            cleaned[-1] = (t, b if b is not None else lb, bd if bd is not None else lbd)
        else:
            cleaned.append((t, b, bd))

    t = [r[0] for r in cleaned]
    b = [r[1] for r in cleaned]
    bd = [r[2] for r in cleaned]

    # --- scaling ---
    if scale_mode == "normalized":
        scale = None
        if isinstance(final_best, (int, float)) and final_best not in (None, 0):
            scale = float(final_best)
        if scale is None:
            # fallback: ultimo valor finito
            for v in reversed(b):
                if v is not None:
                    scale = float(v); break
        if scale is None:
            for v in reversed(bd):
                if v is not None:
                    scale = float(v); break
        if scale is None:
            scale = 1.0
        y_best  = [ (vv/scale) if vv is not None else np.nan for vv in b  ]
        y_bound = [ (vv/scale) if vv is not None else np.nan for vv in bd ]
        ref_y   = 1.0
        y_label = "Valore normalizzato"
        title   = "Evoluzione incumbent vs bound (normalizzato)"
    else:
        # RAW: usa direttamente i numeri del callback + snapshot finale
        y_best  = [ vv if vv is not None else np.nan for vv in b  ]
        y_bound = [ vv if vv is not None else np.nan for vv in bd ]
        ref_y   = final_best if isinstance(final_best, (int, float)) else None
        y_label = "Valore obiettivo"
        title   = "Evoluzione incumbent vs bound"

    # Utility: trasformazione in "step post"
    def to_step(tt, yy):
        xs, ys = [], []
        if not tt:
            return xs, ys
        xs.append(tt[0]); ys.append(yy[0])
        for i in range(1, len(tt)):
            xs.append(tt[i]); ys.append(yy[i-1])  # orizzontale
            xs.append(tt[i]); ys.append(yy[i])    # salto
        return xs, ys

    t_step, yb_step  = to_step(t, y_best)
    _,      ybd_step = to_step(t, y_bound)

    # --- Incumbent & Bound ---
    plt.figure(figsize=(10.5, 5))
    plt.step(t_step, yb_step,  where="post", label="Incumbent")
    if np.isfinite(y_bound).any():
        plt.step(t_step, ybd_step, where="post", label="Best bound")
    plt.scatter(t, y_best,  s=22)
    if np.isfinite(y_bound).any():
        plt.scatter(t, y_bound, s=22)

    if ref_y is not None and math.isfinite(ref_y):
        plt.axhline(ref_y, linestyle="--", linewidth=1, alpha=0.8, label=("C*" if scale_mode=="raw" else "1.0"))

    plt.xlabel("Tempo (s)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_incumbent_bound, dpi=150)
    plt.close()

    # --- Gap % (log) ---
    gaps = []
    for iv, jv in zip(b, bd):
        if iv is None or jv is None or abs(iv) < 1e-12:
            gaps.append(np.nan)
        else:
            gaps.append(abs(iv - jv) / (abs(iv) + 1e-9) * 100.0)

    gx, gy = to_step(t, gaps)
    plt.figure(figsize=(10.5, 5))
    plt.step(gx, gy, where="post")
    plt.yscale("log")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Gap relativo (%)")
    plt.title("Gap relativo vs tempo (scala log)")
    plt.tight_layout()
    plt.savefig(out_gap, dpi=150)
    plt.close()


def main():
    import argparse
    import time
    from pathlib import Path

    ap = argparse.ArgumentParser(description="SALBP Balanced (Gurobi) – Min C")
    ap.add_argument("--tasks", required=True, help="CSV con task_id,time,predecessors")
    ap.add_argument("--stations", type=int, required=True, help="Numero stazioni")
    ap.add_argument("--time-limit", type=int, default=0, help="Time limit (s). 0 = nessun limite")
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    # 1) Carica istanza
    inst = Instance.from_csv(args.tasks)

    # 2) Costruisci modello
    model = SALBPMinMaxModel(inst, num_stations=args.stations)
    model.build()

    # 3) Logger (event-based)
    logger = None
    try:
        if 'ProgressLogger' in globals():
            logger = ProgressLogger(min_dt=0.0)  # nessun throttling
    except Exception:
        logger = None

    # 4) Risolvi
    t0 = time.perf_counter()
    sol = model.solve(
        time_limit=(args.time_limit if args.time_limit and args.time_limit > 0 else None),
        mip_gap=args.mip_gap,
        threads=args.threads,
        log=args.log,
        cb=logger
    )
    total_elapsed = time.perf_counter() - t0

    print(f"Status        : {sol.status}")
    print(f"Min max load C: {sol.C}")

    # 5) Output su console
    loads_sorted = sorted(sol.station_loads, key=lambda t: t[0])  # [(stazione, carico)]
    for s, l in loads_sorted:
        print(f"Stazione {s}: {l}")
    for tsk, st in sorted(sol.assignment, key=lambda x: (x[1], x[0])):
        print(f"Task {tsk} -> Stazione {st}")

    # 6) Metriche (se servono)
    loads_only = [l for _, l in loads_sorted]
    m = balance_metrics(loads_only)
    print("-- Metriche --")
    for k, v in m.items():
        print(f"{k}: {v}")

    # 7) Output dir
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 8) Progress: prendi snapshot + aggiungi sempre il finale
    time_s, best, bound = [], [], []
    try:
        if logger is not None and hasattr(logger, "snaps"):
            time_s = [s.t for s in logger.snaps]
            best   = [s.best for s in logger.snaps]
            bound  = [s.bound for s in logger.snaps]

        final_bound = sol.C if str(sol.status).upper() == "OPTIMAL" else None
        if logger is not None and hasattr(logger, "finalize"):
            logger.finalize(best=sol.C, bound=final_bound)
            time_s = [s.t for s in logger.snaps]
            best   = [s.best for s in logger.snaps]
            bound  = [s.bound for s in logger.snaps]

        # progress.csv (semplice)
        import csv
        with open(outdir / "progress.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "best", "bound"])
            w.writeheader()
            for t, b, bd in zip(time_s, best, bound):
                w.writerow({"t": f"{t:.9f}", "best": "" if b is None else b, "bound": "" if bd is None else bd})

        # ★★★ Plot finale in UNITÀ REALI (raw) — niente più numeri “strani”
        plot_progress(time_s, best, bound, sol.C,
                      str(outdir / "progress_incumbent_bound.png"),
                      str(outdir / "progress_gap.png"),
                      scale_mode="raw")

        print(f"[OK] progress_*.png scritti in {outdir}")
    except Exception as e:
        print(f"[WARN] Sezione progress fallita: {e}")

    # 9) CSV risultati
    import csv as _csv
    with open(outdir / "station_loads.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["station", "load"])
        w.writeheader()
        for s, l in loads_sorted:
            w.writerow({"station": s, "load": l})

    with open(outdir / "assignment.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["task_id", "station"])
        w.writeheader()
        for t, s in sorted(sol.assignment, key=lambda x: (x[1], x[0])):
            w.writerow({"task_id": t, "station": s})

    with open(outdir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(m.keys()))
        w.writeheader()
        w.writerow(m)

    print(f"[OK] CSV salvati in {outdir}")




if __name__ == "__main__":
    main()
