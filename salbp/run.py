from __future__ import annotations
import argparse, os
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


def main():
    import argparse
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

    # 3) Logger/Monitor se disponibile
    logger = None
    try:
        if 'ProgressLogger' in globals():
            logger = ProgressLogger()
    except Exception:
        logger = None

    # 4) Risolvi
    sol = model.solve(
        time_limit=(args.time_limit if args.time_limit and args.time_limit > 0 else None),
        mip_gap=args.mip_gap,
        threads=args.threads,
        log=args.log,
        cb=logger
    )

    print(f"Status        : {sol.status}")
    print(f"Min max load C: {sol.C}")

    # 5) Output su console
    loads_sorted = sorted(sol.station_loads, key=lambda t: t[0])  # [(stazione, carico)]
    for s, l in loads_sorted:
        print(f"Stazione {s}: {l}")
    for t, s in sorted(sol.assignment, key=lambda x: (x[1], x[0])):
        print(f"Task {t} -> Stazione {s}")

    # 6) Metriche
    loads_only = [l for _, l in loads_sorted]
    m = balance_metrics(loads_only)
    print("-- Metriche --")
    for k, v in m.items():
        print(f"{k}: {v}")

    # 7) Cartella output
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 8) SEMPRE: grafico carichi per stazione
    loads_dict = {s: l for s, l in loads_sorted}
    try:
        # prova a usare la funzione (se è stata importata altrove)
        plot_station_loads(loads_dict, str(outdir / "station_loads.png"))  # type: ignore[name-defined]
        print(f"[OK] station_loads.png scritto in {outdir} (monitor)")
    except NameError:
        # fallback locale
        plot_station_loads_fallback(loads_dict, str(outdir / "station_loads.png"))
        print(f"[OK] station_loads.png scritto in {outdir} (fallback)")
    except Exception as e:
        print(f"[WARN] station_loads.png non generato: {e}")

    # 9) Grafici di progresso solo se ci sono snapshot
    try:
        has_snaps = (logger is not None) and hasattr(logger, "snaps") and len(logger.snaps) > 0
        if has_snaps:
            logger.to_csv(str(outdir / "progress.csv"))
            time_s = [s.t for s in logger.snaps]
            best   = [s.best for s in logger.snaps]
            bound  = [s.bound for s in logger.snaps]
            gap    = [s.gap for s in logger.snaps]
            try:
                plot_progress(time_s, best, bound, str(outdir / "progress_incumbent_bound.png"))  # type: ignore[name-defined]
                plot_gap(time_s, gap, str(outdir / "progress_gap.png"))  # type: ignore[name-defined]
                print(f"[OK] progress.csv e grafici di progresso scritti in {outdir} (monitor)")
            except NameError:
                # Nessun fallback qui: i progress hanno senso solo con il monitor
                print("[INFO] Funzioni di plot del monitor non disponibili: skip progress_*.png")
        else:
            print("[INFO] Nessuno snapshot dal callback: salto progress.csv/progress_*.png")
    except Exception as e:
        print(f"[WARN] Grafici di progresso non generati: {e}")

    # 10) CSV risultati
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
