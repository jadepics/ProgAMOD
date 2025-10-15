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

# Se hai creato monitor.py come ti avevo mostrato:
try:
    from salbp.monitor import ProgressLogger, plot_progress, plot_gap, plot_station_loads
    HAVE_MONITOR = True
except Exception:
    HAVE_MONITOR = False
    ProgressLogger = None

def main():
    ap = argparse.ArgumentParser(description="SALBP Balanced (Gurobi) – Min C")
    ap.add_argument("--tasks", required=True, help="CSV con task_id,time,predecessors")
    ap.add_argument("--stations", type=int, required=True, help="Numero stazioni M")
    ap.add_argument("--time-limit", type=int, default=0, help="Time limit (s). 0 = nessun limite")
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    # Risolvi percorso CSV in modo robusto
    task_path = Path(args.tasks)
    if not task_path.exists():
        here = Path(__file__).resolve().parent   # cartella salbp/
        candidates = [here / args.tasks, here.parent / args.tasks]
        for c in candidates:
            if c.exists():
                task_path = c
                break
        if not task_path.exists():
            raise FileNotFoundError(f"CSV non trovato: {args.tasks}")

    inst = Instance.from_csv(str(task_path))
    mdl = SALBPMinMaxModel(inst, args.stations)
    mdl.build()

    logger = ProgressLogger() if HAVE_MONITOR else None

    sol = mdl.solve(time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                    threads=args.threads,
                    log=args.log,
                    cb=logger)

    print(f"Status        : {sol.status}")
    print(f"Min max load C: {sol.C}")
    loads = [l for _, l in sorted(sol.station_loads, key=lambda t: t[0])]
    for s, l in sorted(sol.station_loads, key=lambda t: t[0]):
        print(f"Stazione {s}: {l}")
    for t, s in sorted(sol.assignment, key=lambda x: (x[1], x[0])):
        print(f"Task {t} -> Stazione {s}")

    m = balance_metrics(loads)
    print("-- Metriche --")
    for k, v in m.items():
        print(f"{k}: {v}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Salva progresso + grafici (se il monitor è disponibile)
    if HAVE_MONITOR and logger is not None:
        logger.to_csv(str(outdir / "progress.csv"))
        time_s = [s.t for s in logger.snaps]
        best   = [s.best for s in logger.snaps]
        bound  = [s.bound for s in logger.snaps]
        gap    = [s.gap for s in logger.snaps]
        plot_progress(time_s, best, bound, str(outdir / "progress_incumbent_bound.png"))
        plot_gap(time_s, gap, str(outdir / "progress_gap.png"))
        loads_dict = {s: l for s, l in sol.station_loads}
        plot_station_loads(loads_dict, str(outdir / "station_loads.png"))
        print(f"\nFile generati in {outdir}:")
        print(" - progress.csv")
        print(" - progress_incumbent_bound.png")
        print(" - progress_gap.png")
        print(" - station_loads.png")

    # CSV dei risultati principali
    if pd:
        pd.DataFrame([{"station": s, "load": l} for s, l in sol.station_loads]).to_csv(outdir / "station_loads.csv", index=False)
        pd.DataFrame([{"task_id": t, "station": s} for t, s in sol.assignment]).to_csv(outdir / "assignment.csv", index=False)
        pd.DataFrame([m]).to_csv(outdir / "metrics.csv", index=False)
        print(f"CSV salvati in {outdir}")

if __name__ == "__main__":
    main()
