from __future__ import annotations
import argparse
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

from salbp.instance import Instance
from salbp.model import SALBPMinMaxModel          # modello "y"
from salbp.model_prefix import SALBPPrefixModel   # modello "prefix"
from salbp.metrics import balance_metrics
from salbp.monitor import ProgressLogger, plot_progress, plot_gap, plot_station_loads

def resolve_tasks_path(raw: str) -> Path:
    """Rende robusta la risoluzione del CSV: accetta path assoluti e vari relativi."""
    here = Path(__file__).resolve().parent          # .../salbp
    root = here.parent                              # root progetto
    candidates = [
        Path(raw),                                  # così com'è
        Path.cwd() / raw,                           # relativo alla cwd
        here / raw,                                 # relativo a salbp/
        here / "examples" / Path(raw).name,         # se passi solo il filename
        root / raw,                                 # relativo alla root
        root / "salbp" / "examples" / Path(raw).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"CSV non trovato: {raw}\nProvati:\n" + "\n".join(str(p) for p in candidates))

def solve_and_report(model, inst, args, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)

    logger = ProgressLogger()
    sol = model.solve(time_limit=args.time_limit,
                      mip_gap=args.mip_gap,
                      threads=args.threads,
                      log=args.log,
                      cb=logger)

    # snapshot finale (evita grafici vuoti)
    try:
        logger.ensure_final_snapshot(model.model, sol.C)
    except Exception:
        pass

    m = model.model
    print(f"\n== [{tag}] ==")
    print(f"Model vars={m.NumVars} (bin={m.NumBinVars}, int={m.NumIntVars}) constrs={m.NumConstrs}")
    print(f"Status     : {sol.status}")
    print(f"Runtime (s): {float(getattr(m,'Runtime',0.0)):.3f}")
    print(f"Nodi       : {int(getattr(m,'NodeCount',0))}")
    if sol.C is not None:
        print(f"Obj (C)    : {sol.C}")
    try:
        print(f"Best bound : {float(m.ObjBound)}")
    except Exception:
        pass
    if getattr(m, "MIPGap", None) is not None:
        print(f"Gap finale : {100*float(m.MIPGap):.4f}%")

    # carichi / assegnamento
    loads = [l for _, l in sorted(sol.station_loads)]
    print("-- Carichi --")
    for s, l in sorted(sol.station_loads):
        print(f"Stazione {s}: {l}")
    print("-- Assegnamento (primi 20) --")
    for t, s in sorted(sol.assignment, key=lambda x:(x[1],x[0]))[:20]:
        print(f"{t} -> {s}")

    # metriche
    mtr = balance_metrics(loads)
    print("-- Metriche --")
    for k, v in mtr.items():
        print(f"{k}: {v}")

    # salvataggi
    logger.to_csv(str(outdir / "progress.csv"))
    ts = [s.t for s in logger.snaps]; best=[s.best for s in logger.snaps]; bound=[s.bound for s in logger.snaps]; gap=[s.gap for s in logger.snaps]
    plot_progress(ts, best, bound, str(outdir / "progress_incumbent_bound.png"))
    plot_gap(ts, gap, str(outdir / "progress_gap.png"))
    plot_station_loads({s:l for s,l in sol.station_loads}, str(outdir / "station_loads.png"))

    if pd:
        pd.DataFrame([{"station":s,"load":l} for s,l in sol.station_loads]).to_csv(outdir/"station_loads.csv", index=False)
        pd.DataFrame([{"task_id":t,"station":s} for t,s in sol.assignment]).to_csv(outdir/"assignment.csv", index=False)
        pd.DataFrame([mtr]).to_csv(outdir/"metrics.csv", index=False)

def main():
    ap = argparse.ArgumentParser(description="SALBP Balanced – Min C (y / prefix)")
    ap.add_argument("--tasks", required=True, help="CSV con task_id,time,predecessors")
    ap.add_argument("--stations", type=int, required=True, help="Numero stazioni M")
    ap.add_argument("--formulation", choices=["y","prefix","both"], default="y")
    ap.add_argument("--time-limit", type=int, default=None, help="omesso = senza limite")
    ap.add_argument("--mip-gap", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-cb", action="store_true")
    args = ap.parse_args()

    tasks_path = resolve_tasks_path(args.tasks)
    print(f"→ Istanza: {tasks_path}")

    inst = Instance.from_csv(str(tasks_path))

    print(
        f"[IST] tasks={len(inst.tasks)}, arcs={sum(len(v) for v in inst.preds.values())}, sum_t={sum(inst.times.values())}, M={args.stations}")

    logger = None if args.no_cb else ProgressLogger()
    if args.formulation in ("y","both"):
        mdl_y = SALBPMinMaxModel(inst, args.stations)
        mdl_y.build()
        solve_and_report(mdl_y, inst, args, Path(args.outdir)/"y", tag="y")

    if args.formulation in ("prefix","both"):
        mdl_p = SALBPPrefixModel(inst, args.stations)
        mdl_p.build()
        solve_and_report(mdl_p, inst, args, Path(args.outdir)/"prefix", tag="prefix")

if __name__ == "__main__":
    main()
