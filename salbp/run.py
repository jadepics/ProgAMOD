from __future__ import annotations
import argparse
from pathlib import Path

from salbp.ls_one_move import one_move_local_search  # 1-move LS

from salbp.metrics import build_run_summary, save_run_metrics_per_formulation
from salbp.monitor import plot_progress_compare, plot_gap_compare

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

    logger = ProgressLogger() if not getattr(args, "no_cb", False) else None
    sol = model.solve(time_limit=args.time_limit,
                      mip_gap=args.mip_gap,
                      threads=args.threads,
                      log=args.log,
                      cb=logger)

    # snapshot finale per grafici
    try:
        if logger is not None:
            logger.ensure_final_snapshot(model.model, sol.C)
    except Exception:
        pass

    # --- 1-MOVE (opzionale, PRIMA dei salvataggi) ---
    try:
        if getattr(args, "post_1move", False):
            print(f"[LS 1-move:{tag}] start...")

            # 1) Prova a usare direttamente sol.assignment se presente
            st_map = {}
            ass = getattr(sol, "assignment", None)
            if ass:
                # sol.assignment è iterabile: [(task, station), ...]
                st_map = {t: s for (t, s) in ass}
            else:
                # 2) Fallback: estrai dalle variabili del modello (tipico: model.x[(i,s)])
                try:
                    xvars = getattr(model, "x", None) or {}
                    tmp = {}
                    for key, var in xvars.items():
                        try:
                            val = float(var.X)
                        except Exception:
                            val = float(getattr(var, "X", 0.0))
                        if val >= 0.5:
                            # key può essere (i, s) oppure una chiave diversa: gestiamo il caso tuple
                            if isinstance(key, tuple) and len(key) == 2:
                                i, s = key
                            else:
                                # se il nome della var è tipo "x[A,3]" o "x_A_3" non lo parsiamo qui: meglio fallire chiaro
                                raise RuntimeError("Forma delle chiavi x non supportata: usa tuple (task, station).")
                            tmp[i] = int(s)
                    st_map = tmp
                except Exception as _e:
                    st_map = {}

            # 3) Armonizza il tipo dei task_id (string vs int) per farli combaciare con inst.tasks
            try:
                if st_map:
                    sample_inst = next(iter(inst.tasks))
                    sample_map = next(iter(st_map.keys()))
                    if isinstance(sample_inst, str) and not isinstance(sample_map, str):
                        st_map = {str(k): v for k, v in st_map.items()}
                    elif isinstance(sample_inst, int) and not isinstance(sample_map, int):
                        st_map = {int(k): v for k, v in st_map.items()}
            except Exception:
                pass

            if not st_map:
                raise RuntimeError("Impossibile costruire l'assegnamento per la 1-move (assignment mancante).")

            # 4) Esegui la 1-move
            ls = one_move_local_search(inst,
                                       S=args.stations,
                                       station_of=st_map,
                                       time_limit=getattr(args, "ls_secs", 2.0))

            C_before = sol.C if sol.C is not None else float("inf")
            if ls.C < C_before:
                print(f"[LS 1-move:{tag}] miglioramento: C {C_before} -> {ls.C} (Δ={C_before - ls.C})")
                # ricostruisci il "sol" con i nuovi dati
                new_assignment = sorted(ls.station_of.items(), key=lambda x: (x[1], x[0]))
                new_station_loads = [(s, float(l)) for s, l in enumerate(ls.loads, start=1)]
                sol = type(sol)(
                    status=sol.status,
                    C=float(ls.C),
                    assignment=new_assignment,
                    station_loads=new_station_loads
                )
            else:
                print(f"[LS 1-move:{tag}] nessun miglioramento (C = {C_before})")
    except Exception as e:
        print(f"[LS 1-move:{tag}] SKIP per errore: {e}")

    # --- REPORT / SALVATAGGI ---
    m = model.model
    try:
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
    except Exception:
        pass

    # carichi/metriche
    try:
        loads = [l for _, l in sorted(sol.station_loads)]
    except Exception:
        loads = []
    try:
        row = build_run_summary(
            formulation=tag,
            instance_name=Path(args.tasks).name,
            loads=loads,
            obj_C=sol.C,
            bound=float(getattr(m, "ObjBound", float("nan"))) if getattr(m, "SolCount", 0) else None,
            runtime=float(getattr(m, "Runtime", 0.0)),
            nodes=int(getattr(m, "NodeCount", 0)),
        )
        save_run_metrics_per_formulation(Path(args.outdir), tag, row)
    except Exception as e:
        print(f"[WARN] metriche non salvate: {e}")

    # grafici & CSV
    try:
        if logger is not None:
            logger.to_csv(str(outdir / "progress.csv"))
            ts   = [s.t     for s in logger.snaps]
            best = [s.best  for s in logger.snaps]
            bnd  = [s.bound for s in logger.snaps]
            gap  = [s.gap   for s in logger.snaps]
            plot_progress(ts, best, bnd, str(outdir / "progress_incumbent_bound.png"))
            plot_gap(ts, gap, str(outdir / "progress_gap.png"))
    except Exception as e:
        print(f"[WARN] grafici progresso non salvati: {e}")

    try:
        plot_station_loads({s:l for s,l in sol.station_loads}, str(outdir / "station_loads.png"))
    except Exception as e:
        print(f"[WARN] plot station_loads non salvato: {e}")

    try:
        if pd:
            import pandas as _pd
            _pd.DataFrame([{"station":s,"load":l} for s,l in sol.station_loads]).to_csv(outdir/"station_loads.csv", index=False)
            _pd.DataFrame([{"task_id":t,"station":s} for t,s in sol.assignment]).to_csv(outdir/"assignment.csv", index=False)
    except Exception as e:
        print(f"[WARN] CSV non salvati: {e}")

    return sol, logger

def solve_heuristic_only(inst, args, outdir: Path):
        from salbp.constructive import construct_targetC
        outdir.mkdir(parents=True, exist_ok=True)

        order = "rpw" if args.heuristic == "targetC-rpw" else "lpt"
        print(f"[HEUR] Costruttiva: Target-C ({order})")

        st_map, loads, C = construct_targetC(inst,
                                             S=args.stations,
                                             order=order,
                                             eps_step=args.target_eps_step)

        if not st_map:
            print("[HEUR] Fallita la costruzione con Target-C: aumenta --target-eps-step o S.")
            return

        # opzionale: rifinitura 1-move
        if getattr(args, "post_1move", False):
            print("[HEUR] 1-move post-costruzione...")
            from salbp.ls_one_move import one_move_local_search
            ls = one_move_local_search(inst,
                                       S=args.stations,
                                       station_of=st_map,
                                       time_limit=getattr(args, "ls_secs", 2.0),
                                       accept_equal=getattr(args, "ls_accept_equal", False) if hasattr(args,
                                                                                                       "ls_accept_equal") else False,
                                       tie_metric=getattr(args, "ls_tie", "range") if hasattr(args,
                                                                                              "ls_tie") else "range")
            if ls.C < C or (ls.C == C and sum(ls.loads) == sum(loads)):
                st_map, loads, C = ls.station_of, ls.loads, ls.C
                print(f"[HEUR] 1-move done. C = {C}")

        # --- report & salvataggi minimi ---
        print(f"\n== [heuristic] ==")
        print(f"Obj (C)    : {C}")
        try:
            plot_station_loads({s: l for s, l in enumerate(loads, start=1)}, str(outdir / "station_loads.png"))
        except Exception:
            pass

        try:
            if pd:
                import pandas as _pd
                _pd.DataFrame([{"station": s, "load": l} for s, l in enumerate(loads, start=1)]).to_csv(
                    outdir / "station_loads.csv", index=False)
                _pd.DataFrame([{"task_id": t, "station": s} for t, s in
                               sorted(st_map.items(), key=lambda x: (x[1], x[0]))]).to_csv(outdir / "assignment.csv",
                                                                                           index=False)
        except Exception:
            pass

        return st_map, loads, C

    # carichi / assegnamento
'''loads = [l for _, l in sorted(sol.station_loads)]
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
'''

def main():
    ap = argparse.ArgumentParser(description="SALBP Balanced – Min C (y / prefix)")
    ap.add_argument("--tasks", required=True, help="CSV con task_id,time,predecessors")
    ap.add_argument("--stations", type=int, required=True, help="Numero stazioni M")
    ap.add_argument("--formulation", choices=["y","prefix","both"], default="y")

    # --- opzioni 1-move ---
    ap.add_argument("--post-1move", action="store_true",
                    help="Esegui local search 1-move sulla soluzione prodotta dal modello.")
    ap.add_argument("--ls-secs", type=float, default=2.0,
                    help="Time cap (secondi) per la 1-move (default: 2.0).")
    ap.add_argument("--ls-mode", choices=["first", "best"], default="first",
                    help="(placeholder, attualmente è implementato 'first')")

    ap.add_argument("--heuristic-only", action="store_true",
                    help="Esegui solo euristiche (niente PLI).")
    ap.add_argument("--heuristic", choices=["targetC-rpw", "targetC-lpt"], default="targetC-rpw",
                    help="Costruttiva da usare in heuristic-only.")
    ap.add_argument("--target-eps-step", type=int, default=1,
                    help="Incremento di C nella costruttiva Target-C (default: 1).")
    ap.add_argument("--ls-accept-equal", action="store_true",
                    help="Permetti mosse con C invariato se migliorano il tie-break.")
    ap.add_argument("--ls-tie", choices=["range", "var"], default="range",
                    help="Metrica di tie-break quando C non migliora (default: range).")

    # --- altre opzioni ---
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
    print(f"[IST] tasks={len(inst.tasks)}, arcs={sum(len(v) for v in inst.preds.values())}, "
          f"sum_t={sum(inst.times.values())}, M={args.stations}")

    if args.heuristic_only:
        solve_heuristic_only(inst, args, Path(args.outdir)/"heuristic")
        return

    # risoluzione (UNA VOLTA) per ciascuna formulazione richiesta
    if args.formulation in ("y","both"):
        mdl_y = SALBPMinMaxModel(inst, args.stations)
        mdl_y.build()
        solve_and_report(mdl_y, inst, args, Path(args.outdir)/"y", tag="y")

    if args.formulation in ("prefix","both"):
        mdl_p = SALBPPrefixModel(inst, args.stations)
        mdl_p.build()
        solve_and_report(mdl_p, inst, args, Path(args.outdir)/"prefix", tag="prefix")

    # Grafici comparativi se both
    if args.formulation == "both":
        base = Path(args.outdir)
        plot_progress_compare(base / "y" / "progress.csv", base / "prefix" / "progress.csv",
                              str(base / "compare" / "progress_compare.png"))
        plot_gap_compare(base / "y" / "progress.csv", base / "prefix" / "progress.csv",
                         str(base / "compare" / "gap_compare.png"))
        print(f"\nGrafici comparativi salvati in: {base / 'compare'}")


if __name__ == "__main__":
    main()