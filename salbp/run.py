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
from salbp.monitor import ( ProgressLogger, plot_progress, plot_gap, plot_station_loads, plot_incumbent_step_hist,
                            plot_progress_milestones, plot_bestbound_vs_nodes, plot_gap_targets, plot_nodes_over_time,
                            plot_primal_dual_ribbon,
                            )
from salbp.vnd import vnd_search  # VND metaeuristica


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

    # snapshot finale per progress.csv (include best/bound/nodes)
    try:
        if logger is not None:
            m = model.model
            try:
                _bound = float(getattr(m, "ObjBound", None))
            except Exception:
                _bound = None
            ncount = int(getattr(m, "NodeCount", 0))
            logger.finalize(best=sol.C, bound=_bound, nodes=ncount)
    except Exception as _e_fin:
        print(f"[WARN] finalize logger: {_e_fin}")

    # --- 1-MOVE le, PRIMA dei salvataggi) ---
    try:
        if getattr(args, "post_1move", False):
            print(f"[LS 1-move:{tag}] start...")

            # 1) assignment dalla soluzione o dalle x vars
            st_map = {}
            ass = getattr(sol, "assignment", None)
            if ass:
                st_map = {t: s for (t, s) in ass}
            else:
                try:
                    xvars = getattr(model, "x", None) or {}
                    tmp = {}
                    for key, var in xvars.items():
                        try:
                            val = float(var.X)
                        except Exception:
                            val = float(getattr(var, "X", 0.0))
                        if val >= 0.5:
                            if isinstance(key, tuple) and len(key) == 2:
                                i, s = key
                            else:
                                raise RuntimeError("Chiavi x non supportate (attese tuple (task, station)).")
                            tmp[i] = int(s)
                    st_map = tmp
                except Exception:
                    st_map = {}

            # 2) armonizza tipo id task
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

            # 3) esegui LS con trace
            from salbp.ls_one_move import one_move_local_search
            ls = one_move_local_search(inst,
                                       S=args.stations,
                                       station_of=st_map,
                                       time_limit=getattr(args, "ls_secs", 2.0),
                                       accept_equal=getattr(args, "ls_accept_equal", False) if hasattr(args,
                                                                                                       "ls_accept_equal") else False,
                                       tie_metric=getattr(args, "ls_tie", "range") if hasattr(args,
                                                                                              "ls_tie") else "range",
                                       cap_C=None,
                                       record=True)

            # 4) salva report LS (CSV + PNG) se richiesto
            try:
                if getattr(args, "ls_report", False) and ls.trace:
                    ls_dir = outdir / "ls_1move"
                    ls_dir.mkdir(parents=True, exist_ok=True)
                    from salbp.ls_report import save_ls_csv, plot_ls_c, plot_ls_metric
                    save_ls_csv(ls.trace, ls_dir / "ls_trace.csv")
                    plot_ls_c(ls.trace, ls_dir / "ls_C_over_steps.png", by="step")
                    plot_ls_metric(ls.trace, ls_dir / "ls_range_over_steps.png", metric="range", by="step")
                    plot_ls_metric(ls.trace, ls_dir / "ls_var_over_steps.png", metric="var", by="step")
                    # versione "by time" facoltativa:
                    # plot_ls_c(ls.trace, ls_dir / "ls_C_over_time.png", by="time")
                    # plot_ls_metric(ls.trace, ls_dir / "ls_range_over_time.png", metric="range", by="time")
                    # plot_ls_metric(ls.trace, ls_dir / "ls_var_over_time.png", metric="var", by="time")
            except Exception as _eplot:
                print(f"[LS 1-move:{tag}] report LS non salvato: {_eplot}")

            # 5) applica miglioramento se C scende (o lascialo come reference se reporting only)
            C_before = sol.C if sol.C is not None else float("inf")
            if ls.C < C_before:
                print(f"[LS 1-move:{tag}] miglioramento: C {C_before} -> {ls.C} (Δ={C_before - ls.C})")
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

    # --- VND (opzionale) ---
    try:
        if getattr(args, "post_vnd", False):
            print(f"[VND:{tag}] start...")
            # prendi l'assegnamento corrente (già migliorato da 1-move se attivo)
            st_map = {t: s for (t, s) in sol.assignment} if getattr(sol, "assignment", None) else {}
            if not st_map:
                raise RuntimeError("VND: assignment mancante.")
            vnd = vnd_search(inst,
                             S=args.stations,
                             station_of=st_map,
                             time_limit=float(getattr(args, "vnd_secs", 5.0)),
                             accept_equal=bool(getattr(args, "vnd_accept_equal", False)),
                             tie_metric=str(getattr(args, "vnd_tie", "range")),
                             use_swap=True,
                             use_ejection=True)
            if vnd.C < sol.C:
                print(f"[VND:{tag}] miglioramento: C {sol.C} -> {vnd.C} (iters={vnd.iters}, "
                      f"1m={vnd.moves_1move}, sw={vnd.moves_swap}, ej={vnd.moves_eject})")
                new_assignment = sorted(vnd.station_of.items(), key=lambda x: (x[1], x[0]))
                new_station_loads = [(s, float(l)) for s, l in enumerate(vnd.loads, start=1)]
                sol = type(sol)(
                    status=sol.status,
                    C=float(vnd.C),
                    assignment=new_assignment,
                    station_loads=new_station_loads
                )
            else:
                print(f"[VND:{tag}] nessun miglioramento (C = {sol.C})")
    except Exception as e:
        print(f"[VND:{tag}] SKIP per errore: {e}")

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

        # A1 - Istogramma degli step di miglioramento dell'incumbent
        try:
            plot_incumbent_step_hist(str(outdir / "progress.csv"),
                                     str(outdir / "A1_incumbent_step_hist.png"),
                                     title=f"{Path(args.tasks).name} [{tag}]")
        except Exception as _eA1:
            print(f"[WARN] A1 histogram non salvato: {_eA1}")

        # A2 - Milestones sul progresso (richiede LB)
        try:
            import math as _math
            lb = _math.ceil(sum(inst.times.values()) / args.stations)
            plot_progress_milestones(str(outdir / "progress.csv"),
                                     str(outdir / "A2_progress_milestones.png"),
                                     lb=lb,
                                     title=f"{Path(args.tasks).name} [{tag}]")
        except Exception as _eA2:
            print(f"[WARN] A2 milestones non salvato: {_eA2}")

        # B6 - Best bound vs Nodi (staircase)
        try:
            plot_bestbound_vs_nodes(
                str(outdir / "progress.csv"),
                str(outdir / "B6_bestbound_vs_nodes.png"),
                title=f"{Path(args.tasks).name} [{tag}]"
            )
        except Exception as _eB6:
            print(f"[WARN] B6 bestbound-vs-nodes non salvato: {_eB6}")

        # C8 - Gap vs tempo con soglie (10,5,1,0.5%)
        try:
            plot_gap_targets(
                str(outdir / "progress.csv"),
                str(outdir / "C8_gap_targets.png"),
                targets=(10.0, 5.0, 1.0, 0.5),
                title=f"{Path(args.tasks).name} [{tag}]"
            )
        except Exception as _eC8:
            print(f"[WARN] C8 gap-targets non salvato: {_eC8}")

        # D10 - Nodi esplorati nel tempo con marker sui miglioramenti incumbent
        try:
            plot_nodes_over_time(
                str(outdir / "progress.csv"),
                str(outdir / "D10_nodes_over_time.png"),
                title=f"{Path(args.tasks).name} [{tag}]"
            )
        except Exception as _eD10:
            print(f"[WARN] D10 nodes-over-time non salvato: {_eD10}")

        # E14 - Primal–Dual ribbon (area tra incumbent e bound)
        try:
            plot_primal_dual_ribbon(
                str(outdir / "progress.csv"),
                str(outdir / "E14_primal_dual_ribbon.png"),
                title=f"{Path(args.tasks).name} [{tag}]"
            )
        except Exception as _eE14:
            print(f"[WARN] E14 primal-dual ribbon non salvato: {_eE14}")





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

    if args.heuristic == "targetC-bestfit":
        print("[HEUR] Costruttiva: Target-C (bestfit)")
        from salbp.greedy_bestfit import construct_targetC_bestfit
        st_map, loads, C = construct_targetC_bestfit(inst,
                                                     S=args.stations,
                                                     eps_step=args.target_eps_step)
    else:
        order = "rpw" if args.heuristic == "targetC-rpw" else "lpt"
        print(f"[HEUR] Costruttiva: Target-C ({order})")
        from salbp.constructive import construct_targetC
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
                                   accept_equal=getattr(args, "ls_accept_equal", False) if hasattr(args, "ls_accept_equal") else False,
                                   tie_metric=getattr(args, "ls_tie", "range") if hasattr(args, "ls_tie") else "range",
                                   cap_C=None,
                                   record=True)  # registra la trace per i grafici

        # --- SALVATAGGIO REPORT 1-MOVE (CSV + PNG) ---
        try:
            if getattr(args, "ls_report", False) and ls and ls.trace:
                ls_dir = outdir / "ls_1move"
                ls_dir.mkdir(parents=True, exist_ok=True)
                from salbp.ls_report import save_ls_csv, plot_ls_c, plot_ls_metric
                save_ls_csv(ls.trace, ls_dir / "ls_trace.csv")
                plot_ls_c(ls.trace, ls_dir / "ls_C_over_steps.png", by="step")
                plot_ls_metric(ls.trace, ls_dir / "ls_range_over_steps.png", metric="range", by="step")
                plot_ls_metric(ls.trace, ls_dir / "ls_var_over_steps.png", metric="var", by="step")
        except Exception as _eplot:
            print(f"[HEUR] report LS non salvato: {_eplot}")
        # --- FINE REPORT 1-MOVE ---

        # applica il miglioramento se C è sceso (o tieni la soluzione costruttiva)
        if ls.C < C or (ls.C == C and sum(ls.loads) == sum(loads)):
            st_map, loads, C = ls.station_of, ls.loads, ls.C
            print(f"[HEUR] 1-move done. C = {C}")

    # 👉👉👉 VND FUORI dall'if precedente, così funziona anche senza --post-1move
    if getattr(args, "post_vnd", False):
        try:
            print("[HEUR] VND post-costruzione...")
            vnd = vnd_search(inst,
                             S=args.stations,
                             station_of=st_map,
                             time_limit=float(getattr(args, "vnd_secs", 5.0)),
                             accept_equal=bool(getattr(args, "vnd_accept_equal", False)),
                             tie_metric=str(getattr(args, "vnd_tie", "range")),
                             use_swap=True,
                             use_ejection=True)
            if vnd.C < C:
                st_map, loads, C = vnd.station_of, vnd.loads, vnd.C
                print(f"[HEUR] VND done. C = {C} (iters={vnd.iters}, 1m={vnd.moves_1move}, sw={vnd.moves_swap}, ej={vnd.moves_eject})")
            else:
                print("[HEUR] VND nessun miglioramento")
        except Exception as e:
            print(f"[HEUR] VND SKIP per errore: {e}")

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
    ap.add_argument("--heuristic", choices=["targetC-rpw", "targetC-lpt", "targetC-bestfit"], default="targetC-rpw",
                    help="Costruttiva da usare in heuristic-only.")
    ap.add_argument("--target-eps-step", type=int, default=1,
                    help="Incremento di C nella costruttiva Target-C (default: 1).")
    ap.add_argument("--ls-accept-equal", action="store_true",
                    help="Permetti mosse con C invariato se migliorano il tie-break.")
    ap.add_argument("--ls-tie", choices=["range", "var"], default="range",
                    help="Metrica di tie-break quando C non migliora (default: range).")
    ap.add_argument("--ls-report", action="store_true",
                    help="Salva CSV e grafici dell'esecuzione della 1-move.")
    ap.add_argument("--post-vnd", action="store_true",
                    help="Esegui VND (1-move -> swap -> ejection) come rifinitura.")
    ap.add_argument("--vnd-secs", type=float, default=5.0,
                    help="Time cap (secondi) per la VND (default: 5.0).")
    ap.add_argument("--vnd-accept-equal", action="store_true",
                    help="In VND, permetti miglioramenti a C invariato con tie-break (range/var) nelle fasi locali.")
    ap.add_argument("--vnd-tie", choices=["range", "var"], default="range",
                    help="Tie-break quando C non scende (default: range).")

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