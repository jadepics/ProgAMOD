from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math, csv, os
import matplotlib.pyplot as plt
from gurobipy import GRB

#classe per file di monitoraggio fa callback su Gurobi e raccoglie i dati (ProgressLogger)-> grafici
# plot_progress, plot_gap, plot_station_loads AL MOMENTO NON FUNZIONANO

@dataclass
class Snapshot:
    t: float           # tempo di runtime (s)
    best: float | None # miglior soluzione finora (C)
    bound: float | None# best bound corrente
    gap: float | None  # relative gap (0..1) se both best&bound known
    nodes: float       # nodi esplorati nel b&b

class ProgressLogger:
    def __init__(self):
        self.snaps: List[Snapshot] = []

    def __call__(self, model, where):
        # Log SOLO sugli eventi MIP (progresso) e MIPSOL (nuovo incumbent)
        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
            t = model.cbGet(GRB.Callback.RUNTIME)
            nodes = model.cbGet(GRB.Callback.MIP_NODCNT)
            best = model.cbGet(GRB.Callback.MIP_OBJBST)   # incumbent (minimization)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)  # best bound
            # best o bound possono essere +/- inf all'inizio
            b = float(best) if (best < GRB.INFINITY) else None
            d = float(bound) if (abs(bound) < GRB.INFINITY) else None
            g = None
            if b is not None and d is not None and b != 0:
                # gap = (b - d)/|b| per minimizzazione (bound <= best)
                g = max(0.0, (b - d) / abs(b))
            self.snaps.append(Snapshot(t, b, d, g, nodes))

    def to_csv(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t","best","bound","gap","nodes"])
            w.writeheader()
            for s in self.snaps:
                w.writerow(asdict(s))

def plot_progress(time_s: List[float], best: List[float|None], bound: List[float|None], out_png: str):
    plt.figure()
    # filtra None con compressione semplice
    tb = [(t,v) for t,v in zip(time_s,best) if v is not None]
    td = [(t,v) for t,v in zip(time_s,bound) if v is not None]
    if tb:
        plt.plot([t for t,_ in tb], [v for _,v in tb], label="Incumbent (C)")
    if td:
        plt.plot([t for t,_ in td], [v for _,v in td], label="Best bound")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valore obiettivo")
    plt.title("Evoluzione incumbent vs bound")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_gap(time_s: List[float], gap: List[float|None], out_png: str):
    plt.figure()
    tg = [(t,g) for t,g in zip(time_s,gap) if g is not None and g>0]
    if tg:
        plt.semilogy([t for t,_ in tg], [g for _,g in tg])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Gap relativo")
    plt.title("Gap relativo vs tempo (scala log)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_station_loads(loads: Dict[int, float], out_png: str):
    plt.figure()
    xs = sorted(loads.keys())
    ys = [loads[s] for s in xs]
    plt.bar([str(s) for s in xs], ys)
    plt.xlabel("Stazione")
    plt.ylabel("Carico")
    plt.title("Carichi per stazione")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
