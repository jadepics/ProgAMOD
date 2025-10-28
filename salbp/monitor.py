from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math, csv, os
import matplotlib.pyplot as plt
from gurobipy import GRB
import time
from types import SimpleNamespace

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
    """
    Logger di progresso per Gurobi (event-based).
    - Registra uno snapshot SOLO quando cambiano incumbent o bound.
    - Usa solo attributi consentiti nel contesto corrente.
    - Timestamp sempre strettamente crescente (niente linee verticali).
    - finalize() aggiunge comunque uno snapshot finale.
    """
    def __init__(self, min_dt=0.0, atol=1e-9, rtol=1e-9):
        self.t0 = None
        self.snaps = []               # SimpleNamespace(t, best, bound, gap)
        self._last_t = -1e9
        self._last_best = None
        self._last_bound = None
        self.min_dt = float(min_dt)   # throttling opzionale (default: nessuno)
        self.atol = float(atol)
        self.rtol = float(rtol)

    def _changed(self, old, new):
        if old is None and new is not None: return True
        if new is None and old is not None: return True
        if old is None and new is None:     return False
        # differenza assoluta/relativa
        diff = abs(new - old)
        return diff > max(self.atol, self.rtol * (abs(old) + 1e-12))

    def __call__(self, model, where):
        import gurobipy as gp
        if self.t0 is None:
            self.t0 = time.perf_counter()

        # Leggi solo ciò che è lecito in questo "where"
        try:
            if where == gp.GRB.Callback.MIP:
                best  = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            elif where == gp.GRB.Callback.MIPSOL:
                best  = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            elif where == gp.GRB.Callback.MIPNODE:
                best  = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)
                bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)
            else:
                return
        except Exception:
            return  # attributo non disponibile in questo evento

        # Coercion safe
        def f(x):
            try:
                x = float(x)
                if x == float("inf") or x == float("-inf"):
                    return None
                return x
            except Exception:
                return None

        best  = f(best)
        bound = f(bound)
        t = time.perf_counter() - self.t0

        # Throttling opzionale
        if t - self._last_t < self.min_dt:
            return

        # Registra solo se qualcosa è cambiato
        if not self._changed(self._last_best, best) and not self._changed(self._last_bound, bound):
            return

        # Garantisci tempo crescente
        if t <= self._last_t:
            t = self._last_t + 1e-6

        gap = None
        if best is not None and bound is not None and best != 0:
            gap = abs(best - bound) / (abs(best) + 1e-9) * 100.0

        self.snaps.append(SimpleNamespace(t=t, best=best, bound=bound, gap=gap))
        self._last_t = t
        self._last_best = best
        self._last_bound = bound

    def finalize(self, best=None, bound=None):
        """Aggiunge uno snapshot finale con t > ultimo t, anche se identico nei valori."""
        if self.t0 is None:
            self.t0 = time.perf_counter()
        t = time.perf_counter() - self.t0
        if self.snaps and t <= self._last_t:
            t = self._last_t + 1e-6  # 1 µs di padding

        gap = None
        if best is not None and bound is not None and best != 0:
            gap = abs(best - bound) / (abs(best) + 1e-9) * 100.0

        self.snaps.append(SimpleNamespace(t=t, best=best, bound=bound, gap=gap))
        self._last_t = t
        self._last_best = best
        self._last_bound = bound

    def to_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "best", "bound", "gap"])
            w.writeheader()
            for s in self.snaps:
                w.writerow({
                    "t": f"{s.t:.9f}",
                    "best": "" if s.best is None else s.best,
                    "bound": "" if s.bound is None else s.bound,
                    "gap": "" if s.gap is None else f"{s.gap:.6f}",
                })

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
