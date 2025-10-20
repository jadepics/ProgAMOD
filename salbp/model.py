from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import gurobipy as gp
from gurobipy import GRB
from salbp.instance import Instance


#qui dentro viene costruito il PLI da utilizzare scon Gurobi che viene infatti importata

@dataclass
class Solution:
    status: str
    C: Optional[float]
    assignment: List[Tuple[str, int]]        # (task, stazione)
    station_loads: List[Tuple[int, float]]   # (stazione, carico)

class SALBPMinMaxModel:
    """
    Variabili:
      x[i,s]∈{0,1}  assegnamento task i -> stazione s (1..M)
      y[i]∈[1..M]   indice stazione del task i (intera)
      L[s]≥0        carico della stazione s
      C≥0           massimo carico (da minimizzare)
    Vincoli:
      ∑_s x[i,s] = 1
      y[i] = ∑_s s * x[i,s]
      y[j] ≥ y[i]  ∀(i→j)
      L[s] = ∑_i t_i x[i,s]
      L[s] ≤ C
    Obiettivo: min C
    """
    def __init__(self, inst: Instance, num_stations: int, name: str = "SALBP_MinMax", warm_start: bool = True):
        if num_stations < 1:
            raise ValueError("num_stations deve essere ≥ 1")
        self.inst = inst
        self.M = int(num_stations)
        self.model = gp.Model(name)
        self.x = {}
        self.y = {}
        self.L = {}
        self.C = None
        self._use_warm_start = warm_start

    def build(self) -> None:
        tasks, times, preds, M = self.inst.tasks, self.inst.times, self.inst.preds, self.M
        S = range(1, M + 1)
        tot = sum(times[i] for i in tasks)

        # Variabili
        self.x = self.model.addVars([(i, s) for i in tasks for s in S], vtype=GRB.BINARY, name="x")
        self.y = self.model.addVars(tasks, vtype=GRB.INTEGER, lb=1, ub=M, name="y")
        self.L = self.model.addVars(list(S), vtype=GRB.CONTINUOUS, lb=0.0, ub=tot, name="L")
        self.C = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=tot, name="C")

        # Vincoli
        for i in tasks:
            self.model.addConstr(gp.quicksum(self.x[i, s] for s in S) == 1, name=f"assign[{i}]")
            self.model.addConstr(self.y[i] == gp.quicksum(s * self.x[i, s] for s in S), name=f"link_y[{i}]")

        for j in tasks:
            for p in preds.get(j, []):
                self.model.addConstr(self.y[j] >= self.y[p], name=f"prec[{p}->{j}]")

        for s in S:
            self.model.addConstr(self.L[s] == gp.quicksum(times[i] * self.x[i, s] for i in tasks), name=f"load[{s}]")
            self.model.addConstr(self.L[s] <= self.C, name=f"maxlink[{s}]")

        # Obiettivo
        self.model.setObjective(self.C, GRB.MINIMIZE)

        # Silenzioso di default (metti log=True in solve per vederlo)
        self.model.Params.OutputFlag = 0

        # Warm start banale: tutti i task sulla stazione M (incumbent immediato)
        if self._use_warm_start:
            for i in tasks:
                for s in S:
                    self.x[i, s].Start = 1.0 if s == M else 0.0
                self.y[i].Start = M
            for s in S:
                self.L[s].Start = float(tot if s == M else 0.0)
            self.C.Start = float(tot)

    def solve(self,
              time_limit: Optional[int] = None,
              mip_gap: Optional[float] = None,
              threads: Optional[int] = None,
              log: bool = False,
              cb=None) -> Solution:
        # Parametri del solver
        if time_limit is not None:
            self.model.Params.TimeLimit = float(time_limit)
        if mip_gap is not None:
            self.model.Params.MIPGap = float(mip_gap)
        if threads is not None:
            self.model.Params.Threads = int(threads)
        if log:
            self.model.Params.OutputFlag = 1

        # Esegui con/ senza callback
        if cb is not None:
            self.model.optimize(cb)
        else:
            self.model.optimize()

        # Status leggibile
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        st = status_map.get(self.model.Status, str(self.model.Status))

        # Leggi soluzione SOLO se c'è un incumbent
        has_incumbent = (getattr(self.model, "SolCount", 0) or 0) > 0

        C = None
        assignment: List[Tuple[str, int]] = []
        station_loads: List[Tuple[int, float]] = []

        if has_incumbent:
            C = float(self.C.X)
            for s in range(1, self.M + 1):
                station_loads.append((s, float(self.L[s].X)))
            for i in self.inst.tasks:
                for s in range(1, self.M + 1):
                    if self.x[i, s].X > 0.5:
                        assignment.append((i, s))
                        break

        return Solution(st, C, assignment, station_loads)

    def write_lp(self, path: str) -> None:
        """Esporta il modello in formato LP (debug)."""
        self.model.write(path)
