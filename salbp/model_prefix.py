# salbp/model_prefix.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import gurobipy as gp
from gurobipy import GRB
from salbp.instance import Instance

@dataclass
class Solution:
    status: str
    C: Optional[float]
    assignment: List[Tuple[str, int]]        # (task, stazione)
    station_loads: List[Tuple[int, float]]   # (stazione, carico)

class SALBPPrefixModel:
    """
    Formulazione 'prefix':
      x[i,s] ∈ {0,1}   assegnamento
      u[i,s] ∈ [0,1]   prefisso: 1 se stazione(i) ≤ s
      C      ≥ 0       massimo carico
    Vincoli:
      ∑_s x[i,s] = 1                          ∀i
      u[i,1] = x[i,1]                         ∀i
      u[i,s] = u[i,s-1] + x[i,s]              ∀i, s=2..M
      u[j,s] ≥ u[i,s]                         ∀(i→j), ∀s
      ∑_i t_i x[i,s] ≤ C                      ∀s
    Obiettivo: min C
    """
    def __init__(self, inst: Instance, num_stations: int, name: str = "SALBP_Prefix", warm_start: bool = True):
        if num_stations < 1:
            raise ValueError("num_stations deve essere ≥ 1")
        self.inst = inst
        self.M = int(num_stations)
        self.model = gp.Model(name)
        self.x: Dict[tuple, gp.Var] = {}
        self.u: Dict[tuple, gp.Var] = {}
        self.C: Optional[gp.Var] = None
        self._use_warm_start = warm_start

    def build(self) -> None:
        tasks, times, preds, M = self.inst.tasks, self.inst.times, self.inst.preds, self.M
        S = range(1, M+1)
        tot = sum(times[i] for i in tasks)

        # Variabili
        self.x = self.model.addVars([(i,s) for i in tasks for s in S], vtype=GRB.BINARY, name="x")
        self.u = self.model.addVars([(i,s) for i in tasks for s in S], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="u")
        self.C = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=tot, name="C")

        # Assegnamento unico
        for i in tasks:
            self.model.addConstr(gp.quicksum(self.x[i,s] for s in S) == 1, name=f"assign[{i}]")

        # Definizione prefissi
        for i in tasks:
            self.model.addConstr(self.u[i,1] == self.x[i,1], name=f"u_def1[{i}]")
            for s in range(2, M+1):
                self.model.addConstr(self.u[i,s] == self.u[i,s-1] + self.x[i,s], name=f"u_def[{i},{s}]")

        # Precedenze: u[j,s] ≥ u[i,s]  ∀s
     #   for (j) in tasks:
      #      for p in preds.get(j, []):
       #         for s in S:
        #            self.model.addConstr(self.u[j,s] >= self.u[p,s], name=f"prec[{p}->{j},s={s}]")

        # Precedenze: u[p,s] ≥ u[j,s] ∀s  (p è predecessore di j)
        for j in tasks:
            for p in preds.get(j, []):  # p è un predecessore di j
                for s in S:
                    self.model.addConstr(
                        self.u[p, s] >= self.u[j, s],
                        name=f"prec[{p}->{j},s={s}]"
                    )

        # Capacità per stazione
        for s in S:
            self.model.addConstr(gp.quicksum(times[i]*self.x[i,s] for i in tasks) <= self.C, name=f"cap[{s}]")

        # Obiettivo
        self.model.setObjective(self.C, GRB.MINIMIZE)

        # Silenzioso (metti log=True in solve)
        self.model.Params.OutputFlag = 0

        # Warm start banale: tutti i task sulla stazione M (fattibile)
        if self._use_warm_start:
            for i in tasks:
                for s in S:
                    self.x[i,s].Start = 1.0 if s == M else 0.0
                    # prefisso corrispondente: 0 fino a M-1, 1 a M
                    self.u[i,s].Start = 1.0 if s >= M else 0.0
            if self.C is not None:
                totM = sum(times.values())
                self.C.Start = float(totM)

    def solve(self,
              time_limit: Optional[int] = None,
              mip_gap: Optional[float] = None,
              threads: Optional[int] = None,
              log: bool = False,
              cb=None) -> Solution:
  #      if time_limit is not None: self.model.Params.TimeLimit = float(time_limit)
 #       if mip_gap   is not None: self.model.Params.MIPGap    = float(mip_gap)
#        if threads   is not None: self.model.Params.Threads   = int(threads)
 #       if log: self.model.Params.OutputFlag = 1
        # parametri
        if time_limit is not None and time_limit > 0:
            self.model.Params.TimeLimit = float(time_limit)
        if mip_gap is not None:
            self.model.Params.MIPGap = float(mip_gap)
        if threads is not None:
            self.model.Params.Threads = int(threads)
        if log:
            self.model.Params.OutputFlag = 1

        print(
            f"[DBG] TimeLimit={self.model.Params.TimeLimit}, MIPGap={self.model.Params.MIPGap}, Threads={self.model.Params.Threads}")

        if cb is not None:
            self.model.optimize(cb)
        else:
            self.model.optimize()

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        st = status_map.get(self.model.Status, str(self.model.Status))

        has_incumbent = (getattr(self.model, "SolCount", 0) or 0) > 0

        C = None
        assignment: List[Tuple[str,int]] = []
        station_loads: List[Tuple[int,float]] = []

        if has_incumbent:
            C = float(self.C.X)
            # ricava assegnamenti e carichi da x
            for s in range(1, self.M+1):
                load = 0.0
                for i in self.inst.tasks:
                    if self.x[i,s].X > 0.5:
                        assignment.append((i, s))
                        load += float(self.inst.times[i])
                station_loads.append((s, load))

        return Solution(st, C, assignment, station_loads)

    def write_lp(self, path: str):
        self.model.write(path)
