# salbp/greedy_bestfit.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Set
import math
from dataclasses import dataclass

@dataclass
class GreedyBFResult:
    station_of: Dict[Any, int]
    loads: List[int]
    C: int

def _succs_from_preds(tasks: List[Any], preds: Dict[Any, List[Any]]) -> Dict[Any, Set[Any]]:
    succs: Dict[Any, Set[Any]] = {u: set() for u in tasks}
    for j in tasks:
        for p in preds.get(j) or []:
            succs.setdefault(p, set()).add(j)
    return succs

def _greedy_bestfit_fixedC(instance, S: int, C: int) -> Tuple[bool, Dict[Any, int], List[int]]:
    """
    Station-oriented greedy con priorità Best-Fit a C fissato (SALBP-1 come test di fattibilità).
    Ritorna (feasible, station_of, loads). Se feasible=False, mapping vuoto.
    """
    tasks = list(instance.tasks)
    times = instance.times
    preds = instance.preds

    # quick infeasibility: task singolo più lungo del ciclo
    if any(times[i] > C for i in tasks):
        return False, {}, []

    succs = _succs_from_preds(tasks, preds)

    # indegree iniziale (quanti predecessori mancanti)
    indeg = {i: len(preds.get(i) or []) for i in tasks}
    ready = [i for i in tasks if indeg[i] == 0]

    station_of: Dict[Any, int] = {}
    loads = [0] * S
    assigned = 0

    # riempi stazione per stazione
    for s in range(1, S + 1):
        residual = C
        # mentre riesco a piazzare qualcosa in questa stazione
        while True:
            # candidati: task pronti che entrano nel residuo
            candidates = [i for i in ready if times[i] <= residual]
            if not candidates:
                break

            # Best-Fit = minimizza il residuo (equivale a massimizzare t_i entro il residuo)
            # tie-breaker leggero: più successori, poi ID stabile
            candidates.sort(key=lambda i: (residual - times[i], -(len(succs.get(i) or [])), str(i)))
            i = candidates[0]

            # assegna i alla stazione s
            station_of[i] = s
            loads[s - 1] += int(times[i])
            residual -= int(times[i])
            assigned += 1

            # togli i dai ready
            ready.remove(i)
            # aggiorna readiness dei successori
            for j in succs.get(i, []):
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)

        # passa alla prossima stazione

    feasible = (assigned == len(tasks))
    if not feasible:
        return False, {}, []
    return True, station_of, loads

def construct_targetC_bestfit(instance, S: int, eps_step: int = 1, max_tries: int = 2000) -> Tuple[Dict[Any,int], List[int], int]:
    """
    Target-C usando Best-Fit come regola di priorità.
    Parte da LB = ceil(sum t / S) e aumenta C di eps_step finché la greedy trova una packing fattibile in S stazioni.
    """
    times = instance.times
    sum_t = sum(times.values())
    LB = math.ceil(sum_t / S)

    # prova C = LB, LB+eps_step, ... finché fattibile
    for k in range(max_tries):
        C = LB + k * eps_step
        ok, st_map, loads = _greedy_bestfit_fixedC(instance, S, C)
        if ok:
            return st_map, loads, C

    # fallimento: nessuna packing trovata nei tentativi
    return {}, [], 0
