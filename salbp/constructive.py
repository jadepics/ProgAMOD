# salbp/constructive.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Iterable
import math

def _succs_from_preds(tasks: List, preds: Dict) -> Dict:
    succs = {u: set() for u in tasks}
    for j in tasks:
        pj = preds.get(j) or []
        for p in pj:
            succs.setdefault(p, set()).add(j)
    return succs

def _topo_ready(preds: Dict, assigned: Set) -> Set:
    ready = set()
    for i, P in preds.items():
        if i in assigned:
            continue
        P = P or []
        ok = True
        for p in P:
            if p not in assigned:
                ok = False
                break
        if ok:
            ready.add(i)
    return ready

def _rpw_scores(tasks: List, times: Dict, preds: Dict) -> Dict:
    """Ranked Positional Weight: ti + somma tempi dei successori (transitivi)."""
    succs = _succs_from_preds(tasks, preds)
    memo = {}
    def dfs(u):
        if u in memo:
            return memo[u]
        tot = times[u]
        for v in succs.get(u, []):
            tot += dfs(v)
        memo[u] = tot
        return tot
    return {i: dfs(i) for i in tasks}

def _lower_bound(times: Dict, S: int) -> int:
    sum_t = sum(times.values())
    mxt = max(times.values()) if times else 0
    return int(max(math.ceil(sum_t / S), mxt))

def _try_fill_with_C(tasks: List,
                     times: Dict,
                     preds: Dict,
                     S: int,
                     C: int,
                     order: str = "rpw") -> Tuple[bool, Dict, List[int]]:
    """
    Prova a costruire una soluzione con ciclo C e S stazioni.
    Ritorna (ok, station_of, loads).
    """
    # priorità
    if order == "lpt":
        priority = {i: (times[i],) for i in tasks}
        reverse = True
        def keyfun(i): return (priority[i][0], times[i])
    else:  # "rpw" default
        rpw = _rpw_scores(tasks, times, preds)
        def keyfun(i): return (rpw[i], times[i])
        reverse = True

    station_of = {}
    loads = [0] * S
    assigned = set()
    remain = set(tasks)

    s = 0  # indice 0..S-1
    while remain:
        if s >= S:
            return False, {}, []
        cap = C - loads[s]
        # candidati eligibili (pred assegnati) ordinati per priorità
        elig = [i for i in remain if all((p in assigned) for p in (preds.get(i) or []))]
        if not elig:
            # nessuno pronto: questo succede solo con errori nel DAG (ciclo)
            # o perché abbiamo finito la stazione e serve aprirne una nuova
            s += 1
            continue

        elig.sort(key=keyfun, reverse=reverse)

        placed = False
        for i in elig:
            if times[i] <= cap:
                # assegna
                station_of[i] = s + 1  # 1-based
                loads[s] += times[i]
                assigned.add(i)
                remain.remove(i)
                cap = C - loads[s]
                placed = True
                break
        if not placed:
            # nessun eligibile entra: apri nuova stazione
            s += 1

    return True, station_of, loads

def construct_targetC(instance,
                      S: int,
                      order: str = "rpw",
                      eps_step: int = 1,
                      max_tries: int = 200) -> Tuple[Dict, List[int], int]:
    """
    Costruttiva 'Target-C': parte da LB e aumenta C finché l'impacchettamento con precedenze riesce.
    order: 'rpw' (default) oppure 'lpt'
    Ritorna (station_of, loads, C_usato)
    """
    tasks = list(instance.tasks)
    times = instance.times
    preds = instance.preds

    C = _lower_bound(times, S)
    tries = 0
    while tries < max_tries:
        ok, st_of, loads = _try_fill_with_C(tasks, times, preds, S, C, order=order)
        if ok and len(st_of) == len(tasks):
            return st_of, loads, C
        # aumenta C (step piccolo ma deterministico)
        C += max(1, int(eps_step))
        tries += 1
    # fallback: restituisci l'ultimo tentativo (anche se incompleto), così l'utente vede dove fallisce
    return {}, [], C
