# salbp/ls_one_move.py
from __future__ import annotations
from dataclasses import dataclass
from time import monotonic
from typing import Dict, List, Tuple, Optional, Set, Any

@dataclass
class LSStep:
    step: int
    t: float
    C: int
    rng: int
    var: float
    task: Any | None
    s_from: int | None
    s_to: int | None

@dataclass
class LSSolution:
    station_of: Dict[Any, int]   # task_id -> station (1..S)
    loads: List[int]             # loads[s-1]
    C: int
    trace: List[LSStep] | None = None

# ---- metriche per tie-break ----
def _range_metric(loads: List[int]) -> int:
    return (max(loads) - min(loads)) if loads else 0

def _var_metric(loads: List[int]) -> float:
    if not loads:
        return 0.0
    n = len(loads)
    mean = sum(loads) / n
    return sum((L - mean) ** 2 for L in loads) / n

# ---- util ----
def _succs_from_preds(tasks: List[Any], preds: Dict[Any, List[Any]]) -> Dict[Any, Set[Any]]:
    succs: Dict[Any, Set[Any]] = {u: set() for u in tasks}
    for j in tasks:
        pj = preds.get(j)
        if pj is None:
            pj = []
        for p in pj:
            succs.setdefault(p, set()).add(j)
    return succs

def _recompute_loads_C(station_of: Dict[Any, int], times: Dict[Any, int], S: int) -> Tuple[List[int], int]:
    loads = [0] * S
    for i, s in station_of.items():
        loads[s - 1] += int(times[i])
    C = max(loads) if loads else 0
    return loads, C

def _compute_ES_LS(station_of: Dict[Any, int],
                   preds: Dict[Any, List[Any]],
                   succs: Dict[Any, Set[Any]],
                   S: int) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    ES, LS = {}, {}
    st = station_of
    for i in st.keys():
        if preds.get(i):
            ES[i] = 1 + max(st[p] for p in preds[i])
        else:
            ES[i] = 1
        if succs.get(i):
            LS[i] = min(st[q] for q in succs[i]) - 1
        else:
            LS[i] = S
        if ES[i] > LS[i]:
            ES[i] = LS[i] = st[i]
    return ES, LS

def _critical_stations(loads: List[int], C: int) -> List[int]:
    return [s + 1 for s, L in enumerate(loads) if L == C]

def _first_improving_move(tasks_in_s: List[Any],
                          station_of: Dict[Any, int],
                          loads: List[int],
                          C: int,
                          ES: Dict[Any, int], LS: Dict[Any, int],
                          times: Dict[Any, int],
                          S: int,
                          accept_equal: bool,
                          tie_metric: str,
                          cap_C: Optional[int]) -> Optional[Tuple[Any, int]]:
    # ordina per impatto (tempi grandi prima)
    candidates = sorted(tasks_in_s, key=lambda i: times[i], reverse=True)
    # mete dalla più leggera alla più leggera
    station_order = sorted(range(1, S + 1), key=lambda s: loads[s - 1])

    cur_metric = _var_metric(loads) if tie_metric == "var" else _range_metric(loads)
    cap = cap_C if cap_C is not None else C

    for i in candidates:
        s = station_of[i]
        low, high = ES[i], LS[i]
        if low == high == s:
            continue
        for s2 in station_order:
            if s2 == s or s2 < low or s2 > high:
                continue
            # aggiorna carichi "cheap"
            new_loads = loads[:]   # S è piccolo, copia ok
            new_loads[s - 1]  -= times[i]
            new_loads[s2 - 1] += times[i]
            # rispetto del cap: nessuna stazione può superare cap
            if new_loads[s2 - 1] > cap:
                continue
            new_C = max(new_loads)

            # 1) miglioramento duro su C
            if new_C < C:
                return (i, s2)

            # 2) se ammesso, tie-break a C invariato
            if accept_equal and new_C == C:
                new_metric = _var_metric(new_loads) if tie_metric == "var" else _range_metric(new_loads)
                if new_metric < cur_metric:
                    return (i, s2)
    return None

def one_move_local_search(instance,
                          S: int,
                          station_of: Dict[Any, int],
                          time_limit: Optional[float] = 2.0,
                          accept_equal: bool = False,
                          tie_metric: str = "range",
                          cap_C: Optional[int] = None,
                          record: bool = True) -> LSSolution:
    """
    Local search 1-move con trace:
      - riduce C spostando un task per volta da stazioni critiche verso mete ammissibili;
      - opzionalmente accetta mosse con C invariato se migliorano 'range' o 'var';
      - se record=True, restituisce trace step-by-step (per CSV/plot).
    """
    tasks = list(instance.tasks)
    times = instance.times
    preds = instance.preds
    succs = _succs_from_preds(tasks, preds)

    st = dict(station_of)
    loads, C = _recompute_loads_C(st, times, S)

    trace: List[LSStep] = []
    t0 = monotonic()

    # step 0: baseline
    if record:
        trace.append(LSStep(step=0, t=0.0, C=C, rng=_range_metric(loads), var=_var_metric(loads),
                            task=None, s_from=None, s_to=None))

    step = 0
    while True:
        if time_limit is not None and (monotonic() - t0) >= time_limit:
            break
        ES, LS = _compute_ES_LS(st, preds, succs, S)
        crit = _critical_stations(loads, C)
        if not crit:
            break

        improved = False
        for s in crit:
            tasks_in_s = [i for i in tasks if st.get(i) == s]
            if not tasks_in_s:
                continue
            mv = _first_improving_move(tasks_in_s, st, loads, C, ES, LS, times, S,
                                       accept_equal=accept_equal, tie_metric=tie_metric, cap_C=cap_C)
            if mv:
                i, s2 = mv
                s1 = st[i]
                loads[s1 - 1] -= times[i]
                loads[s2 - 1] += times[i]
                st[i] = s2
                C = max(loads)
                step += 1
                if record:
                    trace.append(LSStep(step=step, t=monotonic()-t0, C=C,
                                        rng=_range_metric(loads), var=_var_metric(loads),
                                        task=i, s_from=s1, s_to=s2))
                improved = True
                break
        if not improved:
            break

    return LSSolution(station_of=st, loads=loads, C=C, trace=trace if record else None)
