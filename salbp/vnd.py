# salbp/vnd.py
from __future__ import annotations
from dataclasses import dataclass
from time import monotonic
from typing import Dict, List, Tuple, Optional, Set, Any

from salbp.ls_one_move import one_move_local_search  # riuso della tua 1-move
import time

def _range_var(loads: list[float]) -> tuple[float, float]:
    if not loads:
        return 0.0, 0.0
    r = max(loads) - min(loads)
    m = sum(loads) / len(loads)
    v = sum((x - m) * (x - m) for x in loads) / len(loads)
    return r, v


# ---------- strutture risultato ----------
@dataclass
class VNDResult:
    station_of: Dict[Any, int]
    loads: List[int]
    C: int
    iters: int
    moves_1move: int
    moves_swap: int
    moves_eject: int
    trace: list | None = None

# ---------- util comuni (copiati, così non tocchiamo ls_one_move) ----------
def _succs_from_preds(tasks: List[Any], preds: Dict[Any, List[Any]]) -> Dict[Any, Set[Any]]:
    succs: Dict[Any, Set[Any]] = {u: set() for u in tasks}
    for j in tasks:
        for p in preds.get(j) or []:
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
        Pi = preds.get(i) or []
        Qi = succs.get(i) or set()
        ES[i] = 1 + max((st[p] for p in Pi), default=0)
        LS[i] = min((st[q] for q in Qi), default=S + 1) - 1
        if ES[i] > LS[i]:
            ES[i] = LS[i] = st[i]
    return ES, LS

def _range_metric(loads: List[int]) -> int:
    return (max(loads) - min(loads)) if loads else 0

def _var_metric(loads: List[int]) -> float:
    if not loads:
        return 0.0
    n = len(loads)
    m = sum(loads) / n
    return sum((L - m) ** 2 for L in loads) / n

def _crit_stations(loads: List[int], C: int) -> List[int]:
    return [s + 1 for s, L in enumerate(loads) if L == C]

# ---------- vicinato SWAP (first-improving) ----------
def _try_swap_first_improving(tasks: List[Any],
                              times: Dict[Any,int],
                              preds: Dict[Any,List[Any]],
                              succs: Dict[Any,Set[Any]],
                              S: int,
                              station_of: Dict[Any,int],
                              loads: List[int],
                              C: int,
                              accept_equal: bool,
                              tie_metric: str) -> Optional[Tuple[Any, Any]]:
    ES, LS = _compute_ES_LS(station_of, preds, succs, S)
    # priorità: scambia tra una stazione critica e una leggera
    crit = _crit_stations(loads, C)
    if not crit:
        return None
    stations_by_lightness = sorted(range(1, S+1), key=lambda s: loads[s-1])
    # metrica corrente (per C invariato)
    cur_metric = _var_metric(loads) if tie_metric == "var" else _range_metric(loads)

    for s in crit:  # dalla più pesante
        tasks_s = [i for i in tasks if station_of[i] == s]
        tasks_s.sort(key=lambda i: times[i], reverse=True)  # i pesanti prima
        for r in stations_by_lightness:
            if r == s:
                continue
            tasks_r = [j for j in tasks if station_of[j] == r]
            if not tasks_r:
                continue
            for i in tasks_s:
                for j in tasks_r:
                    # fattibilità precedenze dopo swap
                    if not (ES[i] <= r <= LS[i] and ES[j] <= s <= LS[j]):
                        continue
                    # nuovi carichi
                    Ls = loads[s-1] - times[i] + times[j]
                    Lr = loads[r-1] - times[j] + times[i]
                    new_loads = list(loads)
                    new_loads[s-1] = Ls
                    new_loads[r-1] = Lr
                    new_C = max(new_loads)
                    # accetta se riduce C
                    if new_C < C:
                        return (i, j)
                    # tie-break a C invariato
                    if accept_equal and new_C == C:
                        new_metric = _var_metric(new_loads) if tie_metric == "var" else _range_metric(new_loads)
                        if new_metric < cur_metric:
                            return (i, j)
    return None

# ---------- vicinato EJECTION "leggera" (via cap C-1) ----------
def _try_ejection_via_cap(instance,
                          S: int,
                          station_of: Dict[Any,int],
                          times: Dict[Any,int],
                          cap_target: int,
                          time_slice: float,
                          tie_metric: str) -> Optional[Tuple[Dict[Any,int], List[int], int]]:
    """
    Prova a 'spingere' tramite 1-move con cap_C = cap_target (tipicamente C-1).
    Se riesce a creare una soluzione con C < C_attuale, restituisce la nuova tripla.
    """
    ls = one_move_local_search(instance,
                               S=S,
                               station_of=station_of,
                               time_limit=time_slice,
                               accept_equal=True,
                               tie_metric=tie_metric,
                               cap_C=cap_target,
                               record=False)
    # Se è riuscita a scendere sotto l'attuale cap, la consideriamo miglioramento
    # (in pratica C' <= cap_target < C)
    return (ls.station_of, ls.loads, ls.C) if ls.C <= cap_target else None

# ---------- VND ----------
def vnd_search(instance,
               S: int,
               station_of: Dict[Any,int],
               time_limit: float = 5.0,
               accept_equal: bool = False,
               tie_metric: str = "range",
               use_swap: bool = True,
               use_ejection: bool = True,
               record: bool = False) -> VNDResult:
    """
    VND: 1-move finché migliora -> se bloccato prova swap -> se ancora bloccato prova ejection (cap C-1).
    Appena trovi un miglioramento con un attrezzo, riparti da 1-move.
    """
    tasks = list(instance.tasks)
    times = instance.times
    preds = instance.preds
    succs = _succs_from_preds(tasks, preds)

    st = dict(station_of)
    loads, C = _recompute_loads_C(st, times, S)

    start = monotonic()
    iters = 0
    m1 = ms = me = 0  # contatori mosse

    t0 = time.perf_counter()
    trace = [] if record else None
    step_idx = 0

    def _emit(phase: str, move: str, oldC: int, C: int, loads: list[float]):
        """Registra una riga trace (solo se record=True)."""
        if trace is None:
            return
        t = time.perf_counter() - t0
        rng, var = _range_var(loads)
        trace.append({
            "step": step_idx,  # indice “miglioramento”
            "t": round(t, 9),  # secondi
            "C": int(C),
            "dC": int(oldC - C),
            "phase": phase,  # "1move" | "swap" | "eject" | "init"
            "move": move,  # descrizione breve, opzionale
            "range": float(rng),
            "var": float(var),
        })

    # snapshot iniziale (C corrente)
    if record:
        # supponiamo che qui tu abbia già definito C e loads attuali
        _emit("init", "", C, C, loads)

    while True:
        if time_limit is not None and (monotonic() - start) >= time_limit:
            break
        iters += 1
        improved_any = False

        # --- N1: 1-move descent completa ---
        t_left = max(0.0, time_limit - (monotonic() - start)) if time_limit is not None else None
        if t_left is not None and t_left <= 0.0:
            break
        ls = one_move_local_search(instance,
                                   S=S,
                                   station_of=st,
                                   time_limit=t_left * 0.5 if t_left is not None else 1.0,
                                   accept_equal=accept_equal,
                                   tie_metric=tie_metric,
                                   cap_C=None,
                                   record=False)
        if ls.C < C:
            oldC = C
            st, loads, C = ls.station_of, ls.loads, ls.C
            improved_any = True
            m1 += 1
            if record:
                step_idx += 1
                _emit("1move", "descent", oldC, C, loads)
            continue

        # --- N2: SWAP (se abilitato) ---
        if use_swap:
            #ES, LS = _compute_ES_LS(st, preds, succs, S)
            sw = _try_swap_first_improving(tasks, times, preds, succs, S, st, loads, C,
                                           accept_equal=accept_equal, tie_metric=tie_metric)
            if sw is not None:
                i, j = sw
                s = st[i];
                r = st[j]
                oldC = C
                # applica swap
                loads[s - 1] = loads[s - 1] - times[i] + times[j]
                loads[r - 1] = loads[r - 1] - times[j] + times[i]
                st[i], st[j] = r, s
                C = max(loads)
                improved_any = True
                ms += 1
                if record:
                    step_idx += 1
                    _emit("swap", f"{i}<->{j}", oldC, C, loads)
                continue

        # --- N3: EJECTION via cap (se abilitato) ---
        if use_ejection:
            t_left = max(0.0, time_limit - (monotonic() - start)) if time_limit is not None else None
            if t_left is not None and t_left <= 0.0:
                break
            trial = _try_ejection_via_cap(instance, S, st, times, cap_target=C-1,
                                          time_slice=t_left * 0.5 if t_left is not None else 1.0,
                                          tie_metric=tie_metric)
            if trial is not None:
                oldC = C
                st, loads, C = trial
                improved_any = True
                me += 1
                if record:
                    step_idx += 1
                    _emit("eject", f"cap->{oldC - 1}", oldC, C, loads)
                continue

        if not improved_any:
            break

    return VNDResult(station_of=st, loads=loads, C=C, iters=iters,
                     moves_1move=m1, moves_swap=ms, moves_eject=me, trace=trace)
