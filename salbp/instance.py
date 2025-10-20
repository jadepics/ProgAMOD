from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import csv

#questa è la classe che carica e valida i dati dal file excel

@dataclass
class Instance:
    tasks: List[str]
    times: Dict[str, int]
    preds: Dict[str, List[str]]  # j -> lista dei predecessori

    @staticmethod
    def from_csv(path: str) -> "Instance":
        tasks, times, preds = [], {}, {}
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or not {"task_id","time"}.issubset(r.fieldnames):
                raise ValueError("CSV deve avere 'task_id' e 'time' (predecessors opzionale).")
            for row in r:
                tid = str(row["task_id"]).strip()
                if not tid:
                    continue
                t = int(round(float(row["time"])))  # se hai decimali, scala tutti i tempi (es. *10)
                pr_raw = (row.get("predecessors") or "").strip()
                pr = []
                if pr_raw:
                    for sep in ["|", ",", ";", " "]:
                        if sep in pr_raw:
                            pr = [p.strip() for p in pr_raw.split(sep) if p.strip()]
                            break
                    if not pr:
                        pr = [pr_raw]
                if tid in times:
                    raise ValueError(f"task_id duplicato: {tid}")
                tasks.append(tid); times[tid]=t; preds[tid]=pr
        _validate_instance(tasks, times, preds)
        return Instance(tasks, times, preds)

def _validate_instance(tasks: List[str], times: Dict[str,int], preds: Dict[str,List[str]]):
    if any(times[t] <= 0 for t in tasks):
        raise ValueError("Tutti i tempi devono essere > 0.")
    ids = set(tasks)
    for j in tasks:
        for p in preds.get(j, []):
            if p not in ids:
                raise ValueError(f"Predecessore '{p}' di '{j}' inesistente.")
    # Kahn per aciclicità
    indeg = {u:0 for u in tasks}; succ = {u:[] for u in tasks}
    for j in tasks:
        for p in preds.get(j, []):
            indeg[j]+=1; succ[p].append(j)
    Q = [u for u in tasks if indeg[u]==0]; seen=0
    while Q:
        u = Q.pop(); seen+=1
        for v in succ[u]:
            indeg[v]-=1
            if indeg[v]==0: Q.append(v)
    if seen != len(tasks):
        raise ValueError("Il grafo di precedenze ha almeno un ciclo.")
