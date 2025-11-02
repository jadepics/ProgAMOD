# salbp/metrics.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
import math, csv

def balance_metrics(loads: List[float]) -> Dict[str, float]:
    if not loads:
        return {"C": None, "LB": None, "gap_vs_LB": None, "range": None, "variance": None,
                "utilization": None, "balance_index": None, "total_time": 0.0, "stations": 0}
    C = max(loads)
    total = float(sum(loads))
    M = len(loads)
    LB_sum = math.ceil(total / M) if M > 0 else 0
    LB_max = max(loads) if loads else 0
    LB = max(LB_sum, LB_max)
    gap_vs_LB = (C - LB) if (C is not None and LB is not None) else None
    rng = max(loads) - min(loads)
    mean = total / M
    var = sum((l - mean) ** 2 for l in loads) / M if M > 0 else 0.0
    util = total / (C * M) if (C and M and C > 0) else None
    return {
        "C": float(C),
        "LB": float(LB),
        "gap_vs_LB": float(gap_vs_LB) if gap_vs_LB is not None else None,
        "range": float(rng),
        "variance": float(var),
        "utilization": float(util) if util is not None else None,
        "balance_index": float(util) if util is not None else None,
        "total_time": total,
        "stations": M,
    }

def build_run_summary(*, formulation: str, instance_name: str, loads: List[float],
                      obj_C: Optional[float], bound: Optional[float],
                      runtime: float, nodes: int) -> Dict[str, object]:
    m = balance_metrics(loads)
    row = {
        "formulation": formulation,
        "instance": instance_name,
        "obj_C": obj_C,
        "best_bound": bound,
        "runtime_sec": runtime,
        "nodes": nodes,
    }
    row.update(m)
    return row

def write_dict_csv_one(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_run_metrics_per_formulation(base_out: Path, formulation: str, row: Dict[str, object]) -> None:
    """Salva le metriche nella cartella della formulazione e in un file aggregato globale."""
    # 1) nella cartella specifica (outputs/y/metrics.csv o outputs/prefix/metrics.csv)
    write_dict_csv_one(base_out / formulation / "metrics.csv", row)
    # 2) nel file aggregato (outputs/metrics_all.csv) con il campo formulation
    write_dict_csv_one(base_out / "metrics_all.csv", row)
