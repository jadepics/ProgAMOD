from __future__ import annotations
from typing import List, Dict
import math, statistics

def balance_metrics(loads: List[float]) -> Dict[str, float]:
    if not loads: return {}
    M = len(loads); C = max(loads); tot = sum(loads)
    LB = math.ceil(tot / M) if M>0 else 0
    rng = max(loads)-min(loads) if loads else 0
    var = statistics.pvariance(loads) if len(loads)>1 else 0.0
    util = tot/(M*C) if C>0 else 1.0
    balance_index = 1 - (M*C - tot)/(M*C) if C>0 else 1.0
    return {"C":C, "LB":LB, "gap_vs_LB":C-LB, "range":rng, "variance":var,
            "utilization":util, "balance_index":balance_index,
            "total_time":tot, "stations":M}
