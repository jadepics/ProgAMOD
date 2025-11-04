# salbp/ls_report.py
from __future__ import annotations
from typing import List
import csv

def save_ls_csv(trace, csv_path):
    """
    Salva la trace della 1-move in CSV.
    Colonne: step,t,C,range,var,task,from,to
    """
    rows = []
    for st in trace:
        rows.append({
            "step": st.step,
            "t": st.t,
            "C": st.C,
            "range": st.rng,
            "var": st.var,
            "task": st.task,
            "from": st.s_from,
            "to": st.s_to,
        })
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["step","t","C","range","var","task","from","to"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def plot_ls_c(trace, outfile, by: str = "step"):
    """
    Line-plot di C durante la 1-move. by='step' o 'time'
    """
    import matplotlib.pyplot as plt
    if not trace:
        return
    x = [s.step for s in trace] if by == "step" else [s.t for s in trace]
    y = [s.C for s in trace]
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Step" if by == "step" else "Tempo (s)")
    plt.ylabel("C")
    plt.title("1-move: C durante la ricerca")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def plot_ls_metric(trace, outfile, metric: str = "range", by: str = "step"):
    """
    Line-plot di 'range' o 'var' durante la 1-move.
    """
    import matplotlib.pyplot as plt
    if not trace:
        return
    x = [s.step for s in trace] if by == "step" else [s.t for s in trace]
    if metric == "var":
        y = [s.var for s in trace]
        ylabel = "Varianza carichi"
        title = "1-move: varianza carichi"
    else:
        y = [s.rng for s in trace]
        ylabel = "Range carichi (max-min)"
        title = "1-move: range carichi"
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Step" if by == "step" else "Tempo (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()
