from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math, csv, os
import matplotlib.pyplot as plt
from gurobipy import GRB
import time
from types import SimpleNamespace
from pathlib import Path
import math


#classe per file di monitoraggio fa callback su Gurobi e raccoglie i dati (ProgressLogger)-> grafici
# plot_progress, plot_gap, plot_station_loads AL MOMENTO NON FUNZIONANO

@dataclass
class Snapshot:
    t: float           # tempo di runtime (s)
    best: float | None # miglior soluzione finora (C)
    bound: float | None# best bound corrente
    gap: float | None  # relative gap (0..1) se both best&bound known
    nodes: int | None       # nodi esplorati nel b&b

class ProgressLogger:
    """
    Logger di progresso per Gurobi (event-based).
    - Registra uno snapshot SOLO quando cambiano incumbent o bound.
    - Usa solo attributi consentiti nel contesto corrente.
    - Timestamp sempre strettamente crescente (niente linee verticali).
    - finalize() aggiunge comunque uno snapshot finale.
    """
    def __init__(self, min_dt=0.0, atol=1e-9, rtol=1e-9):
        self.t0 = None
        self.snaps = []               # SimpleNamespace(t, best, bound, gap)
        self._last_t = -1e9
        self._last_best = None
        self._last_bound = None
        self.min_dt = float(min_dt)   # throttling opzionale (default: nessuno)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self._last_nodes = 0

    def _changed(self, old, new):
        if old is None and new is not None: return True
        if new is None and old is not None: return True
        if old is None and new is None:     return False
        # differenza assoluta/relativa
        diff = abs(new - old)
        return diff > max(self.atol, self.rtol * (abs(old) + 1e-12))

    def __call__(self, model, where):
        import gurobipy as gp
        if self.t0 is None:
            self.t0 = time.perf_counter()

        # Leggi solo ciò che è lecito in questo "where"
        try:
            if where == gp.GRB.Callback.MIP:
                best  = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
                bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            elif where == gp.GRB.Callback.MIPSOL:
                best  = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                bound = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            elif where == gp.GRB.Callback.MIPNODE:
                best  = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)
                bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)
            else:
                return
        except Exception:
            return  # attributo non disponibile in questo evento

        # Coercion safe
        def f(x):
            try:
                x = float(x)
                if x == float("inf") or x == float("-inf"):
                    return None
                return x
            except Exception:
                return None

        best  = f(best)
        bound = f(bound)
        t = time.perf_counter() - self.t0

        # Throttling opzionale
        if t - self._last_t < self.min_dt:
            return

        # Registra solo se qualcosa è cambiato
        if not self._changed(self._last_best, best) and not self._changed(self._last_bound, bound):
            return

        # Garantisci tempo crescente
        if t <= self._last_t:
            t = self._last_t + 1e-6

        gap = None
        if best is not None and bound is not None and best != 0:
            gap = abs(best - bound) / (abs(best) + 1e-9) * 100.0

        # --- NODI ESPLORATI (solo dentro callback) ---
        nodes = self._last_nodes
        try:
            from gurobipy import GRB
            # valido in tutti gli stati MIP*; se l'istanza è risolta alla radice,
            # questo restituisce 0 o 1 a seconda della versione
            nodes = int(model.cbGet(GRB.Callback.MIP_NODCNT))
        except Exception:
             # se il where non permette cbGet o è LP, teniamo l'ultimo visto
             pass
        self._last_nodes = nodes
        # --- SNAPSHOT ---
        self.snaps.append(Snapshot(t=t, best=best, bound=bound, gap=gap, nodes=nodes))
        self._last_t = t
        self._last_best = best
        self._last_bound = bound

    def finalize(self, best=None, bound=None, nodes=None):
        """Aggiunge uno snapshot finale (t > ultimo t) con best/bound/nodes."""
        if self.t0 is None:
            self.t0 = time.perf_counter()
        t = time.perf_counter() - self.t0
        if self.snaps and t <= self._last_t:
            t = self._last_t + 1e-6  # 1 µs di padding

        gap = None
        if best is not None and bound is not None and best != 0:
            gap = abs(best - bound) / (abs(best) + 1e-9) * 100.0

        # usa i nodi passati da run.py, altrimenti ripiega sull’ultimo visto in callback
        if nodes is None:
            nodes = getattr(self, "_last_nodes", None)

        self.snaps.append(Snapshot(t=t, best=best, bound=bound, gap=gap, nodes=nodes))
        self._last_t = t
        self._last_best = best
        self._last_bound = bound
        self._last_nodes = nodes if nodes is not None else self._last_nodes


    def to_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "best", "bound", "gap", "nodes"])
            w.writeheader()
            for s in self.snaps:
                w.writerow({
                    "t": f"{s.t:.9f}",
                    "best": "" if s.best is None else s.best,
                    "bound": "" if s.bound is None else s.bound,
                    "gap": "" if s.gap is None else f"{s.gap:.6f}",
                    "nodes": "" if getattr(s, "nodes", None) in (None, "") else int(s.nodes),
                })


def plot_progress(time_s: List[float], best: List[float|None], bound: List[float|None], out_png: str):
    plt.figure()
    # filtra None con compressione semplice
    tb = [(t,v) for t,v in zip(time_s,best) if v is not None]
    td = [(t,v) for t,v in zip(time_s,bound) if v is not None]
    if tb:
        plt.plot([t for t,_ in tb], [v for _,v in tb], label="Incumbent (C)")
    if td:
        plt.plot([t for t,_ in td], [v for _,v in td], label="Best bound")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valore obiettivo")
    plt.title("Evoluzione incumbent vs bound")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_gap(time_s: List[float], gap: List[float|None], out_png: str):
    plt.figure()
    tg = [(t,g) for t,g in zip(time_s,gap) if g is not None and g>0]
    if tg:
        plt.semilogy([t for t,_ in tg], [g for _,g in tg])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Gap relativo")
    plt.title("Gap relativo vs tempo (scala log)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_incumbent_step_hist(progress_csv_path: str, out_png: str, title: str | None = None):
    """
    A1 - Istogramma degli step dell'incumbent.
    Legge progress.csv (t, best, bound, gap), estrae le diminuzioni di 'best' e
    plottizza la distribuzione degli step ΔC = C_prev - C_new (1,2,3,...).
    """
    import math
    from collections import Counter
    import matplotlib.pyplot as plt

    # Lettura CSV (pandas se disponibile, altrimenti csv.DictReader)
    best_vals = []
    try:
        import pandas as _pd
        df = _pd.read_csv(progress_csv_path)
        cand_cols = [c for c in ["best", "incumbent", "objective", "obj", "C", "best_obj"] if c in df.columns]
        if not cand_cols:
            raise RuntimeError("Colonna 'best' non trovata in progress.csv")
        col = cand_cols[0]
        best_vals = [float(x) for x in df[col].tolist() if _pd.notna(x)]
    except Exception:
        import csv
        with open(progress_csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            cand = None
            # prova a intuire la colonna migliore
            for row in r:
                if cand is None:
                    keys = list(row.keys())
                    for k in ["best", "incumbent", "objective", "obj", "C", "best_obj"]:
                        if k in keys:
                            cand = k
                            break
                    if cand is None:
                        raise RuntimeError("Colonna 'best' non trovata in progress.csv")
                try:
                    v = float(row[cand])
                    if not math.isnan(v):
                        best_vals.append(v)
                except Exception:
                    pass

    # Se non ci sono abbastanza punti, salva un grafico "vuoto" informativo
    if len(best_vals) < 2:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Nessun miglioramento registrato\n(progresso insufficiente)", ha="center", va="center")
        plt.axis("off")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close()
        return

    # Estrai gli step (Δ positivi) e discretizza a interi (C è intero nei SALBP)
    steps = []
    prev = best_vals[0]
    for cur in best_vals[1:]:
        try:
            if cur < prev - 1e-9:
                d = prev - cur
                k = int(round(d))
                if k > 0:
                    steps.append(k)
        finally:
            prev = cur

    # Se nessun miglioramento, grafico informativo
    if not steps:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "Nessun miglioramento dell'incumbent", ha="center", va="center")
        plt.axis("off")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close()
        return

    # Conta per ampiezza step
    cnt = Counter(steps)
    xs = sorted(cnt.keys())
    ys = [cnt[x] for x in xs]

    # Plot
    plt.figure(figsize=(7, 4))
    plt.bar([str(x) for x in xs], ys)
    for i, y in enumerate(ys):
        plt.text(i, y + max(ys) * 0.02, str(y), ha="center", va="bottom", fontsize=9)
    plt.xlabel("ΔC per miglioramento (tacca)")
    plt.ylabel("Occorrenze")
    if title:
        plt.title(title)
    else:
        plt.title("Istogramma step incumbent (ΔC)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

def plot_progress_milestones(progress_csv_path: str, out_png: str, lb: float | int | None = None, title: str | None = None):
    """
    A2 - Progress con milestones (anti-overlap):
      - prima soluzione
      - ultimo miglioramento
      - LB+1 / LB (se lb è passato)
    Evita sovrapposizioni usando offset, bbox e auto-allineamento vicino ai bordi.
    """
    import matplotlib.pyplot as plt
    import math
    # --- carica CSV in modo robusto
    ts, best, bnd = [], [], []
    try:
        import pandas as _pd
        df = _pd.read_csv(progress_csv_path)
        tcol = next((c for c in ["t", "time", "seconds"] if c in df.columns), None)
        bcol = next((c for c in ["best", "incumbent", "objective", "obj", "C", "best_obj"] if c in df.columns), None)
        ccol = next((c for c in ["bound", "bestbound", "objbound"] if c in df.columns), None)
        if tcol is None or bcol is None:
            raise RuntimeError("Colonne attese mancanti in progress.csv")
        ts = [float(x) for x in df[tcol].tolist()]
        best = [float(x) for x in df[bcol].tolist()]
        if ccol:
            bnd = [None if _pd.isna(x) else float(x) for x in df[ccol].tolist()]
    except Exception:
        import csv
        with open(progress_csv_path, "r", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
        def _pick(cands):
            for k in cands:
                if k in rows[0].keys():
                    return k
            return None
        tcol = _pick(["t","time","seconds"])
        bcol = _pick(["best","incumbent","objective","obj","C","best_obj"])
        ccol = _pick(["bound","bestbound","objbound"])
        if tcol is None or bcol is None:
            raise RuntimeError("Colonne attese mancanti in progress.csv")
        for row in rows:
            try:
                ts.append(float(row[tcol]))
                best.append(float(row[bcol]))
                if ccol:
                    try:    bnd.append(float(row[ccol]))
                    except: bnd.append(None)
            except Exception:
                pass

    if len(ts) < 1 or len(best) < 1:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "progress.csv vuoto/insufficiente", ha="center", va="center")
        plt.axis("off")
        if title: plt.title(title)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        return

    # --- trova milestones
    eps = 1e-9
    first_idx = None
    last_idx  = None
    prev = best[0]
    for i in range(1, len(best)):
        if best[i] < prev - eps:
            if first_idx is None:
                first_idx = i
            last_idx = i
        prev = best[i]

    lb_idx = None
    lb1_idx = None
    if lb is not None:
        for i, v in enumerate(best):
            if lb1_idx is None and v <= lb + 1 + eps:
                lb1_idx = i
            if lb_idx  is None and v <= lb + eps:
                lb_idx  = i
            if lb_idx is not None and lb1_idx is not None:
                break

    # --- plot
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.plot(ts, best, label="Incumbent (best)", linewidth=2)
    if any(x is not None for x in bnd):
        bx = [t for t, x in zip(ts, bnd) if x is not None]
        by = [x for x in bnd if x is not None]
        if bx:
            ax.plot(bx, by, linestyle="--", label="Bound", linewidth=1.5)

    xmax = max(ts) if ts else 1.0

    def _mark(idx, label, dy=8):
        if idx is None: return
        x = ts[idx]; y = best[idx]
        ax.axvline(x, linestyle=":", color="#888", linewidth=1.1, alpha=0.9)
        ax.scatter([x],[y], s=30, zorder=3)
        # automatico: se vicino al bordo destro, allinea a destra e usa offset negativo
        right_side = (x > 0.92 * xmax)
        ha = "right" if right_side else "left"
        dx = -6 if right_side else 6
        ax.annotate(f"{label}\n{round(x,3)}s",
                    xy=(x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                    clip_on=False)

    # offset verticali crescenti per separare visivamente le etichette
    _mark(first_idx, "prima soluzione",         dy=10)
    _mark(last_idx,  "ultimo miglioramento",    dy=28)
    if lb is not None:
        _mark(lb1_idx, f"best ≤ LB+1 (LB={int(lb)})", dy=46)
        _mark(lb_idx,  f"best = LB (LB={int(lb)})",   dy=64)

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("C / Bound")
    ax.set_title(title if title else "Progresso con milestones")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gap_targets(progress_csv_path: str,
                     out_png: str,
                     targets=(10.0, 5.0, 1.0, 0.5),
                     title: str | None = None):
    """
    C8 - Gap (%) vs tempo (semilog) con soglie orizzontali (10,5,1,0.5% di default).
    - Legge progress.csv (t, gap) – se gap manca lo calcola da best/bound.
    - Annota il primo istante in cui ciascuna soglia è raggiunta (gap <= target).
    - Mostra anche la media temporale del gap (AUC/T).
    """
    import math
    import matplotlib.pyplot as plt
    import csv

    # --- carica dati (robusto)
    ts, gaps = [], []

    def _try_parse_float(x):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    try:
        import pandas as _pd
        df = _pd.read_csv(progress_csv_path)
        tcol = next((c for c in ["t", "time", "seconds"] if c in df.columns), None)
        gcol = "gap" if "gap" in df.columns else None
        bcol = next((c for c in ["best", "incumbent", "objective", "obj", "C", "best_obj"] if c in df.columns), None)
        ccol = next((c for c in ["bound", "bestbound", "objbound"] if c in df.columns), None)

        if tcol is None:
            raise RuntimeError("Colonna tempo non trovata")
        ts = [float(x) for x in df[tcol].tolist()]

        if gcol:
            gaps = [_try_parse_float(x) for x in df[gcol].tolist()]
        else:
            # calcola gap da best/bound se possibile (in %)
            if bcol and ccol:
                _best = [None if _pd.isna(x) else float(x) for x in df[bcol].tolist()]
                _bnd  = [None if _pd.isna(x) else float(x) for x in df[ccol].tolist()]
                for be, bo in zip(_best, _bnd):
                    if be is not None and bo is not None and abs(be) > 1e-12:
                        gaps.append(abs(be - bo) / (abs(be) + 1e-9) * 100.0)
                    else:
                        gaps.append(None)
            else:
                gaps = [None] * len(ts)
    except Exception:
        # fallback csv.DictReader
        rows = []
        with open(progress_csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f); rows = list(r)
        if not rows:
            plt.figure(figsize=(7, 4))
            plt.text(0.5, 0.5, "progress.csv vuoto", ha="center", va="center")
            plt.axis("off")
            if title: plt.title(title)
            plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
            return

        def pick(cands):
            ks = rows[0].keys()
            for k in cands:
                if k in ks: return k
            return None

        tcol = pick(["t","time","seconds"])
        gcol = pick(["gap"])
        bcol = pick(["best","incumbent","objective","obj","C","best_obj"])
        ccol = pick(["bound","bestbound","objbound"])

        for row in rows:
            ts.append(_try_parse_float(row.get(tcol)))
            if gcol and row.get(gcol) not in ("", None):
                gaps.append(_try_parse_float(row.get(gcol)))
            else:
                be = _try_parse_float(row.get(bcol))
                bo = _try_parse_float(row.get(ccol))
                if be is not None and bo is not None and abs(be) > 1e-12:
                    gaps.append(abs(be - bo) / (abs(be) + 1e-9) * 100.0)
                else:
                    gaps.append(None)

    # Pulisci punti non validi mantenendo l'allineamento temporale
    pairs = [(t, g) for (t, g) in zip(ts, gaps) if t is not None and g is not None and g > 0]
    if not pairs:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "Gap assente o nullo", ha="center", va="center")
        plt.axis("off")
        if title: plt.title(title)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        return

    ts = [p[0] for p in pairs]
    gaps = [p[1] for p in pairs]

    # --- calcolo tempi di attraversamento soglie (prima volta)
    def first_cross(target, logy=True):
        if target is None or target <= 0:
            return None

        for i in range(1, len(gaps)):
            g0, g1 = gaps[i - 1], gaps[i]
            if g0 is None or g1 is None or g0 <= 0 or g1 <= 0:
                continue
            # attraversamento dall'alto verso il basso
            if g1 <= target < g0:
                t0, t1 = ts[i - 1], ts[i]
                if logy:
                    lg0, lg1, lgt = math.log10(g0), math.log10(g1), math.log10(target)
                    if abs(lg1 - lg0) < 1e-12:
                        return t1
                    w = (lg0 - lgt) / (lg0 - lg1)  # peso tra i due punti in spazio log(y)
                else:
                    if abs(g1 - g0) < 1e-12:
                        return t1
                    w = (g0 - target) / (g0 - g1)  # interpolazione lineare in y
                return t0 + w * (t1 - t0)

        # se l'ultimo punto è già <= target
        if gaps[-1] <= target:
            return ts[-1]
        return None

    cross_times = {t: first_cross(t, logy=True) for t in targets}


    # --- AUC/T (media temporale del gap)
    # trapezi su tempo lineare, risposta in %
    auc = 0.0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]
        if dt > 0:
            auc += 0.5 * (gaps[i] + gaps[i-1]) * dt
    avg_gap = auc / (ts[-1] - ts[0]) if ts[-1] > ts[0] else gaps[-1]

    # --- plot
    plt.figure(figsize=(8.5, 5))
    plt.semilogy(ts, gaps, linewidth=2, label="Gap (%)")

    # soglie
    for tgt in targets:
        plt.axhline(tgt, linestyle="--", linewidth=1.2)
        tt = cross_times.get(tgt, None)
        txt = f" ≤{tgt}%: "
        if tt is not None:
            plt.scatter([tt], [tgt], s=28)
            txt += f"{tt:.3f}s"
            # piccola etichetta spostata
            plt.text(tt, tgt, f" {txt}", va="bottom", ha="left", fontsize=9)
        else:
            txt += "–"
            # annotazione discreta sul margine alto
            plt.text(ts[-1], tgt, f" {txt}", va="bottom", ha="left", fontsize=9)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Gap (%) – scala log")
    if title:
        plt.title(title + f"\nAvg gap={avg_gap:.3f}%")
    else:
        plt.title(f"Gap vs tempo (Avg={avg_gap:.3f}%)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_nodes_over_time(progress_csv_path: str, out_png: str, title: str | None = None):
    """
    D10 - Nodi esplorati vs tempo con marker sui miglioramenti dell'incumbent.
    Anti-overlap:
      - etichetta solo miglioramenti distanziati nel tempo (min_dx) oppure con ΔC>=3;
      - sfasa verticalmente le etichette vicine (livelli ±12, ±24, ...);
      - usa bbox bianca semi-trasparente.
    """
    import csv
    import matplotlib.pyplot as plt

    # --- carica progress.csv
    ts, nodes, best = [], [], []
    try:
        import pandas as _pd
        df = _pd.read_csv(progress_csv_path)
        tcol = next((c for c in ["t","time","seconds"] if c in df.columns), None)
        ncol = "nodes" if "nodes" in df.columns else None
        bcol = next((c for c in ["best","incumbent","objective","obj","C","best_obj"] if c in df.columns), None)
        if not (tcol and ncol and bcol):
            raise RuntimeError("Colonne mancanti per D10")
        ts    = [float(x) for x in df[tcol].tolist()]
        nodes = [0 if _pd.isna(x) else int(x) for x in df[ncol].tolist()]
        best  = [float(x) for x in df[bcol].tolist()]
    except Exception:
        with open(progress_csv_path, "r", newline="", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for row in rr:
                try:
                    t = float(row.get("t", row.get("time", row.get("seconds",""))))
                    n = int(row.get("nodes","0") or 0)
                    b = float(row.get("best", row.get("incumbent", row.get("objective", row.get("obj", row.get("C",""))))))
                    ts.append(t); nodes.append(n); best.append(b)
                except Exception:
                    pass

    if not ts or not nodes or not best:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "Dati insufficienti per D10", ha="center", va="center")
        plt.axis("off")
        if title: plt.title(title)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        return

    # --- individua gli step di miglioramento dell'incumbent
    recs = []  # (t, nodes, ΔC)
    prev = best[0]
    for i in range(1, len(best)):
        cur = best[i]
        if cur < prev - 1e-9:
            dC = int(round(prev - cur))
            recs.append((ts[i], nodes[i], dC))
        prev = cur

    # --- plot
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    # linea nodi nel tempo
    try:
        ax.step(ts, nodes, where="post", linewidth=2)
    except Exception:
        ax.plot(ts, nodes, drawstyle="steps-post", linewidth=2)

    # annotazioni anti-overlap
    min_dx = 0.25  # secondi minimi tra etichette (evita ammucchiamento all'inizio)
    level_offsets = [12, -12, 24, -24, 36, -36]  # sfasamento verticale
    level_i = 0
    last_x_labeled = -1e9

    for (t, n, dC) in recs:
        # regola di selezione: etichetta se ΔC>=3 OPPURE sufficientemente distante dall'ultima etichetta
        if dC >= 3 or (t - last_x_labeled) >= min_dx:
            ax.scatter([t], [n], s=26, zorder=3)
            dx = 6
            dy = level_offsets[level_i % len(level_offsets)]
            ax.annotate(f"ΔC={dC}",
                        xy=(t, n), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
                        clip_on=False)
            last_x_labeled = t
            level_i += 1
        else:
            # sempre marca il punto ma senza testo
            ax.scatter([t], [n], s=14, zorder=3)

    # velocità media nodi/s (solo informativa)
    dur = max(ts) - min(ts) if ts else 0.0
    if dur > 0:
        avg = (nodes[-1] - nodes[0]) / dur
        ax.set_title(f"{title}\nAvg nodes/sec = {avg:.2f}" if title else f"Avg nodes/sec = {avg:.2f}")
    else:
        if title: ax.set_title(title)

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Nodi esplorati")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)



def plot_bestbound_vs_nodes(progress_csv_path: str, out_png: str, title: str | None = None):
    """
    B6 - Best bound vs Nodi (staircase).
    Legge progress.csv e plottizza il best bound (dual bound) in funzione dei nodi esplorati.
    Richiede la colonna 'nodes' nel CSV.
    """
    import matplotlib.pyplot as plt
    import csv

    recs = []
    try:
        import pandas as _pd
        df = _pd.read_csv(progress_csv_path)
        bcol = next((c for c in ["bound", "bestbound", "objbound"] if c in df.columns), None)
        if "nodes" in df.columns and bcol:
            for _, r in df.iterrows():
                b = None if _pd.isna(r[bcol]) else float(r[bcol])
                n = None if _pd.isna(r["nodes"]) else int(r["nodes"])
                if b is not None and n is not None:
                    recs.append((n, b))
    except Exception:
        with open(progress_csv_path, "r", newline="", encoding="utf-8") as f:
            rr = csv.DictReader(f)
            for row in rr:
                try:
                    b = row.get("bound", row.get("bestbound", row.get("objbound", "")))
                    n = row.get("nodes", "")
                    if b not in ("", None) and n not in ("", None):
                        recs.append((int(n), float(b)))
                except Exception:
                    pass

    if not recs:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "Dati insufficienti per B6 (bound/nodes)", ha="center", va="center")
        plt.axis("off")
        if title: plt.title(title)
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
        return

    # deduplica per nodo: tieni l'ULTIMO bound visto per ciascun valore di nodes
    recs.sort(key=lambda x: x[0])
    dedup = {}
    for n, b in recs:
        dedup[n] = b  # sovrascrive: rimane l'ultimo per quel n
    nodes = sorted(dedup.keys())
    bounds = [dedup[n] for n in nodes]

    plt.figure(figsize=(8, 5))
    if len(recs) == 1:
        plt.hlines(bounds[0], nodes[0]-0.5, nodes[0]+0.5, linewidth=2)
        plt.scatter([nodes[0]], [bounds[0]], s=30)
    else:
        try:
            plt.step(nodes, bounds, where="post", linewidth=2)
        except Exception:
            plt.plot(nodes, bounds, drawstyle="steps-post", linewidth=2)

    plt.xlabel("Nodi esplorati")
    plt.ylabel("Best bound")
    if title: plt.title(title)
    else: plt.title("Best bound vs Nodi")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_primal_dual_ribbon(progress_csv_path: str, out_png: str, title: str | None = None):
    """
    E14 - Primal–Dual ribbon (area tra incumbent e best bound) vs tempo.
    - Legge progress.csv (t, best, bound, gap)
    - Disegna incumbent (linea piena) e bound (tratteggiata)
    - Riempie l'area tra le due curve quando entrambe presenti
    - Mostra nel titolo:
        * gap finale (se disponibile)
        * AUC-gap normalizzato: media temporale del gap percentuale
    """
    import csv
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    # --- carica CSV in modo robusto
    ts, best, bnd, gaps = [], [], [], []
    try:
        import pandas as _pd
        df = _pd.read_csv(progress_csv_path)
        tcol = next((c for c in ["t", "time", "seconds"] if c in df.columns), None)
        bcol = next((c for c in ["best", "incumbent", "objective", "obj", "C", "best_obj"] if c in df.columns), None)
        ccol = next((c for c in ["bound", "bestbound", "objbound"] if c in df.columns), None)
        gcol = next((c for c in ["gap", "gap_perc"] if c in df.columns), None)

        if tcol is None or bcol is None:
            raise RuntimeError("Colonne attese mancanti in progress.csv")

        ts   = [float(x) for x in df[tcol].tolist()]
        best = [None if _pd.isna(x) else float(x) for x in df[bcol].tolist()]
        bnd  = [None if (ccol is None or _pd.isna(x)) else float(x) for x in (df[ccol].tolist() if ccol else [])]
        if not bnd:  # se non c'è colonna bound, crea lista di None di pari lunghezza
            bnd = [None] * len(ts)
        if gcol:
            gaps = [None if _pd.isna(x) else float(x) for x in df[gcol].tolist()]
        else:
            gaps = [None] * len(ts)
    except Exception:
        with open(progress_csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows = list(r)

        if not rows:
            plt.figure(figsize=(7, 4))
            plt.text(0.5, 0.5, "progress.csv vuoto", ha="center", va="center")
            plt.axis("off")
            if title: plt.title(title)
            plt.tight_layout(); os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=150); plt.close()
            return

        keys = rows[0].keys()
        def pick(cands):
            for k in cands:
                if k in keys: return k
            return None
        tcol = pick(["t","time","seconds"])
        bcol = pick(["best","incumbent","objective","obj","C","best_obj"])
        ccol = pick(["bound","bestbound","objbound"])
        gcol = pick(["gap","gap_perc"])

        if tcol is None or bcol is None:
            plt.figure(figsize=(7,4))
            plt.text(0.5,0.5,"Colonne t/best mancanti",ha="center",va="center"); plt.axis("off")
            if title: plt.title(title)
            plt.tight_layout(); os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=150); plt.close()
            return

        for row in rows:
            try:
                ts.append(float(row[tcol]))
            except Exception:
                ts.append(None)
            try:
                best.append(float(row[bcol]))
            except Exception:
                best.append(None)
            try:
                bnd.append(float(row[ccol])) if ccol else bnd.append(None)
            except Exception:
                bnd.append(None)
            try:
                gaps.append(float(row[gcol])) if gcol else gaps.append(None)
            except Exception:
                gaps.append(None)

    # filtra i NaN/None consistenti
    M = len(ts)
    if M == 0:
        plt.figure(figsize=(7,4))
        plt.text(0.5,0.5,"Nessun dato",ha="center",va="center"); plt.axis("off")
        if title: plt.title(title)
        plt.tight_layout(); os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True); plt.savefig(out_png, dpi=150); plt.close()
        return

    # --- AUC-gap normalizzato (media temporale del gap %)
    # usiamo trapezi tra punti successivi dove gap è noto
    auc_sum = 0.0
    Tsum    = 0.0
    for i in range(1, M):
        if ts[i-1] is None or ts[i] is None:
            continue
        dt = max(0.0, ts[i] - ts[i-1])
        g1 = gaps[i-1] if gaps[i-1] is not None else None
        g2 = gaps[i]   if gaps[i]   is not None else None
        if g1 is not None and g2 is not None:
            auc_sum += 0.5 * (g1 + g2) * dt
            Tsum    += dt
    avg_gap = (auc_sum / Tsum) if Tsum > 0 else None

    # gap finale
    final_gap = None
    if gaps and gaps[-1] is not None:
        final_gap = gaps[-1]
    elif (best and bnd and best[-1] is not None and bnd[-1] is not None and best[-1] != 0):
        final_gap = abs(best[-1] - bnd[-1]) / (abs(best[-1]) + 1e-9) * 100.0

    # --- plot
    plt.figure(figsize=(9, 5.2))
    # curve (quando presenti)
    t_plot = [t for t in ts if t is not None]
    if any(x is not None for x in best):
        best_xy = [(t, y) for t, y in zip(ts, best) if (t is not None and y is not None)]
        if best_xy:
            plt.plot([x for x,_ in best_xy], [y for _,y in best_xy], label="Incumbent (C)", linewidth=2)

    if any(x is not None for x in bnd):
        bnd_xy = [(t, y) for t, y in zip(ts, bnd) if (t is not None and y is not None)]
        if bnd_xy:
            plt.plot([x for x,_ in bnd_xy], [y for _,y in bnd_xy], linestyle="--", label="Best bound", linewidth=1.6)

    # area tra le due curve dove entrambe disponibili
    fill_xy = [(t, c, d) for t, c, d in zip(ts, best, bnd) if (t is not None and c is not None and d is not None)]
    if len(fill_xy) >= 2:
        tF = [t for t,_,_ in fill_xy]
        cF = [c for _,c,_ in fill_xy]
        dF = [d for *_,d in fill_xy]
        plt.fill_between(tF, cF, dF, alpha=0.18, step=None)

    # titolo con metriche
    parts = []
    if title:
        parts.append(title)
    if avg_gap is not None:
        parts.append(f"AUC-gap≈{avg_gap:.2f}%")
    if final_gap is not None:
        parts.append(f"gap finale={final_gap:.2f}%")
    if parts:
        plt.title("\n".join([parts[0], "  ".join(parts[1:])]))
    else:
        plt.title("Primal–Dual ribbon")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Valore obiettivo")
    if len(plt.gca().lines) > 0:
        plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()



def plot_station_loads(loads: Dict[int, float], out_png: str):
    plt.figure()
    xs = sorted(loads.keys())
    ys = [loads[s] for s in xs]
    plt.bar([str(s) for s in xs], ys)
    plt.xlabel("Stazione")
    plt.ylabel("Carico")
    plt.title("Carichi per stazione")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _load_progress_csv(path: Path):
    T, B, D, G = [], [], [], []
    if not path.exists():
        return T, B, D, G
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["t"]) if row.get("t") else None
            b = float(row["best"]) if row.get("best") not in ("", None) else None
            d = float(row["bound"]) if row.get("bound") not in ("", None) else None
            g = float(row["gap"]) if row.get("gap") not in ("", None) else None
            if t is not None:
                T.append(t); B.append(b); D.append(d); G.append(g)
    return T, B, D, G

def plot_progress_compare(y_csv: Path, p_csv: Path, out_png: str):
    Ty, By, Dy, _ = _load_progress_csv(y_csv)
    Tp, Bp, Dp, _ = _load_progress_csv(p_csv)
    plt.figure()
    if By and any(v is not None for v in By):
        plt.plot(Ty, [v for v in By], label="y – incumbent")
    if Dy and any(v is not None for v in Dy):
        plt.plot(Ty, [v for v in Dy], label="y – bound")
    if Bp and any(v is not None for v in Bp):
        plt.plot(Tp, [v for v in Bp], label="prefix – incumbent", linestyle="--")
    if Dp and any(v is not None for v in Dp):
        plt.plot(Tp, [v for v in Dp], label="prefix – bound", linestyle="--")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Valore obiettivo")
    plt.title("Incumbent & bound – confronto y vs prefix")
    if len(plt.gca().lines) > 0:
        plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_gap_compare(y_csv: Path, p_csv: Path, out_png: str):
    Ty, _, _, Gy = _load_progress_csv(y_csv)
    Tp, _, _, Gp = _load_progress_csv(p_csv)
    plt.figure()
    if Gy and any(v not in (None, 0) for v in Gy):
        plt.semilogy(Ty, [g if g and g > 0 else 1e-12 for g in Gy], label="y")
    if Gp and any(v not in (None, 0) for v in Gp):
        plt.semilogy(Tp, [g if g and g > 0 else 1e-12 for g in Gp], label="prefix")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Gap relativo (%)")
    plt.title("Gap – confronto y vs prefix")
    if len(plt.gca().lines) > 0:
        plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()