def plot_q1_apx_box(metrics_source, out_png: str):
    """
    Q1 – Boxplot qualità (APX = C / LB) aggregando CSV di metriche.
    Robusto: accetta run_metrics_*.csv, metrics*.csv, *summary*.csv, qualsiasi CSV che contenga
    almeno 'obj_C' e ('bound' o 'LB1') o direttamente 'apx'.
    """
    import csv
    import math
    from pathlib import Path
    import statistics as stats
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import re

    base = Path(metrics_source)

    # 🔍 Cerca più pattern, ricorsivamente
    patterns = ["run_metrics_*.csv", "metrics*.csv", "*summary*.csv", "*.metrics.csv", "*.csv"]
    cand = []
    for pat in patterns:
        cand.extend(base.rglob(pat))
    # rimuovi duplicati mantenendo l'ordine
    seen = set()
    csv_files = []
    for p in cand:
        if p.suffix.lower() == ".csv" and p not in seen:
            seen.add(p)
            csv_files.append(p)

    if not csv_files:
        raise RuntimeError(f"Nessun CSV di metriche trovato in {base}")

    def _to_float(v):
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def _guess_label(row_dict, filepath: Path):
        # 1) campo esplicito
        for k in ("algorithm", "formulation", "tag", "algo"):
            if row_dict.get(k):
                return str(row_dict[k])
        # 2) dal filename tipo run_metrics_<tag>.csv
        m = re.search(r"run_metrics_([A-Za-z0-9+\-_.]+)\.csv$", filepath.name)
        if m:
            return m.group(1)
        # 3) dal path: cartelle note
        parts = [x.lower() for x in filepath.parts]
        for candid in ("y", "prefix", "heuristic", "heuristic+vnd", "heuristic+1move", "heuristic+1move+vnd"):
            if candid in parts:
                return candid
        # fallback: nome file senza estensione
        return filepath.stem

    series = defaultdict(list)
    used_files = 0

    for f in csv_files:
        try:
            with open(f, "r", newline="", encoding="utf-8") as fh:
                rd = csv.DictReader(fh)
                has_any = False
                for r in rd:
                    has_any = True
                    label = _guess_label(r, f)

                    apx = _to_float(r.get("apx"))
                    if apx is None:
                        C = _to_float(r.get("obj_C")) or _to_float(r.get("C"))
                        bound = _to_float(r.get("bound")) or _to_float(r.get("best_bound"))
                        LB1 = _to_float(r.get("LB1"))
                        # priorità: bound, poi LB1
                        if apx is None and C is not None and bound is not None and bound > 0:
                            apx = C / bound
                        if apx is None and C is not None and LB1 is not None and LB1 > 0:
                            apx = C / LB1

                    if apx is not None and math.isfinite(apx):
                        series[label].append(apx)

                if has_any:
                    used_files += 1
        except Exception:
            # file non leggibile; ignora e continua
            continue

    if not series:
        raise RuntimeError(
            "Nessun dato valido per Q1. Verifica che i CSV contengano 'obj_C' e "
            "'bound' o 'LB1' oppure un campo 'apx'."
        )

    # Ordina le etichette per mediana (più "basso è meglio")
    order = sorted(series.keys(), key=lambda k: stats.median(series[k]))
    data = [series[k] for k in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, showmeans=True, meanline=True, widths=0.5, patch_artist=False)
    plt.title("Q1 – Qualità (APX = C / LB)")
    plt.xlabel("Algoritmo / Formulazione")
    plt.ylabel("APX (minore è meglio)")
    plt.xticks(range(1, len(order) + 1), order, rotation=0)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


