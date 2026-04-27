"""Idempotent: insert a Param-count vs Overall-MSE chart cell after ALL_MODELS.

Reads each model's hparam_search.csv (best-config row for n_params) and
walk_forward.json (per-horizon MSE means), plots one dot per model with
log-scale x-axis. Baseline (no params) and missing files are skipped.

Usage: .venv/bin/python tools/insert_param_vs_mse_chart.py [--dry]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "research.ipynb"

SENTINEL = "def plot_params_vs_mse("

CELL_SOURCE = """\
def plot_params_vs_mse(results_paths, horizons=None, styles=None, log_x=True,
                        annotate=True, figsize=(10, 6)):
    \"\"\"Scatter: x=parameter count of best HP config, y=mean MSE across horizons,
    one dot per model. Reads `hparam_search.csv` (best row) and `walk_forward.json`
    from each results directory. Models with no n_params (e.g. baseline) are skipped.
    \"\"\"
    import matplotlib.pyplot as plt
    horizons = horizons or HORIZONS
    styles   = styles   or MODEL_STYLES

    points = []
    for name, wf_path in results_paths.items():
        if not os.path.exists(wf_path):
            print(f"[SKIP] {name} — {wf_path} not found")
            continue
        with open(wf_path) as f:
            agg = json.load(f)["aggregate"]
        mse_overall = float(np.mean([agg[f"MSE_t{h}_mean"] for h in horizons]))

        hp_csv = os.path.join(os.path.dirname(wf_path), "hparam_search.csv")
        n_params = None
        if os.path.exists(hp_csv):
            hp_df = pd.read_csv(hp_csv).sort_values("val_loss")
            if "n_params" in hp_df.columns and len(hp_df) > 0:
                n_params = int(hp_df.iloc[0]["n_params"])
        if n_params is None or n_params <= 0:
            print(f"[SKIP] {name} — no n_params (likely baseline)")
            continue
        points.append((name, n_params, mse_overall))

    if not points:
        print("No models with both n_params and walk-forward results.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    for name, n_params, mse in points:
        _, color = styles.get(name, ("-", None))
        ax.scatter(n_params, mse, color=color, s=90, zorder=3)
        if annotate:
            ax.annotate(name, (n_params, mse), xytext=(6, 4),
                        textcoords="offset points", fontsize=9)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Parameter count" + (" (log scale)" if log_x else ""))
    ax.set_ylabel(f"Overall MSE  (mean of t+{horizons})")
    ax.set_title("Розмір моделі vs якість прогнозу (Walk-Forward 2015–2025)")
    ax.grid(True, alpha=0.3, zorder=1)
    plt.tight_layout()
    plt.show()


plot_params_vs_mse(ALL_MODELS)
"""


def find_all_models_idx(nb) -> int:
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.source.lstrip().startswith("ALL_MODELS = {"):
            return i
    raise RuntimeError("Could not find the ALL_MODELS cell")


def already_inserted(nb) -> bool:
    return any(SENTINEL in c.source for c in nb.cells if c.cell_type == "code")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    nb = nbformat.read(NB_PATH, as_version=4)

    if already_inserted(nb):
        print("Param-vs-MSE chart cell already present — nothing to do.")
        return

    idx = find_all_models_idx(nb)
    insert_at = idx + 1
    print(f"Inserting chart cell at index {insert_at} (after ALL_MODELS at {idx}).")

    if args.dry:
        print("(dry-run — no file written)")
        return

    nb.cells.insert(insert_at, new_code_cell(CELL_SOURCE))
    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
