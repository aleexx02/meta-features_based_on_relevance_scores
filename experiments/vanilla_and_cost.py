# vanilla_and_cost.py
# ==============================================================================
# SHARED add-on used by ALL experiments (1c, 2, 3, 4, 5). Adds two things
# without re-running the expensive ABFS / Komorniczak extraction:
#
#   1. VANILLA BASELINE — trivial meta-features: per-feature mean + std of each
#      raw window, run through the SAME prequential sweep as ABFS/Komorniczak.
#      The only difference from the real approaches is the meta-features, so it
#      is a fair floor. Saved as preq_vanilla_{ba,f1,kappa}_{tag}.npy.
#
#   2. EXTRACTION COST — wall-clock time + peak python-heap memory for ABFS and
#      Komorniczak extraction, timed on ONE representative cell per experiment.
#      Written to extraction_cost_{exp}.csv.
#
# Everything experiment-specific (how to build a stream's windows, warmup,
# concept labels, the cell list, the results dir) is passed IN via an
# ExperimentSpec. The core logic below is identical for every experiment.
#
# Each experiment provides a `window_provider(cell, seed) -> (windows, labels)`
# where `windows` is a list of raw instance blocks (each n_inst x n_feat) AFTER
# the experiment's warmup, and `labels` is the aligned concept label per window.
# ==============================================================================

import os
import time
import tracemalloc
import csv
import numpy as np

from classifier_sweep_prequential import run_prequential_sweep


# ------------------------------------------------------------------
#  VANILLA FEATURES
# ------------------------------------------------------------------
def vanilla_features_from_windows(windows):
    """windows: list of raw instance blocks (each n_inst x n_feat).
    Returns (n_windows, 2*n_feat): per-window [mean || std]."""
    feats = []
    for w in windows:
        Xc = np.asarray(w, dtype=float)
        feats.append(np.concatenate([Xc.mean(axis=0), Xc.std(axis=0)]))
    X = np.array(feats, dtype=float)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X


# ------------------------------------------------------------------
#  COST TIMER
# ------------------------------------------------------------------
def timed(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, dt, peak / 1e6  # MB


# ------------------------------------------------------------------
#  EXPERIMENT SPEC — the only thing that varies between experiments
# ------------------------------------------------------------------
class ExperimentSpec:
    """
    name            : 'exp2', 'exp3', ...
    results_dir     : absolute path to results/experiment_X/
    cells           : list of tags (strings) to process
    seeds           : list of replication seeds (1 for real streams)
    window_provider : fn(cell, seed) -> (windows, labels)
                      windows already post-warmup; labels aligned per window.
    cost_abfs       : fn(cell, seed) -> n_windows   (times ABFS extraction)
    cost_komor      : fn(cell, seed) -> n_windows   (times Komor extraction)
    n_features_of   : fn(cell) -> int               (for the cost table)
    """
    def __init__(self, name, results_dir, cells, seeds,
                 window_provider, cost_abfs, cost_komor, n_features_of):
        self.name = name
        self.results_dir = results_dir
        self.cells = cells
        self.seeds = seeds
        self.window_provider = window_provider
        self.cost_abfs = cost_abfs
        self.cost_komor = cost_komor
        self.n_features_of = n_features_of


def _save(results_dir, array, prefix, tag):
    np.save(os.path.join(results_dir, f'{prefix}_{tag}.npy'), array)


def _vanilla_done(results_dir, tag):
    return all(os.path.exists(os.path.join(results_dir, f'preq_vanilla_{m}_{tag}.npy'))
               for m in ('ba', 'f1', 'kappa'))


# ------------------------------------------------------------------
#  CORE RUNNER — identical for every experiment
# ------------------------------------------------------------------
def run_experiment(spec, do_vanilla=True, do_cost=True):
    os.makedirs(spec.results_dir, exist_ok=True)
    cost_rows = []
    cost_done = False

    for cell in spec.cells:
        # -------- VANILLA (all cells, all seeds) --------
        if do_vanilla:
            if _vanilla_done(spec.results_dir, cell):
                print(f"  [{spec.name}] vanilla exists: {cell}")
            else:
                ba_reps, f1_reps, k_reps = [], [], []
                for seed in spec.seeds:
                    windows, labels = spec.window_provider(cell, seed)
                    Xv = vanilla_features_from_windows(windows)
                    yv = np.asarray(labels)
                    (m_ba, s_ba, t_ba, m_f1, s_f1, t_f1,
                     m_k, s_k, t_k) = run_prequential_sweep(Xv, yv)
                    ba_reps.append(t_ba); f1_reps.append(t_f1); k_reps.append(t_k)
                # real streams have 1 seed -> shape (n_windows, n_clfs);
                # synthetic have 5 -> (n_reps, n_windows, n_clfs). Match the
                # convention the rest of each experiment already uses.
                ba_out = np.array(ba_reps) if len(ba_reps) > 1 else ba_reps[0]
                f1_out = np.array(f1_reps) if len(f1_reps) > 1 else f1_reps[0]
                k_out  = np.array(k_reps)  if len(k_reps)  > 1 else k_reps[0]
                _save(spec.results_dir, ba_out, 'preq_vanilla_ba',    cell)
                _save(spec.results_dir, f1_out, 'preq_vanilla_f1',    cell)
                _save(spec.results_dir, k_out,  'preq_vanilla_kappa', cell)
                print(f"  [{spec.name}] vanilla saved: {cell}")

        # -------- COST (one representative cell) --------
        if do_cost and not cost_done:
            seed = spec.seeds[0]
            n_win, t_abfs, mem_abfs = timed(spec.cost_abfs, cell, seed)
            _,     t_kom,  mem_kom  = timed(spec.cost_komor, cell, seed)
            nf = spec.n_features_of(cell)
            for method, dt, mem in [('abfs', t_abfs, mem_abfs),
                                    ('komorniczak', t_kom, mem_kom)]:
                cost_rows.append(dict(
                    experiment=spec.name, tag=cell, method=method,
                    n_features=nf, n_windows=n_win,
                    time_s=round(dt, 3),
                    ms_per_window=round(1000 * dt / max(n_win, 1), 3),
                    peak_mb=round(mem, 1)))
            cost_done = True
            print(f"  [{spec.name}] cost measured on: {cell}")

    if do_cost and cost_rows:
        path = os.path.join(spec.results_dir, f'extraction_cost_{spec.name}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
            w.writeheader(); w.writerows(cost_rows)
        print(f"  [{spec.name}] cost -> {path}")