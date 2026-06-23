#  evaluate_concept_classification_4.py
# ==============================================================================
# Experiment 4: recurring SEA / STAGGER, chunk_size x n_drifts grid
#
# Grid: generator {sea, stagger} x drift {sudden, gradual} x
#       chunk_size {100, 200, 500, 1000} x n_drifts {1, 3, 7, 15}
#       ->  64 cells.
# 500,000 instances per cell. Concepts CYCLE through the generator's
# set (segment i -> concept i mod n_concepts), so recurrence grows with
# n_drifts: at n_drifts=1 only 2 concepts appear once each (no
# recurrence); from n_drifts=7 on the concepts genuinely recur. SEA has
# 4 concepts, STAGGER 3 -- fixed labelling rules, so a drift-count axis
# necessarily reuses concepts (recurrence is forced by the generators).
#
# The concept label per window is the GENERATIVE concept id, so a
# recurring concept carries the SAME label each time it appears -- that
# repetition is the whole experiment. (If it were positional, recurrence
# would be invisible.)
#
# Cells come from streams.generate_synthetic_streams.exp4_specs(), the
# single source of truth. Streams are regenerated per cell per
# replication seed (nothing pre-saved, like Experiment 2). One plain
# serial script: loops all 64 cells x N_REPLICATIONS seeds, ABFS (3
# versions) + Komorniczak (9 measures), prequential sweep, stacked into
# (n_reps, n_windows, n_clfs). Re-running skips completed cells; pymfe
# is cached per (cell, measure, seed) so interrupted runs resume cheaply.
#
# NOTE: this grid is large (64 cells x 5 reps x 12 feature sets). It is
# the heaviest evaluation in the project -- expect a long first run; the
# pymfe cache makes subsequent runs fast.
#
# WARMUP_WINDOWS = 0. ABFS window size ties to the cell's chunk_size.
#
# Output: results/experiment_4/  (shape (n_reps, n_windows, n_clfs))
#   preq_abfs_{version}_ba_{cell}.npy   / _f1_ / _kappa_
#   preq_komor_{measure}_ba_{cell}.npy  / _f1_ / _kappa_
#   concept_labels_{cell}.npy           (rep-0 concept labels)
#   where {cell} = {gen}_chunk{cs}_ndrift{nd}_{drift}
# Figures: results/experiment_4/figures/
#   heatmap_comparison_komorniczak_ABFS_preq_exp4_{cell}.png
#
# Run from project root:
#   python experiments/experiment_4/evaluate_concept_classification_4.py
# ==============================================================================

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from pymfe.mfe import MFE
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
)
from streams.generate_synthetic_streams import exp4_specs, SEED
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

KOMOR_CACHE_DIR = os.path.join(PROJECT_ROOT, 'external', 'komorniczak',
                               'results', 'synthetic_recurring')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
FIGURES_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_4', 'figures')
os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs(FIGURES_DIR,     exist_ok=True)
os.makedirs(KOMOR_CACHE_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================
WARMUP_WINDOWS = 0
N_REPLICATIONS = 5

SPECS = exp4_specs()

_seed_rng = np.random.RandomState(SEED)
RANDOM_STATES = [SEED] + [int(s) for s in
                                 _seed_rng.randint(100, 100000, N_REPLICATIONS - 1)]
print(f"Replication seeds: {RANDOM_STATES}")
print(f"Experiment 4 grid: {len(SPECS)} cells x {N_REPLICATIONS} reps")

MEASURES = [
    'clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical',
]

ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {
    'aggstats':     'Aggstats (v1.1)',
    'raw':          'Raw scores (v2.0)',
    'raw_temporal': 'Raw + temporal (v2.1)',
}

EXPECTED_N_CLFS = len(BASE_CLFS_PREQUENTIAL)
EXP_TAG = 'exp4'


# ============================================================
#  HELPERS
# ============================================================

def _shape_ok(arr):
    return (arr.ndim == 3 and arr.shape[0] == N_REPLICATIONS
            and arr.shape[2] == EXPECTED_N_CLFS)


def already_done_abfs(cell, version):
    for p in (f'preq_abfs_{version}_ba', f'preq_abfs_{version}_f1',
              f'preq_abfs_{version}_kappa'):
        path = os.path.join(RESULTS_DIR, f'{p}_{cell}.npy')
        if not os.path.exists(path) or not _shape_ok(np.load(path)):
            return False
    return os.path.exists(os.path.join(RESULTS_DIR, f'concept_labels_{cell}.npy'))


def already_done_komor(cell, measure):
    for p in (f'preq_komor_{measure}_ba', f'preq_komor_{measure}_f1',
              f'preq_komor_{measure}_kappa'):
        path = os.path.join(RESULTS_DIR, f'{p}_{cell}.npy')
        if not os.path.exists(path) or not _shape_ok(np.load(path)):
            return False
    return True


def save(array, prefix, cell):
    np.save(os.path.join(RESULTS_DIR, f'{prefix}_{cell}.npy'), array)


def chunk_iter(data, concept_per_chunk, chunk_size):
    X_full = data[:, :-1]; y_full = data[:, -1]
    for ci in range(len(concept_per_chunk)):
        s = ci * chunk_size; e = s + chunk_size
        yield ci, X_full[s:e], y_full[s:e], int(concept_per_chunk[ci])


# ============================================================
#  COMBINED HEATMAP -- Komorniczak vs ABFS, mean final window over reps
# ============================================================

def plot_combined_heatmap(cell, n_concepts, chunk_size, tba_versions):
    clf_names = [n for n, _ in BASE_CLFS_PREQUENTIAL]
    n_clfs    = len(clf_names)

    komor_matrix = np.full((len(MEASURES), n_clfs), np.nan)
    for m_id, measure in enumerate(MEASURES):
        path = os.path.join(RESULTS_DIR, f'preq_komor_{measure}_ba_{cell}.npy')
        if not os.path.exists(path):
            print(f"  Heatmap: missing {measure} -- skipping."); return
        komor_matrix[m_id, :] = np.mean(np.load(path)[:, -1, :], axis=0)

    abfs_matrix = np.full((len(ABFS_VERSIONS), n_clfs), np.nan)
    for v_id, version in enumerate(ABFS_VERSIONS):
        if version in tba_versions:
            abfs_matrix[v_id, :] = np.mean(tba_versions[version][:, -1, :], axis=0)
        else:
            path = os.path.join(RESULTS_DIR, f'preq_abfs_{version}_ba_{cell}.npy')
            if not os.path.exists(path):
                print(f"  Heatmap: missing ABFS {version} -- skipping."); return
            abfs_matrix[v_id, :] = np.mean(np.load(path)[:, -1, :], axis=0)

    random_baseline = 1.0 / n_concepts
    fig, axes = plt.subplots(1, 2, figsize=(26, max(5, len(MEASURES) * 0.75)),
                             gridspec_kw={'width_ratios': [3, 1.5]})
    ax = axes[0]
    ax.imshow(komor_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(len(MEASURES)):
        for j in range(n_clfs):
            val = komor_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if val > 0.6 else 'black')
    ax.set_xticks(range(n_clfs)); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES, fontsize=10)
    ax.set_title('Komorniczak meta-features -- balanced accuracy', fontsize=12)

    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(len(ABFS_VERSIONS)):
        for j in range(n_clfs):
            val = abfs_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if val > 0.6 else 'black')
    ax.set_xticks(range(n_clfs)); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(ABFS_VERSIONS)))
    ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
    ax.set_title('ABFS meta-features -- balanced accuracy', fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f'Komorniczak vs ABFS -- {cell}\n'
                 f'Prequential | chunk_size={chunk_size} | mean of {N_REPLICATIONS} reps | '
                 f'random baseline = {random_baseline:.3f}', fontsize=13)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR,
                       f'heatmap_comparison_komorniczak_ABFS_preq_{EXP_TAG}_{cell}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved heatmap: {out}")


# ============================================================
#  EXTRACTION
# ============================================================

def extract_abfs(data, concept_per_chunk, n_features, chunk_size):
    abfs = ABFS_match(n_features=n_features, categorical_features=[],
                      accuracy_window_size=chunk_size, class_window_size=chunk_size)
    mf_agg, mf_raw, mf_rt, labels = [], [], [], []
    wt_prev = None
    for ci, X_chunk, y_chunk, concept in chunk_iter(data, concept_per_chunk, chunk_size):
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt = abfs.relevance_scores(); dc = abfs.pop_drift_count(); ts = abfs.time_since_drift
        if ci >= WARMUP_WINDOWS:
            mf_agg.append(extract_metafeatures(wt, wt_prev, dc, ts))
            mf_raw.append(extract_metafeatures_raw(wt))
            mf_rt.append(extract_metafeatures_raw_temporal(wt, wt_prev))
            labels.append(concept)
        wt_prev = wt
    def clean(a):
        a = np.array(a, dtype=float); a[np.isnan(a)] = 0; a[np.isinf(a)] = 0; return a
    return clean(mf_agg), clean(mf_raw), clean(mf_rt), np.array(labels)


def extract_komor(cell, measure, seed, data, concept_per_chunk, chunk_size):
    cache = os.path.join(KOMOR_CACHE_DIR, f'komor_{cell}_{measure}_seed{seed}.npy')
    if os.path.exists(cache):
        return np.load(cache)
    mfe = MFE(groups=[measure], suppress_warnings=True)
    out = []
    for ci, X_chunk, y_chunk, concept in chunk_iter(data, concept_per_chunk, chunk_size):
        if ci < WARMUP_WINDOWS:
            continue
        try:
            mfe.fit(X_chunk, y_chunk)
            _, ft = mfe.extract(suppress_warnings=True)
            ft = np.array(ft, dtype=float); ft[np.isnan(ft)] = 0; ft[np.isinf(ft)] = 0
        except Exception as e:
            print(f"    chunk {ci}: pymfe failed ({e}) -- skipping."); continue
        out.append(np.append(ft, concept))
    result = np.array(out)
    np.save(cache, result)
    return result


# ============================================================
#  MAIN -- loop over all cells
# ============================================================

for spec in SPECS:
    cell       = spec['name']
    n_features = spec['n_features']
    chunk_size = spec['chunk_size']
    builder    = spec['builder']

    print(f"\n{'='*70}\nCell: {cell}  "
          f"(gen={spec['gen_name']}, chunk_size={chunk_size}, "
          f"n_drifts={spec['n_drifts']}, order={spec['order']}, "
          f"concepts={spec['n_concepts']})\n{'='*70}")

    abfs_needed  = [v for v in ABFS_VERSIONS if not already_done_abfs(cell, v)]
    komor_needed = [m for m in MEASURES if not already_done_komor(cell, m)]
    tba_versions = {}
    n_concepts   = spec['n_concepts']

    if not abfs_needed and not komor_needed:
        print("  Cell already complete -- loading ABFS BA for heatmap.")
        for v in ABFS_VERSIONS:
            tba_versions[v] = np.load(os.path.join(RESULTS_DIR, f'preq_abfs_{v}_ba_{cell}.npy'))
        plot_combined_heatmap(cell, n_concepts, chunk_size, tba_versions)
        continue

    abfs_acc  = {v: {'ba': [], 'f1': [], 'kappa': []} for v in ABFS_VERSIONS}
    komor_acc = {m: {'ba': [], 'f1': [], 'kappa': []} for m in MEASURES}

    for rep_id, seed in enumerate(RANDOM_STATES):
        print(f"\n  --- rep {rep_id+1}/{N_REPLICATIONS} (seed={seed}) ---")
        data, cpc = builder(seed)

        X_agg, X_raw, X_rt, y = extract_abfs(data, cpc, n_features, chunk_size)
        X_by_version = {'aggstats': X_agg, 'raw': X_raw, 'raw_temporal': X_rt}
        if rep_id == 0:
            n_concepts = len(np.unique(y))
            print(f"    metafeatures: agg={X_agg.shape} raw={X_raw.shape} rt={X_rt.shape}; "
                  f"concepts={n_concepts}")
            save(y, 'concept_labels', cell)

        for version in ABFS_VERSIONS:
            _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(X_by_version[version], y)
            abfs_acc[version]['ba'].append(tba)
            abfs_acc[version]['f1'].append(tf1)
            abfs_acc[version]['kappa'].append(tk)

        for measure in MEASURES:
            kd = extract_komor(cell, measure, seed, data, cpc, chunk_size)
            Xk = kd[:, :-1]; yk = kd[:, -1].astype(int)
            Xk[np.isnan(Xk)] = 0; Xk[np.isinf(Xk)] = 0
            _, _, tba_k, _, _, tf1_k, _, _, tk_k = run_prequential_sweep(Xk, yk)
            komor_acc[measure]['ba'].append(tba_k)
            komor_acc[measure]['f1'].append(tf1_k)
            komor_acc[measure]['kappa'].append(tk_k)

    for version in ABFS_VERSIONS:
        tba_arr = np.array(abfs_acc[version]['ba'])
        save(tba_arr,                              f'preq_abfs_{version}_ba',    cell)
        save(np.array(abfs_acc[version]['f1']),    f'preq_abfs_{version}_f1',    cell)
        save(np.array(abfs_acc[version]['kappa']), f'preq_abfs_{version}_kappa', cell)
        tba_versions[version] = tba_arr
        mf = np.mean(tba_arr[:, -1, :], axis=0)
        print(f"  [ABFS {version}] " + "  ".join(
            f"{n}={mf[i]:.3f}" for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    for measure in MEASURES:
        save(np.array(komor_acc[measure]['ba']),    f'preq_komor_{measure}_ba',    cell)
        save(np.array(komor_acc[measure]['f1']),    f'preq_komor_{measure}_f1',    cell)
        save(np.array(komor_acc[measure]['kappa']), f'preq_komor_{measure}_kappa', cell)

    plot_combined_heatmap(cell, n_concepts, chunk_size, tba_versions)

print("\nExperiment 4 complete.")