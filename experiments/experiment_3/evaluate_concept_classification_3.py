# evaluate_concept_classification_3.py
# ==============================================================================
# Experiment 3: Real-World Stream Evaluation (INSECTS)
#
# Evaluates ALL THREE ABFS meta-feature versions on three INSECTS
# real-world data streams and compares against ALL 9 Komorniczak measure
# groups (pre-extracted by E1_extract_real.py).
#
# ABFS versions:
#   - v1.1 aggstats    : 8-dim aggregate statistics
#   - v2.0 raw scores  : n_features-dim normalized relevance vector
#   - v2.1 raw+temporal: v2.0 + delta_mean + cosine_sim (n_features+2-dim)
#
# Streams:
#   - INSECTS-abrupt_imbalanced_norm    : 236 chunks, 33 features, 2 concepts
#   - INSECTS-gradual_imbalanced_norm   : 236 chunks, 33 features, 6 concepts
#   - INSECTS-incremental_imbalanced_norm: 236 chunks, 33 features, 6 concepts
#
# chunk_size = 300 (matches Komorniczak et al.)
#
# Key differences from synthetic experiments:
#   - No warmup skip: all windows included in the meta-dataset.
#   - Single stream per name (no random seed replications)
#   - n_features = 33 (real stream dimensionality)
#   - Concept labels from manually annotated ground truth drift indices
#   - Prequential evaluation only (no batch CV)
#   - All 9 Komorniczak measure groups evaluated
#
# Outputs saved to results/experiment_3/:
#   abfs_y_{stream_name}.npy                         shape: (n_windows,)
#   preq_abfs_{version}_ba_{stream_name}.npy         shape: (n_windows, n_clfs)
#   preq_abfs_{version}_f1_{stream_name}.npy         shape: (n_windows, n_clfs)
#   preq_abfs_{version}_kappa_{stream_name}.npy      shape: (n_windows, n_clfs)
#   preq_komor_{measure}_ba_{stream_name}.npy        shape: (n_windows, n_clfs)
#   preq_komor_{measure}_f1_{stream_name}.npy        shape: (n_windows, n_clfs)
#   preq_komor_{measure}_kappa_{stream_name}.npy     shape: (n_windows, n_clfs)
#
#   Figures saved to results/experiment_3/figures/:
#     heatmap_combined_exp3_{stream_name}.png
#       Left:  9 Komorniczak measure groups (rows) x classifiers (cols)
#       Right: 3 ABFS versions (rows)             x classifiers (cols)
#
# Run from project root:
#   python experiments/experiment_3/evaluate_concept_classification_3.py
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

import strlearn as sl
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
)
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL

# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

KOMOR_REPO         = os.path.expanduser('~/code_komor/data')
INSECTS_STREAM_DIR = os.path.join(KOMOR_REPO, 'real_streams_pr')
INSECTS_GT_DIR     = os.path.join(KOMOR_REPO, 'real_streams_gt')
KOMOR_RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'external', 'komorniczak', 'results', 'real')
RESULTS_DIR        = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
FIGURES_DIR        = os.path.join(PROJECT_ROOT, 'results', 'experiment_3', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  CONFIGURATION
# ============================================================
CHUNK_SIZE     = 300
N_FEATURES     = 33
WARMUP_WINDOWS = 0

INSECTS_STREAMS = [
    'INSECTS-abrupt_imbalanced_norm',
    'INSECTS-gradual_imbalanced_norm',
    'INSECTS-incremental_imbalanced_norm',
]

N_CONCEPTS = {
    'INSECTS-abrupt_imbalanced_norm':      2,
    'INSECTS-gradual_imbalanced_norm':     6,
    'INSECTS-incremental_imbalanced_norm': 6,
}

MEASURES = [
    'clustering',
    'complexity',
    'concept',
    'general',
    'info-theory',
    'itemset',
    'landmarking',
    'model-based',
    'statistical',
]

# all three ABFS versions to evaluate
ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {
    'aggstats':     'Aggstats (v1.1)',
    'raw':          'Raw scores (v2.0)',
    'raw_temporal': 'Raw + temporal (v2.1)',
}

EXPECTED_N_CLFS = len(BASE_CLFS_PREQUENTIAL)

# ============================================================
#  HELPERS
# ============================================================

def load_gt(stream_name):
    path = os.path.join(INSECTS_GT_DIR, f'{stream_name}.npy')
    return np.load(path)


def already_done_abfs(stream_name, version):
    """Return True if all 3 result files for this ABFS version exist
    with correct shapes."""
    prefixes = [
        f'preq_abfs_{version}_ba',
        f'preq_abfs_{version}_f1',
        f'preq_abfs_{version}_kappa',
    ]
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_N_CLFS:
            print(f"  WARNING: {path} has unexpected shape {arr.shape} — will rerun.")
            return False
    # also check y labels file
    if not os.path.exists(os.path.join(RESULTS_DIR, f'abfs_y_{stream_name}.npy')):
        return False
    return True


def already_done_komor(stream_name, measure):
    prefixes = [
        f'preq_komor_{measure}_ba',
        f'preq_komor_{measure}_f1',
        f'preq_komor_{measure}_kappa',
    ]
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_N_CLFS:
            print(f"  WARNING: {path} has unexpected shape {arr.shape} — will rerun.")
            return False
    return True


def save(array, prefix, stream_name):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    np.save(path, array)
    return path


def print_label_dist(name, y):
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  {name} label distribution: {dist}")
    return dist


# ============================================================
#  COMBINED HEATMAP
#  Mirrors plot_heatmap_balanced_accuracy_comparison from plot_results.py:
#    Left:  Komorniczak — 9 measure groups as rows
#    Right: ABFS        — 3 versions as rows
# ============================================================

def plot_combined_heatmap(stream_name, n_concepts, tba_versions):
    """
    Parameters
    ----------
    stream_name  : str
    n_concepts   : int
    tba_versions : dict  {version: np.ndarray (n_windows, n_clfs)}
                   final cumulative BA per ABFS version
    """
    clf_names = [n for n, _ in BASE_CLFS_PREQUENTIAL]
    n_clfs    = len(clf_names)

    # ---- build Komorniczak matrix (9 x n_clfs) ----
    komor_matrix = np.full((len(MEASURES), n_clfs), np.nan)
    for m_id, measure in enumerate(MEASURES):
        path = os.path.join(RESULTS_DIR,
                            f'preq_komor_{measure}_ba_{stream_name}.npy')
        if not os.path.exists(path):
            print(f"  Combined heatmap: missing {measure} — skipping plot.")
            return
        komor_matrix[m_id, :] = np.load(path)[-1, :]

    # ---- build ABFS matrix (3 x n_clfs) ----
    abfs_matrix = np.full((len(ABFS_VERSIONS), n_clfs), np.nan)
    for v_id, version in enumerate(ABFS_VERSIONS):
        if version in tba_versions:
            abfs_matrix[v_id, :] = tba_versions[version][-1, :]
        else:
            path = os.path.join(RESULTS_DIR,
                                f'preq_abfs_{version}_ba_{stream_name}.npy')
            if not os.path.exists(path):
                print(f"  Combined heatmap: missing ABFS {version} — skipping plot.")
                return
            abfs_matrix[v_id, :] = np.load(path)[-1, :]

    random_baseline = 1.0 / n_concepts

    fig, axes = plt.subplots(
        1, 2, figsize=(26, max(5, len(MEASURES) * 0.75)),
        gridspec_kw={'width_ratios': [3, 1.5]}
    )

    # ---- left: Komorniczak ----
    ax = axes[0]
    ax.imshow(komor_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i, measure in enumerate(MEASURES):
        for j in range(n_clfs):
            val = komor_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=txt_color)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(MEASURES)))
    ax.set_yticklabels(MEASURES, fontsize=10)
    ax.set_title('Komorniczak meta-features — balanced accuracy', fontsize=12)

    # ---- right: ABFS ----
    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    abfs_row_labels = [ABFS_LABELS[v] for v in ABFS_VERSIONS]
    for i in range(len(ABFS_VERSIONS)):
        for j in range(n_clfs):
            val = abfs_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=txt_color)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(ABFS_VERSIONS)))
    ax.set_yticklabels(abfs_row_labels, fontsize=10)
    ax.set_title('ABFS meta-features — balanced accuracy', fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f'Komorniczak vs ABFS — {stream_name}\n'
        f'Prequential evaluation  |  random baseline = {random_baseline:.3f}',
        fontsize=13
    )
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR,
                            f'heatmap_combined_exp3_{stream_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined heatmap: {out_path}")


# ============================================================
#  ABFS META-FEATURE EXTRACTION — all 3 versions in one pass
# ============================================================

def extract_abfs_metafeatures(stream_name, drift_chunks):
    """
    Run ABFS on one INSECTS stream and return all three meta-feature
    versions extracted in a single pass.

    Returns
    -------
    X_aggstats    : np.ndarray, shape (n_valid_chunks, 8)
    X_raw         : np.ndarray, shape (n_valid_chunks, N_FEATURES)
    X_raw_temporal: np.ndarray, shape (n_valid_chunks, N_FEATURES + 2)
    y             : np.ndarray, shape (n_valid_chunks,)
    """
    stream_path = os.path.join(INSECTS_STREAM_DIR, f'{stream_name}.npy')
    stream = sl.streams.NPYParser(stream_path,
                                  chunk_size=CHUNK_SIZE,
                                  n_chunks=100000)

    abfs = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = CHUNK_SIZE,
        class_window_size    = CHUNK_SIZE,
    )

    mf_aggstats     = []
    mf_raw          = []
    mf_raw_temporal = []
    concept_labels  = []
    wt_prev         = None
    window_counter  = 0

    for chunk_idx in range(100000):

        concept = int(np.sum(drift_chunks <= chunk_idx))

        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break

        if len(np.unique(y_chunk)) < 2:
            window_counter += 1
            continue

        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()       # resets internal counter
        t_since     = abfs.time_since_drift        # direct attribute access

        # v1.1 aggstats
        mf_aggstats.append(
            extract_metafeatures(wt, wt_prev, drift_count, t_since)
        )

        # v2.0 raw scores
        mf_raw.append(extract_metafeatures_raw(wt))

        # v2.1 raw + temporal
        mf_raw_temporal.append(
            extract_metafeatures_raw_temporal(wt, wt_prev)
        )

        concept_labels.append(concept)
        wt_prev = wt
        window_counter += 1

    def clean(arr):
        a = np.array(arr, dtype=float)
        a[np.isnan(a)] = 0
        a[np.isinf(a)] = 0
        return a

    return (
        clean(mf_aggstats),
        clean(mf_raw),
        clean(mf_raw_temporal),
        np.array(concept_labels),
    )


# ============================================================
#  MAIN SWEEP
# ============================================================

for stream_name in INSECTS_STREAMS:
    n_concepts = N_CONCEPTS[stream_name]

    print(f"\n{'='*70}")
    print(f"Stream: {stream_name}")
    print(f"Concepts: {n_concepts}, random baseline: {1/n_concepts:.3f}")
    print(f"{'='*70}")

    drift_chunks = load_gt(stream_name)
    print(f"  Drift chunks: {drift_chunks}")

    # ---- ABFS: check which versions still need to be run ----
    versions_needed = [v for v in ABFS_VERSIONS
                       if not already_done_abfs(stream_name, v)]

    tba_versions = {}   # {version: tba array} — populated below

    if not versions_needed:
        print("  All ABFS versions already done — loading from disk.")
        y_abfs = np.load(os.path.join(RESULTS_DIR, f'abfs_y_{stream_name}.npy'))
        print_label_dist('ABFS (loaded)', y_abfs)
        for version in ABFS_VERSIONS:
            tba_versions[version] = np.load(
                os.path.join(RESULTS_DIR,
                             f'preq_abfs_{version}_ba_{stream_name}.npy')
            )
    else:
        print(f"\n  Extracting ABFS meta-features (all 3 versions in one pass)...")
        X_aggstats, X_raw, X_raw_temporal, y_abfs = \
            extract_abfs_metafeatures(stream_name, drift_chunks)

        print(f"  aggstats shape:     {X_aggstats.shape}")
        print(f"  raw shape:          {X_raw.shape}")
        print(f"  raw_temporal shape: {X_raw_temporal.shape}")
        print_label_dist('ABFS', y_abfs)

        # save y labels once
        save(y_abfs, 'abfs_y', stream_name)

        X_by_version = {
            'aggstats':     X_aggstats,
            'raw':          X_raw,
            'raw_temporal': X_raw_temporal,
        }

        for version in ABFS_VERSIONS:
            if already_done_abfs(stream_name, version):
                print(f"  ABFS [{version}] already done — loading from disk.")
                tba_versions[version] = np.load(
                    os.path.join(RESULTS_DIR,
                                 f'preq_abfs_{version}_ba_{stream_name}.npy')
                )
                continue

            print(f"\n  Running prequential (ABFS {version})...")
            X = X_by_version[version]
            _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(X, y_abfs)

            save(tba, f'preq_abfs_{version}_ba',    stream_name)
            save(tf1, f'preq_abfs_{version}_f1',    stream_name)
            save(tk,  f'preq_abfs_{version}_kappa', stream_name)

            tba_versions[version] = tba
            print(f"  [Preq ABFS {version}] " + "  ".join(
                f"{n}={tba[-1, i]:.3f}"
                for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    # ---- Komorniczak: loop over all 9 measure groups ----
    for measure in MEASURES:

        print(f"\n  --- Measure group: {measure} ---")

        if already_done_komor(stream_name, measure):
            print(f"  Komorniczak [{measure}] already done — skipping.")
            continue

        komor_path = os.path.join(
            KOMOR_RESULTS_DIR,
            f'komor_real_{stream_name}_{measure}.npy'
        )
        if not os.path.exists(komor_path):
            print(f"  Komorniczak features not found: {komor_path}")
            print(f"  Run E1_extract_real.py first. Skipping {measure}.")
            continue

        komor_data = np.load(komor_path)
        X_komor    = komor_data[:, :-1]
        y_komor    = komor_data[:, -1].astype(int)
        X_komor[np.isnan(X_komor)] = 0
        X_komor[np.isinf(X_komor)] = 0

        print(f"  Komor meta-dataset: {X_komor.shape}, concepts: {np.unique(y_komor)}")

        y_komor_dist = print_label_dist(f'Komor [{measure}]', y_komor)
        y_abfs_dist  = print_label_dist('ABFS', y_abfs)
        if y_komor_dist != y_abfs_dist:
            print(f"  WARNING: label distributions differ — check alignment.")

        print(f"  Running prequential (Komorniczak [{measure}])...")
        _, _, tba_k, _, _, tf1_k, _, _, tk_k = run_prequential_sweep(X_komor, y_komor)

        save(tba_k, f'preq_komor_{measure}_ba',    stream_name)
        save(tf1_k, f'preq_komor_{measure}_f1',    stream_name)
        save(tk_k,  f'preq_komor_{measure}_kappa', stream_name)
        print(f"  Saved Komorniczak [{measure}] results.")

        print(f"  [Preq Komor {measure}] " + "  ".join(
            f"{n}={tba_k[-1, i]:.3f}"
            for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    # ---- combined heatmap: one per stream, after all results are ready ----
    plot_combined_heatmap(
        stream_name  = stream_name,
        n_concepts   = n_concepts,
        tba_versions = tba_versions,
    )

print("\nExperiment 3 complete.")