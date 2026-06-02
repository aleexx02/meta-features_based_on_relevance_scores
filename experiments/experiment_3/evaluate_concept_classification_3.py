# evaluate_concept_classification_3.py
# ==============================================================================
# Experiment 3: Real-World Stream Evaluation (INSECTS)
#
# Evaluates ABFS-based raw score meta-features (v2.0) on three INSECTS
# real-world data streams and compares against Komorniczak statistical
# meta-features (pre-extracted by E1_extract_real.py).
#
# Streams:
#   - INSECTS-abrupt_imbalanced_norm    : 236 chunks, 33 features, 2 concepts
#   - INSECTS-gradual_imbalanced_norm   : 236 chunks, 33 features, 6 concepts
#   - INSECTS-incremental_imbalanced_norm: 236 chunks, 33 features, 6 concepts
#
# chunk_size = 300 (matches Komorniczak et al.)
#
# Key differences from synthetic experiments (1a-1c):
#   - No warmup skip: all 236 windows included in the meta-dataset.
#     ABFS still updates for all windows but we cannot afford to skip
#     any on such short streams. The first few windows will have noisier
#     relevance scores but skipping would lose critical concept transitions
#     (e.g. INSECTS-gradual has a drift at chunk 9).
#   - Single stream per name (no random seed replications — the stream is fixed)
#   - n_features = 33 (real stream dimensionality)
#   - Concept labels come from manually annotated ground truth drift indices
#
# Two evaluation protocols:
#   - Shuffled CV (Experiment 1a protocol): RepeatedStratifiedKFold(2, 5)
#   - Prequential (Experiment 1c protocol): test-then-train per window
#
# Outputs saved to results/experiment_3/:
#   cv_abfs_ba_{stream_name}.npy    shape: (n_folds, n_clfs)
#   cv_abfs_f1_{stream_name}.npy    shape: (n_folds, n_clfs)
#   cv_abfs_kappa_{stream_name}.npy shape: (n_folds, n_clfs)
#   cv_komor_ba_{stream_name}.npy   shape: (n_folds, n_clfs)
#   cv_komor_f1_{stream_name}.npy   shape: (n_folds, n_clfs)
#   cv_komor_kappa_{stream_name}.npy shape: (n_folds, n_clfs)
#   preq_abfs_ba_{stream_name}.npy   shape: (n_windows, n_clfs)
#   preq_abfs_f1_{stream_name}.npy   shape: (n_windows, n_clfs)
#   preq_abfs_kappa_{stream_name}.npy shape: (n_windows, n_clfs)
#   preq_komor_ba_{stream_name}.npy   shape: (n_windows, n_clfs)
#   preq_komor_f1_{stream_name}.npy   shape: (n_windows, n_clfs)
#   preq_komor_kappa_{stream_name}.npy shape: (n_windows, n_clfs)
#
#   Figures saved to results/experiment_3/figures/:
#     heatmap_comparison_komorniczak_ABFS_cv_{stream_name}.png
#     heatmap_comparison_komorniczak_ABFS_preq_{stream_name}.png
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
from metafeatures.mf_extraction import extract_metafeatures_raw
from classifier_sweep_komor import run_classifier_sweep, BASE_CLFS
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from plot_results import plot_heatmap_balanced_accuracy_comparison_exp2 as plot_heatmap

# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

KOMOR_REPO        = os.path.expanduser('~/code_komor/data')
INSECTS_STREAM_DIR = os.path.join(KOMOR_REPO, 'real_streams_pr')
INSECTS_GT_DIR     = os.path.join(KOMOR_REPO, 'real_streams_gt')
KOMOR_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'external', 'komorniczak', 'results', 'real')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_3', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  CONFIGURATION
# ============================================================
CHUNK_SIZE     = 300   # matches Komorniczak et al.
N_FEATURES     = 33    # INSECTS stream dimensionality after NPYParser
WARMUP_WINDOWS = 0     # no warmup skip on short real streams — see header comment

INSECTS_STREAMS = [
    'INSECTS-abrupt_imbalanced_norm',
    'INSECTS-gradual_imbalanced_norm',
    'INSECTS-incremental_imbalanced_norm',
]

# number of concepts per stream (from ground truth)
N_CONCEPTS = {
    'INSECTS-abrupt_imbalanced_norm':     2,
    'INSECTS-gradual_imbalanced_norm':    6,
    'INSECTS-incremental_imbalanced_norm': 6,
}

# ============================================================
#  HELPERS
# ============================================================

def load_gt(stream_name):
    """Load ground truth drift chunk indices."""
    path = os.path.join(INSECTS_GT_DIR, f'{stream_name}.npy')
    return np.load(path)


def get_concept_labels(drift_chunks, n_total_chunks):
    """
    Assign concept labels to each chunk based on drift chunk indices.
    Concept increments at each drift boundary.
    """
    labels  = np.zeros(n_total_chunks, dtype=int)
    concept = 0
    for i in range(n_total_chunks):
        if i in drift_chunks:
            concept += 1
        labels[i] = concept
    return labels


def already_done(stream_name):
    """Return True if all 12 result files for this stream exist."""
    prefixes = [
        'cv_abfs_ba',    'cv_abfs_f1',    'cv_abfs_kappa',
        'cv_komor_ba',   'cv_komor_f1',   'cv_komor_kappa',
        'preq_abfs_ba',  'preq_abfs_f1',  'preq_abfs_kappa',
        'preq_komor_ba', 'preq_komor_f1', 'preq_komor_kappa',
    ]
    return all(
        os.path.exists(os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy'))
        for p in prefixes
    )


def save(array, prefix, stream_name):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    np.save(path, array)
    return path


# ============================================================
#  ABFS META-FEATURE EXTRACTION (raw scores v2.0)
#
#  No warmup skip: all windows included in meta-dataset.
#  ABFS still updates for every window but we cannot afford
#  to discard windows on a 236-chunk stream.
# ============================================================

def extract_abfs_metafeatures(stream_name, drift_chunks):
    """
    Run ABFS on one INSECTS stream and return raw score meta-features (v2.0).

    Returns
    -------
    X : np.ndarray, shape (n_valid_chunks, N_FEATURES)
    y : np.ndarray, shape (n_valid_chunks,)
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

    meta_features  = []
    concept_labels = []
    window_counter = 0

    for chunk_idx in range(100000):

        # get concept label for this chunk
        concept = int(np.sum(drift_chunks <= chunk_idx))

        # get chunk
        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break

        # skip single-class chunks (same as Komorniczak)
        if len(np.unique(y_chunk)) < 2:
            window_counter += 1
            continue

        # update ABFS instance by instance
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

        wt = abfs.relevance_scores()
        abfs.pop_drift_count()

        # no warmup skip — include all windows
        meta_features.append(extract_metafeatures_raw(wt))
        concept_labels.append(concept)

        window_counter += 1

    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    return X, y


# ============================================================
#  MAIN SWEEP
# ============================================================

for stream_name in INSECTS_STREAMS:
    n_concepts = N_CONCEPTS[stream_name]

    print(f"\n{'='*70}")
    print(f"Stream: {stream_name}")
    print(f"Concepts: {n_concepts}, random baseline: {1/n_concepts:.3f}")
    print(f"{'='*70}")

    if already_done(stream_name):
        print("  All 12 result files exist — skipping.")
        continue

    # load ground truth drift chunks
    drift_chunks = load_gt(stream_name)
    print(f"  Drift chunks: {drift_chunks}")

    # --- ABFS meta-feature extraction ---
    print("\n  Extracting ABFS meta-features...")
    X_abfs, y_abfs = extract_abfs_metafeatures(stream_name, drift_chunks)
    print(f"  ABFS meta-dataset: {X_abfs.shape}, concepts: {np.unique(y_abfs)}")

    # --- Komorniczak meta-features (pre-extracted by E1_extract_real.py) ---
    komor_path = os.path.join(KOMOR_RESULTS_DIR,
                              f'komor_real_{stream_name}_statistical.npy')
    if not os.path.exists(komor_path):
        print(f"  Komorniczak features not found: {komor_path}")
        print("  Run external/komorniczak/E1_extract_real.py first.")
        continue

    komor_data = np.load(komor_path)
    X_komor    = komor_data[:, :-1]
    y_komor    = komor_data[:, -1].astype(int)
    X_komor[np.isnan(X_komor)] = 0
    X_komor[np.isinf(X_komor)] = 0
    print(f"  Komor meta-dataset: {X_komor.shape}, concepts: {np.unique(y_komor)}")

    # ---- shuffled CV (Experiment 1a protocol) ----
    print("\n  Running shuffled CV...")
    np.random.seed(1233)
    mba_a, _, rba_a, _, _, rf1_a, _, _, rk_a = run_classifier_sweep(
        X_abfs, y_abfs, shuffle_seed=None)
    np.random.seed(1233)
    mba_k, _, rba_k, _, _, rf1_k, _, _, rk_k = run_classifier_sweep(
        X_komor, y_komor, shuffle_seed=None)

    print(f"  [CV ABFS ] " + "  ".join(
        f"{n}={mba_a[i]:.3f}" for i, (n, _) in enumerate(BASE_CLFS)))
    print(f"  [CV Komor] " + "  ".join(
        f"{n}={mba_k[i]:.3f}" for i, (n, _) in enumerate(BASE_CLFS)))

    # ---- prequential (Experiment 1c protocol) ----
    print("\n  Running prequential...")
    mba_a2, _, tba_a, _, _, tf1_a, _, _, tk_a = run_prequential_sweep(
        X_abfs, y_abfs)
    mba_k2, _, tba_k, _, _, tf1_k, _, _, tk_k = run_prequential_sweep(
        X_komor, y_komor)

    print(f"  [Preq ABFS ] " + "  ".join(
        f"{n}={tba_a[-1, i]:.3f}"
        for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))
    print(f"  [Preq Komor] " + "  ".join(
        f"{n}={mba_k2[i]:.3f}"
        for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    # ---- save results ----
    # CV shape:          (n_folds, n_clfs) = (10, 5)
    #   single stream, no replications — not stacked like synthetic experiments
    # Prequential shape: (n_windows, n_clfs)
    save(rba_a,  'cv_abfs_ba',    stream_name)
    save(rf1_a,  'cv_abfs_f1',    stream_name)
    save(rk_a,   'cv_abfs_kappa', stream_name)
    save(rba_k,  'cv_komor_ba',   stream_name)
    save(rf1_k,  'cv_komor_f1',   stream_name)
    save(rk_k,   'cv_komor_kappa',stream_name)

    save(tba_a,  'preq_abfs_ba',    stream_name)
    save(tf1_a,  'preq_abfs_f1',    stream_name)
    save(tk_a,   'preq_abfs_kappa', stream_name)
    save(tba_k,  'preq_komor_ba',   stream_name)
    save(tf1_k,  'preq_komor_f1',   stream_name)
    save(tk_k,   'preq_komor_kappa',stream_name)

    print(f"\n  Saved 12 result files for {stream_name}")

    # ---- CV heatmap ----
    # rba_a shape: (n_folds, n_clfs) — mean/std/median over folds
    plot_heatmap(
        mean_ba_abfs    = np.mean(rba_a,   axis=0),
        std_ba_abfs     = np.std(rba_a,    axis=0),
        median_ba_abfs  = np.median(rba_a, axis=0),
        mean_ba_komor   = np.mean(rba_k,   axis=0),
        std_ba_komor    = np.std(rba_k,    axis=0),
        median_ba_komor = np.median(rba_k, axis=0),
        BASE_CLFS       = BASE_CLFS,
        drift_type      = stream_name,
        n_concepts      = n_concepts,
        tag             = f'cv_{stream_name}',
        FIGURES_DIR     = FIGURES_DIR,
    )

    # ---- prequential heatmap ----
    # tba_a shape: (n_windows, n_clfs) — use final window value
    # no replications so std=0 and median=mean
    final_abfs  = tba_a[-1, :]   # (n_clfs,)
    final_komor = tba_k[-1, :]
    plot_heatmap(
        mean_ba_abfs    = final_abfs,
        std_ba_abfs     = np.zeros_like(final_abfs),
        median_ba_abfs  = final_abfs,
        mean_ba_komor   = final_komor,
        std_ba_komor    = np.zeros_like(final_komor),
        median_ba_komor = final_komor,
        BASE_CLFS       = BASE_CLFS_PREQUENTIAL,
        drift_type      = stream_name,
        n_concepts      = n_concepts,
        tag             = f'preq_{stream_name}',
        FIGURES_DIR     = FIGURES_DIR,
    )

print("\nExperiment 3 complete.")