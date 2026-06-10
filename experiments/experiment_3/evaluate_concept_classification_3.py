# evaluate_concept_classification_3.py
# ==============================================================================
# Experiment 3: Real-World Stream Evaluation (INSECTS)
#
# Evaluates ABFS-based raw score meta-features (v2.0) on three INSECTS
# real-world data streams and compares against ALL 9 Komorniczak measure
# groups (pre-extracted by E1_extract_real.py).
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
#   abfs_y_{stream_name}.npy                     shape: (n_windows,)
#   preq_abfs_ba_{stream_name}.npy               shape: (n_windows, n_clfs)
#   preq_abfs_f1_{stream_name}.npy               shape: (n_windows, n_clfs)
#   preq_abfs_kappa_{stream_name}.npy            shape: (n_windows, n_clfs)
#   preq_komor_{measure}_ba_{stream_name}.npy    shape: (n_windows, n_clfs)
#   preq_komor_{measure}_f1_{stream_name}.npy    shape: (n_windows, n_clfs)
#   preq_komor_{measure}_kappa_{stream_name}.npy shape: (n_windows, n_clfs)
#
#   Figures saved to results/experiment_3/figures/
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
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from plot_results import plot_heatmap_balanced_accuracy_comparison_exp2 as plot_heatmap

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

EXPECTED_N_CLFS = len(BASE_CLFS_PREQUENTIAL)

# ============================================================
#  HELPERS
# ============================================================

def load_gt(stream_name):
    """Load ground truth drift chunk indices."""
    path = os.path.join(INSECTS_GT_DIR, f'{stream_name}.npy')
    return np.load(path)


def already_done_abfs(stream_name):
    """Return True if all ABFS result files exist with correct shapes.
    Includes y labels file so we never need to re-extract just for comparison."""
    prefixes = ['preq_abfs_ba', 'preq_abfs_f1', 'preq_abfs_kappa']
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_N_CLFS:
            print(f"  WARNING: {path} has unexpected shape {arr.shape} — will rerun.")
            return False
    # also check y labels file exists
    y_path = os.path.join(RESULTS_DIR, f'abfs_y_{stream_name}.npy')
    if not os.path.exists(y_path):
        return False
    return True


def already_done_komor(stream_name, measure):
    """Return True if all Komorniczak result files for this measure exist
    with correct shapes."""
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
#  ABFS META-FEATURE EXTRACTION (raw scores v2.0)
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

        wt = abfs.relevance_scores()
        abfs.pop_drift_count()

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

    drift_chunks = load_gt(stream_name)
    print(f"  Drift chunks: {drift_chunks}")

    # ---- ABFS meta-feature extraction ----
    if already_done_abfs(stream_name):
        print("  ABFS result files exist and valid — loading from disk.")
        tba_a  = np.load(os.path.join(RESULTS_DIR, f'preq_abfs_ba_{stream_name}.npy'))
        # load saved y labels directly — no need to re-extract the stream
        y_abfs = np.load(os.path.join(RESULTS_DIR, f'abfs_y_{stream_name}.npy'))
        print_label_dist('ABFS (loaded)', y_abfs)
    else:
        print("\n  Extracting ABFS meta-features...")
        X_abfs, y_abfs = extract_abfs_metafeatures(stream_name, drift_chunks)
        print(f"  ABFS meta-dataset: {X_abfs.shape}, concepts: {np.unique(y_abfs)}")
        print_label_dist('ABFS', y_abfs)

        # save y labels so we can load them without re-extracting next time
        save(y_abfs, 'abfs_y', stream_name)

        print("\n  Running prequential (ABFS)...")
        _, _, tba_a, _, _, tf1_a, _, _, tk_a = run_prequential_sweep(X_abfs, y_abfs)

        save(tba_a, 'preq_abfs_ba',    stream_name)
        save(tf1_a, 'preq_abfs_f1',    stream_name)
        save(tk_a,  'preq_abfs_kappa', stream_name)
        print(f"  Saved ABFS prequential results.")

        print(f"  [Preq ABFS] " + "  ".join(
            f"{n}={tba_a[-1, i]:.3f}"
            for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    # ---- Komorniczak: loop over all 9 measure groups ----
    for measure in MEASURES:

        print(f"\n  --- Measure group: {measure} ---")

        if already_done_komor(stream_name, measure):
            print(f"  Komorniczak [{measure}] results exist and valid — skipping.")
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

        # print and compare label distributions — catch misalignment early
        y_komor_dist = print_label_dist(f'Komor [{measure}]', y_komor)
        y_abfs_dist  = print_label_dist('ABFS', y_abfs)
        if y_komor_dist != y_abfs_dist:
            print(f"  WARNING: label distributions differ between ABFS and "
                  f"Komorniczak [{measure}] — check alignment before trusting results.")

        print(f"  Running prequential (Komorniczak [{measure}])...")
        _, _, tba_k, _, _, tf1_k, _, _, tk_k = run_prequential_sweep(X_komor, y_komor)

        save(tba_k, f'preq_komor_{measure}_ba',    stream_name)
        save(tf1_k, f'preq_komor_{measure}_f1',    stream_name)
        save(tk_k,  f'preq_komor_{measure}_kappa', stream_name)
        print(f"  Saved Komorniczak [{measure}] prequential results.")

        print(f"  [Preq Komor {measure}] " + "  ".join(
            f"{n}={tba_k[-1, i]:.3f}"
            for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

        # ---- heatmap: ABFS vs this Komorniczak measure group ----
        final_abfs  = tba_a[-1, :]
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
            tag             = f'preq_{measure}_{stream_name}',
            FIGURES_DIR     = FIGURES_DIR,
        )

print("\nExperiment 3 complete.")