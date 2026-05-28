# evaluate_concept_classification_2.py
# ==============================================================================
# Experiment 2: Stream Configuration Sensitivity
#
# Tests how our ABFS-based meta-features (raw scores, v2.0 and raw + temporal, v2.1) and
# Komorniczak statistical meta-features respond to changes in:
#   - chunk_size: 100, 200, 500, 1000
#   - n_informative: 3, 5, 10, 15 (n_features fixed at 20)
#
# This gives a 4x4 grid of 16 configurations, evaluated on
# both sudden and gradual drift -> 32 stream variants total.
#
# The cell (chunk_size=200, n_informative=10) reproduces the
# Experiment 1a baseline as an internal consistency check.
# Note: n_features=20 instead of 10, so 10 extra noise features
# are present in the baseline cell. ABFS is expected to assign
# them near-zero relevance scores.
#
# Two evaluation protocols per configuration:
#   - Shuffled CV (Experiment 1a protocol):
#       RepeatedStratifiedKFold(n_splits=2, n_repeats=5),
#       5 replications -> 50 evaluations per classifier
#   - Prequential (Experiment 1c protocol):
#       test-then-train per window, 5 replications
#
# Meta-features: raw scores (v2.0) only.
#
# Komorniczak baseline: statistical measure group, re-extracted
# on the same streams using pymfe, evaluated under both protocols.
## Komorniczak baseline:
#   In Experiments 1a, 1b, and 1c, Komorniczak features were
#   loaded directly from pre-extracted .npy files produced by
#   E1_extract_synthetic.py using chunk_size=200 and n_informative=10.
#   Those files cannot be reused here because Experiment 2 varies both
#   parameters: a file extracted at chunk_size=200 does not describe 
#   the same stream windows as one generated at chunk_size=100 or
#   chunk_size=500, and changing n_informative changes which features 
#   are relevant in the stream itself. Re-extracting pymfe statistical
#   features on the same streams we generate ensures that both
#   ABFS and Komorniczak always describe exactly the same chunks,
#   making the comparison fair across all grid cells.
#
# Output all generated files saved in results/experiment_2/:
#
#   Naming convention:
#     {protocol}_{features}_{metric}_chunk{cs}_ninf{ni}_{drift}.npy
#       where:
#         - protocol: 'cv' or 'preq'
#         - features: 'abfs' or 'komor'
#         - metric: 'ba', 'f1', or 'kappa'
#         - cs: chunk_size (100, 200, 500, 1000)
#         - ni: n_informative (3, 5, 10, 15)
#         - drift: 'sudden' or 'gradual'
#
#   CV protocol (shape: n_replications x n_folds x n_clfs):
#     cv_abfs_ba_chunk{cs}_ninf{ni}_{drift}.npy
#     cv_abfs_f1_chunk{cs}_ninf{ni}_{drift}.npy
#     cv_abfs_kappa_chunk{cs}_ninf{ni}_{drift}.npy
#     cv_komor_ba_chunk{cs}_ninf{ni}_{drift}.npy
#     cv_komor_f1_chunk{cs}_ninf{ni}_{drift}.npy
#     cv_komor_kappa_chunk{cs}_ninf{ni}_{drift}.npy
#
#   Prequential protocol (shape: n_replications x n_windows x n_clfs):
#     preq_abfs_ba_chunk{cs}_ninf{ni}_{drift}.npy
#     preq_abfs_f1_chunk{cs}_ninf{ni}_{drift}.npy
#     preq_abfs_kappa_chunk{cs}_ninf{ni}_{drift}.npy
#     preq_komor_ba_chunk{cs}_ninf{ni}_{drift}.npy
#     preq_komor_f1_chunk{cs}_ninf{ni}_{drift}.npy
#     preq_komor_kappa_chunk{cs}_ninf{ni}_{drift}.npy
#
#   Total: 16 configs x 2 drift types x 6 files per protocol (3 metrics x 2 meta-features sets) x 2 protocols
#                         = 384 .npy files

#   Figures saved in results/experiment_2/figures/:
#     heatmap_comparison_komorniczak_ABFS_chunk{cs}_ninf{ni}_{drift}.png
#       one per grid cell per drift type, ABFS vs Komorniczak side by side
#     Total: 16 configs x 2 drift types = 32 figures
# ==============================================================================


import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
 
sys.path.append('..') # experiments/
sys.path.append('../..') # project root

import matplotlib.pyplot as plt

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from classifier_sweep_komor import run_classifier_sweep, BASE_CLFS
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from plot_results import plot_heatmap_balanced_accuracy_comparison_exp2
from pymfe.mfe import MFE


# ============================================================
#  FIXED CONFIGURATION
# ============================================================
 
N_CHUNKS = 5000
N_FEATURES = 20
WARMUP_WINDOWS = 10
N_REPLICATIONS = 5
 
CHUNK_SIZES = [100, 200, 500, 1000]
N_INFORMATIVES = [3, 5, 10, 15]
 
DRIFT_CONFIGS = [('sudden', 20, 9999), # 21 concepts
    ('gradual', 6, 5), # 25 concepts
]
 
np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)



# ==========
#  HELPERS
# ==========

def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing, stream.n_drifts)[1][::chunk_size]
    concept = 0
    decreasing = True
    labels = []
    for chunk in range(n_chunks):
        if decreasing:
            if concept % 4 == 0:
                if e[chunk] < 0.9: concept += 1
            if concept % 4 == 1:
                if e[chunk] < 0.75: concept += 1
            if concept % 4 == 2:
                if e[chunk] < 0.25: concept += 1
            if concept % 4 == 3:
                if e[chunk] < 0.1:
                    concept += 1
                    decreasing = False
        else:
            if concept % 4 == 0:
                if e[chunk] > 0.1: concept += 1
            if concept % 4 == 1:
                if e[chunk] > 0.25: concept += 1
            if concept % 4 == 2:
                if e[chunk] > 0.75: concept += 1
            if concept % 4 == 3:
                if e[chunk] > 0.9:
                    concept += 1
                    decreasing = True
        labels.append(concept)
    return np.array(labels)


def get_concept_labels(stream, drift_type, n_chunks, chunk_size):
    """Return per-chunk concept labels for either drift type."""
    if drift_type == 'sudden':
        cs = stream.concept_selector.copy()
        return np.array([int(np.bincount(cs[i*chunk_size:(i+1)*chunk_size]).argmax()) for i in range(n_chunks)])
    else:
        return assign_labels_gradual(stream, n_chunks, chunk_size)
 
 
def make_tag(chunk_size, n_informative, drift_type):
    """Consistent filename tag for one grid cell."""
    return f"chunk{chunk_size}_ninf{n_informative}_{drift_type}"
 
 
def already_done(tag):
    """Return True if all 12 result files for this cell exist."""
    prefixes = ['cv_abfs_ba', 'cv_abfs_f1', 'cv_abfs_kappa', 'cv_komor_ba', 'cv_komor_f1', 'cv_komor_kappa',
        'preq_abfs_ba', 'preq_abfs_f1', 'preq_abfs_kappa', 'preq_komor_ba', 'preq_komor_f1', 'preq_komor_kappa']
    return all(os.path.exists(os.path.join(RESULTS_DIR, f'{p}_{tag}.npy')) for p in prefixes)
 
 
def save(array, prefix, tag):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{tag}.npy')
    np.save(path, array)
    return path
 



# ============================================================
#  ABFS META-FEATURE EXTRACTION (raw scores v2.0)
# ============================================================
 
def extract_abfs_metafeatures(random_state, drift_type, n_drifts, concept_sigmoid_spacing, chunk_size, n_informative):
    """
    Run ABFS on one stream and return raw score meta-features (v2.0).
 
    Returns
    -------
    X : np.ndarray, shape (N_CHUNKS - WARMUP_WINDOWS, N_FEATURES)
    y : np.ndarray, shape (N_CHUNKS - WARMUP_WINDOWS,)
    """

    config = dict(
        n_drifts = n_drifts,
        n_chunks = N_CHUNKS,
        chunk_size = chunk_size,
        n_features = N_FEATURES,
        n_informative = n_informative,
        n_redundant = 0,
        n_repeated = 0,
        concept_sigmoid_spacing = concept_sigmoid_spacing,
        random_state = random_state,
    )
 
    stream = StreamGenerator(**config)
 
    # pass 1: concept labels
    abfs = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = chunk_size,
        class_window_size    = chunk_size,
    )
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
 
    concept_labels_all = get_concept_labels(stream, drift_type,
                                            N_CHUNKS, chunk_size)
 
    # pass 2: extract meta-features
    abfs = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = chunk_size,
        class_window_size    = chunk_size,
    )
 
    meta_features  = []
    concept_labels = []
    wt_prev        = None
    window_counter = 0
 
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
 
        wt = abfs.relevance_scores()
        abfs.pop_drift_count()
 
        if window_counter >= WARMUP_WINDOWS:
            meta_features.append(extract_metafeatures_raw(wt))
            concept_labels.append(concept_labels_all[window_counter])
 
        wt_prev = wt
        window_counter += 1
    
    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y


# ============================================================
#  KOMORNICZAK BASELINE EXTRACTION
#
#  Re-extracted using pymfe on the same streams because the
#  pre-extracted .npy files from Experiments 1a-1c used
#  chunk_size=200 and n_informative=10 and cannot be reused
#  here. See header comment for full explanation.
# ============================================================
 
def extract_komor_metafeatures(random_state, drift_type, n_drifts,
                                concept_sigmoid_spacing,
                                chunk_size, n_informative):
    """
    Extract Komorniczak statistical meta-features using pymfe,
    skipping the first WARMUP_WINDOWS chunks for alignment with ABFS.
 
    Returns
    -------
    X : np.ndarray, shape (N_CHUNKS - WARMUP_WINDOWS, n_komor_features)
    y : np.ndarray, shape (N_CHUNKS - WARMUP_WINDOWS,)
    """
    config = dict(
        n_drifts                = n_drifts,
        n_chunks                = N_CHUNKS,
        chunk_size              = chunk_size,
        n_features              = N_FEATURES,
        n_informative           = n_informative,
        n_redundant             = 0,
        n_repeated              = 0,
        concept_sigmoid_spacing = concept_sigmoid_spacing,
        random_state            = random_state,
    )
 
    stream = StreamGenerator(**config)
 
    # pass 1: concept labels
    abfs_dummy = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = chunk_size,
        class_window_size    = chunk_size,
    )
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs_dummy.update(X_chunk[i], y_chunk[i])
 
    concept_labels_all = get_concept_labels(stream, drift_type,
                                            N_CHUNKS, chunk_size)
 
    # pass 2: pymfe statistical features per chunk
    mfe = MFE(groups=['statistical'], suppress_warnings=True)
 
    meta_features  = []
    concept_labels = []
 
    stream.reset()
    window_counter = 0

    for X_chunk, y_chunk in stream:   # one chunk at a time — no OOM
        if window_counter < WARMUP_WINDOWS:
            window_counter += 1
            continue
 
        try:
            mfe.fit(X_chunk, y_chunk)
            _, ft_vals = mfe.extract(suppress_warnings=True)
            ft_vals = np.array(ft_vals, dtype=float)
            ft_vals[np.isnan(ft_vals)] = 0
            ft_vals[np.isinf(ft_vals)] = 0
        except Exception:
            ft_vals = np.zeros(1)
 
        meta_features.append(ft_vals)
        concept_labels.append(concept_labels_all[window_counter])
        window_counter += 1


    # pad to uniform width (pymfe can return varying lengths per chunk)
    lengths    = [len(f) for f in meta_features]
    target_len = max(set(lengths), key=lengths.count)
    padded = []
    for f in meta_features:
        if len(f) == target_len:
            padded.append(f)
        elif len(f) < target_len:
            padded.append(np.concatenate([f, np.zeros(target_len - len(f))]))
        else:
            padded.append(f[:target_len])
 
    X = np.array(padded, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y



# ============================================================
#  MAIN SWEEP
# ============================================================
 
for drift_type, n_drifts, concept_sigmoid_spacing in DRIFT_CONFIGS:
    n_concepts      = 25 if drift_type == 'gradual' else n_drifts + 1
    random_baseline = 1 / n_concepts
 
    print(f"\n{'#'*70}")
    print(f"DRIFT TYPE: {drift_type.upper()} ({n_concepts} concepts, "
          f"random baseline={random_baseline:.3f})")
    print(f"{'#'*70}")
 
    for chunk_size in CHUNK_SIZES:
        for n_informative in N_INFORMATIVES:
 
            tag = make_tag(chunk_size, n_informative, drift_type)
 
            print(f"\n{'='*70}")
            print(f"chunk_size={chunk_size} | n_informative={n_informative}"
                  f" | drift={drift_type}  [tag: {tag}]")
            print(f"{'='*70}")
 
            if already_done(tag):
                print("  All 12 result files exist — skipping.")
                continue
 
            cv_abfs_ba     = []
            cv_abfs_f1     = []
            cv_abfs_kappa  = []
            cv_komor_ba    = []
            cv_komor_f1    = []
            cv_komor_kappa = []
 
            pr_abfs_ba     = []
            pr_abfs_f1     = []
            pr_abfs_kappa  = []
            pr_komor_ba    = []
            pr_komor_f1    = []
            pr_komor_kappa = []
 
            for rep_id, rs in enumerate(RANDOM_STATES):
                print(f"\n  Replication {rep_id+1}/{N_REPLICATIONS} "
                      f"(seed={rs})...")
 
                # --- feature extraction ---
                X_abfs, y_abfs = extract_abfs_metafeatures(
                    rs, drift_type, n_drifts, concept_sigmoid_spacing,
                    chunk_size, n_informative)
 
                X_komor, y_komor = extract_komor_metafeatures(
                    rs, drift_type, n_drifts, concept_sigmoid_spacing,
                    chunk_size, n_informative)
 
                # --- shuffled CV ---
                np.random.seed(1233)
                mba_a, _, rba, _, _, rf1, _, _, rk = run_classifier_sweep(
                    X_abfs, y_abfs, shuffle_seed=None)
                cv_abfs_ba.append(rba)
                cv_abfs_f1.append(rf1)
                cv_abfs_kappa.append(rk)
 
                np.random.seed(1233)
                mba_k, _, rba, _, _, rf1, _, _, rk = run_classifier_sweep(
                    X_komor, y_komor, shuffle_seed=None)
                cv_komor_ba.append(rba)
                cv_komor_f1.append(rf1)
                cv_komor_kappa.append(rk)
 
                print(f"    [CV   ABFS ] " + "  ".join(
                    f"{n}={mba_a[i]:.3f}" for i, (n, _) in enumerate(BASE_CLFS)))
                print(f"    [CV   Komor] " + "  ".join(
                    f"{n}={mba_k[i]:.3f}" for i, (n, _) in enumerate(BASE_CLFS)))
                    
 
                # --- prequential ---
                mba_a2, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(
                    X_abfs, y_abfs)
                pr_abfs_ba.append(tba)
                pr_abfs_f1.append(tf1)
                pr_abfs_kappa.append(tk)
 
                mba_k2, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(
                    X_komor, y_komor)
                pr_komor_ba.append(tba)
                pr_komor_f1.append(tf1)
                pr_komor_kappa.append(tk)
 
                print(f"    [Preq ABFS ] " + "  ".join(
                    f"{n}={pr_abfs_ba[-1][-1, i]:.3f}"
                    for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))
                print(f"    [Preq Komor] " + "  ".join(
                    f"{n}={mba_k2[i]:.3f}"
                    for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))


            # --- save all 12 files for this cell ---
            # CV shape:         (n_replications, n_folds,   n_clfs)
            # Prequential shape: (n_replications, n_windows, n_clfs)
            save(np.array(cv_abfs_ba),    'cv_abfs_ba',    tag)
            save(np.array(cv_abfs_f1),    'cv_abfs_f1',    tag)
            save(np.array(cv_abfs_kappa), 'cv_abfs_kappa', tag)
            save(np.array(cv_komor_ba),   'cv_komor_ba',   tag)
            save(np.array(cv_komor_f1),   'cv_komor_f1',   tag)
            save(np.array(cv_komor_kappa),'cv_komor_kappa',tag)
 
            save(np.array(pr_abfs_ba),    'preq_abfs_ba',    tag)
            save(np.array(pr_abfs_f1),    'preq_abfs_f1',    tag)
            save(np.array(pr_abfs_kappa), 'preq_abfs_kappa', tag)
            save(np.array(pr_komor_ba),   'preq_komor_ba',   tag)
            save(np.array(pr_komor_f1),   'preq_komor_f1',   tag)
            save(np.array(pr_komor_kappa),'preq_komor_kappa',tag)
 
            print(f"\n  Saved 12 files (*_{tag}.npy) -> {RESULTS_DIR}")
 


            # ---- per-cell comparison heatmap ----
            plot_heatmap_balanced_accuracy_comparison_exp2(
                mean_ba_abfs   = np.mean(np.array(cv_abfs_ba),  axis=(0, 1)),
                std_ba_abfs    = np.std(np.array(cv_abfs_ba),   axis=(0, 1)),
                median_ba_abfs = np.median(np.array(cv_abfs_ba),axis=(0, 1)),
                mean_ba_komor  = np.mean(np.array(cv_komor_ba),  axis=(0, 1)),
                std_ba_komor   = np.std(np.array(cv_komor_ba),   axis=(0, 1)),
                median_ba_komor= np.median(np.array(cv_komor_ba),axis=(0, 1)),
                BASE_CLFS      = BASE_CLFS,
                drift_type     = drift_type,
                n_concepts     = n_concepts,
                tag            = tag,
                FIGURES_DIR    = FIGURES_DIR,
            )