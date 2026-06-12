# evaluate_concept_classification_2.py
# ==============================================================================
# Experiment 2: Stream Configuration Sensitivity
#
# Tests how ALL THREE ABFS meta-feature versions and all 9 Komorniczak
# measure groups respond to changes in:
#   - chunk_size   : 100, 200, 500, 1000
#   - n_informative: 3, 5, 10, 15 (n_features fixed at 20)
#
# ABFS versions:
#   - v1.1 aggstats    : 8-dim aggregate statistics
#   - v2.0 raw scores  : 20-dim normalized relevance vector
#   - v2.1 raw+temporal: 22-dim (v2.0 + delta_mean + cosine_sim)
#
# 4x4 grid = 16 configurations x 2 drift types = 32 stream variants.
# Evaluation protocol: Prequential only (test-then-train per window).
# 5 replications per cell.
#
# Komorniczak baseline: all 9 measure groups re-extracted using pymfe
# on the same streams.
#
# Outputs saved to results/experiment_2/:
#   preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy  (n_reps, n_windows, n_clfs)
#   preq_abfs_{version}_f1_...
#   preq_abfs_{version}_kappa_...
#   preq_komor_{measure}_ba_...
#   preq_komor_{measure}_f1_...
#   preq_komor_{measure}_kappa_...
#
#   Figures saved in results/experiment_2/figures/:
#     heatmap_comparison_komorniczak_ABFS_preq_chunk{cs}_ninf{ni}_{drift}.png
# ==============================================================================

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')    # experiments/
sys.path.append('../..') # project root

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
)
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from plot_results import plot_heatmap_balanced_accuracy_comparison
from pymfe.mfe import MFE


# ============================================================
#  FIXED CONFIGURATION
# ============================================================

N_CHUNKS       = 5000
N_FEATURES     = 20
WARMUP_WINDOWS = 10
N_REPLICATIONS = 5

CHUNK_SIZES    = [500]
N_INFORMATIVES = [10, 15]

DRIFT_CONFIGS = [
    ('gradual',  6,    5)  # 25 concepts
]

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

ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {
    'aggstats':     'Aggstats (v1.1)',
    'raw':          'Raw scores (v2.0)',
    'raw_temporal': 'Raw + temporal (v2.1)',
}

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)



CLF_NAMES = [n for n, _ in BASE_CLFS_PREQUENTIAL]
N_CLFS = len(CLF_NAMES)


# ============================================================
#  HELPERS
# ============================================================

def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing, stream.n_drifts)[1][::chunk_size]
    concept   = 0
    decreasing = True
    labels    = []
    for chunk in range(n_chunks):
        if decreasing:
            if concept % 4 == 0:
                if e[chunk] < 0.9:  concept += 1
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
                if e[chunk] > 0.1:  concept += 1
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
    if drift_type == 'sudden':
        cs = stream.concept_selector.copy()
        return np.array([
            int(np.bincount(cs[i*chunk_size:(i+1)*chunk_size]).argmax())
            for i in range(n_chunks)
        ])
    else:
        return assign_labels_gradual(stream, n_chunks, chunk_size)


def make_tag(chunk_size, n_informative, drift_type):
    return f"chunk{chunk_size}_ninf{n_informative}_{drift_type}"


def already_done_abfs(tag, version):
    prefixes = [f'preq_abfs_{version}_ba', f'preq_abfs_{version}_f1',
                f'preq_abfs_{version}_kappa']
    return all(os.path.exists(os.path.join(RESULTS_DIR, f'{p}_{tag}.npy'))
               for p in prefixes)
 
 
def already_done_komor(tag, measure):
    prefixes = [f'preq_komor_{measure}_ba', f'preq_komor_{measure}_f1',
                f'preq_komor_{measure}_kappa']
    return all(os.path.exists(os.path.join(RESULTS_DIR, f'{p}_{tag}.npy'))
               for p in prefixes)


def save(array, prefix, tag):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{tag}.npy')
    np.save(path, array)
    return path


# ============================================================
#  HEATMAP COMPARISON 
#  Left:  9 Komorniczak measure groups x classifiers
#  Right: ABFS raw scores v2.0         x classifiers
# ============================================================

def plot_combined_heatmap(tag, drift_type, n_concepts, abfs_results_dict):
    """
    abfs_results_dict: {version: np.ndarray (n_reps, n_windows, n_clfs)}
    """
    # Komorniczak matrix (9 x n_clfs)
    komor_matrix     = np.full((len(MEASURES), N_CLFS), np.nan)
    komor_std_matrix = np.full((len(MEASURES), N_CLFS), np.nan)
    for m_id, measure in enumerate(MEASURES):
        path = os.path.join(RESULTS_DIR, f'preq_komor_{measure}_ba_{tag}.npy')
        if not os.path.exists(path):
            print(f"  Heatmap: missing {measure} — skipping.")
            return
        arr = np.load(path)  # (n_reps, n_windows, n_clfs)
        komor_matrix[m_id, :]     = np.mean(arr[:, -1, :], axis=0)
        komor_std_matrix[m_id, :] = np.std(arr[:, -1, :],  axis=0)
 
    # ABFS matrix (3 x n_clfs)
    abfs_matrix     = np.full((len(ABFS_VERSIONS), N_CLFS), np.nan)
    abfs_std_matrix = np.full((len(ABFS_VERSIONS), N_CLFS), np.nan)
    for v_id, version in enumerate(ABFS_VERSIONS):
        if version in abfs_results_dict:
            arr = abfs_results_dict[version]
        else:
            path = os.path.join(RESULTS_DIR, f'preq_abfs_{version}_ba_{tag}.npy')
            if not os.path.exists(path):
                print(f"  Heatmap: missing ABFS {version} — skipping.")
                return
            arr = np.load(path)
        abfs_matrix[v_id, :]     = np.mean(arr[:, -1, :], axis=0)
        abfs_std_matrix[v_id, :] = np.std(arr[:, -1, :],  axis=0)
 
    plot_heatmap_balanced_accuracy_comparison(
        komor_matrix     = komor_matrix,
        komor_std_matrix = komor_std_matrix,
        abfs_matrix      = abfs_matrix,
        abfs_std_matrix  = abfs_std_matrix,
        MEASURES         = MEASURES,
        ABFS_VERSIONS    = ABFS_VERSIONS,
        ABFS_LABELS      = ABFS_LABELS,
        clf_names        = CLF_NAMES,
        drift_type       = drift_type,
        n_concepts       = n_concepts,
        tag              = f'preq_{tag}',
        FIGURES_DIR      = FIGURES_DIR,
    )


# ============================================================
#  ABFS META-FEATURE EXTRACTION — all 3 versions in one pass
# ============================================================
 
def extract_abfs_metafeatures(random_state, drift_type, n_drifts,
                               concept_sigmoid_spacing, chunk_size, n_informative):
    """
    Returns X_aggstats, X_raw, X_raw_temporal, y  (all after warmup skip)
    """
    config = dict(
        n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=chunk_size,
        n_features=N_FEATURES, n_informative=n_informative,
        n_redundant=0, n_repeated=0,
        concept_sigmoid_spacing=concept_sigmoid_spacing,
        random_state=random_state,
    )
    stream = StreamGenerator(**config)
 
    # pass 1: concept labels (needs full stream run first)
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size, class_window_size=chunk_size)
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
    concept_labels_all = get_concept_labels(stream, drift_type, N_CHUNKS, chunk_size)
 
    # pass 2: extract all 3 versions
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size, class_window_size=chunk_size)
    mf_aggstats, mf_raw, mf_raw_temporal, concept_labels = [], [], [], []
    wt_prev, window_counter = None, 0
 
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()
        t_since     = abfs.time_since_drift
 
        if window_counter >= WARMUP_WINDOWS:
            mf_aggstats.append(extract_metafeatures(wt, wt_prev, drift_count, t_since))
            mf_raw.append(extract_metafeatures_raw(wt))
            mf_raw_temporal.append(extract_metafeatures_raw_temporal(wt, wt_prev))
            concept_labels.append(concept_labels_all[window_counter])
 
        wt_prev = wt
        window_counter += 1
 
    def clean(arr):
        a = np.array(arr, dtype=float)
        a[np.isnan(a)] = 1; a[np.isinf(a)] = 1
        return a
 
    return clean(mf_aggstats), clean(mf_raw), clean(mf_raw_temporal), np.array(concept_labels)


# ============================================================
#  KOMORNICZAK BASELINE EXTRACTION (one measure group at a time)
# ============================================================
 
def extract_komor_metafeatures(random_state, drift_type, n_drifts,
                                concept_sigmoid_spacing, chunk_size,
                                n_informative, measure):
    config = dict(
        n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=chunk_size,
        n_features=N_FEATURES, n_informative=n_informative,
        n_redundant=0, n_repeated=0,
        concept_sigmoid_spacing=concept_sigmoid_spacing,
        random_state=random_state,
    )
    stream = StreamGenerator(**config)
 
    abfs_dummy = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                            accuracy_window_size=chunk_size, class_window_size=chunk_size)
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs_dummy.update(X_chunk[i], y_chunk[i])
    concept_labels_all = get_concept_labels(stream, drift_type, N_CHUNKS, chunk_size)
 
    mfe = MFE(groups=[measure], suppress_warnings=True)
    meta_features, concept_labels = [], []
 
    stream.reset()
    window_counter = 0
    for X_chunk, y_chunk in stream:
        if window_counter < WARMUP_WINDOWS:
            window_counter += 1
            continue
        try:
            mfe.fit(X_chunk, y_chunk)
            _, ft_vals = mfe.extract(suppress_warnings=True)
            ft_vals = np.array(ft_vals, dtype=float)
            ft_vals[np.isnan(ft_vals)] = 0; ft_vals[np.isinf(ft_vals)] = 0
        except Exception:
            ft_vals = np.zeros(1)
        meta_features.append(ft_vals)
        concept_labels.append(concept_labels_all[window_counter])
        window_counter += 1
 
    lengths    = [len(f) for f in meta_features]
    target_len = max(set(lengths), key=lengths.count)
    padded = []
    for f in meta_features:
        if   len(f) == target_len: padded.append(f)
        elif len(f) <  target_len: padded.append(np.concatenate([f, np.zeros(target_len - len(f))]))
        else:                      padded.append(f[:target_len])
 
    X = np.array(padded, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1; X[np.isinf(X)] = 1
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
            print(f"chunk_size={chunk_size} | n_informative={n_informative} "
                  f"| drift={drift_type}  [tag: {tag}]")
            print(f"{'='*70}")
 
            # ---- ABFS: all 3 versions ----
            abfs_results = {v: {'ba': [], 'f1': [], 'kappa': []}
                            for v in ABFS_VERSIONS}
            versions_needed = [v for v in ABFS_VERSIONS
                               if not already_done_abfs(tag, v)]
 
            abfs_arrays = {}   # {version: (n_reps, n_windows, n_clfs)}
 
            if not versions_needed:
                print("  All ABFS versions done — loading from disk.")
                for version in ABFS_VERSIONS:
                    abfs_arrays[version] = np.load(
                        os.path.join(RESULTS_DIR, f'preq_abfs_{version}_ba_{tag}.npy'))
            else:
                X_by_version_reps = {v: [] for v in ABFS_VERSIONS}
                y_reps            = []
 
                for rep_id, rs in enumerate(RANDOM_STATES):
                    print(f"\n  [ABFS] Replication {rep_id+1}/{N_REPLICATIONS} (seed={rs})...")
                    Xa, Xr, Xrt, y = extract_abfs_metafeatures(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)
                    X_by_version_reps['aggstats'].append(Xa)
                    X_by_version_reps['raw'].append(Xr)
                    X_by_version_reps['raw_temporal'].append(Xrt)
                    y_reps.append(y)
 
                for version in ABFS_VERSIONS:
                    if already_done_abfs(tag, version):
                        print(f"  ABFS [{version}] already done — loading.")
                        abfs_arrays[version] = np.load(
                            os.path.join(RESULTS_DIR, f'preq_abfs_{version}_ba_{tag}.npy'))
                        continue
 
                    ba_list, f1_list, kappa_list = [], [], []
                    for rep_id in range(N_REPLICATIONS):
                        X = X_by_version_reps[version][rep_id]
                        y = y_reps[rep_id]
                        _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(X, y)
                        ba_list.append(tba); f1_list.append(tf1); kappa_list.append(tk)
                        print(f"    [Preq ABFS {version} rep{rep_id+1}] " + "  ".join(
                            f"{n}={tba[-1, i]:.3f}"
                            for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))
 
                    ba_arr = np.array(ba_list)
                    save(ba_arr,            f'preq_abfs_{version}_ba',    tag)
                    save(np.array(f1_list), f'preq_abfs_{version}_f1',    tag)
                    save(np.array(kappa_list), f'preq_abfs_{version}_kappa', tag)
                    abfs_arrays[version] = ba_arr
                    print(f"  Saved ABFS [{version}] for {tag}")
 
            # ---- Komorniczak: all 9 measure groups ----
            for measure in MEASURES:
                print(f"\n  --- Measure: {measure} ---")
 
                if already_done_komor(tag, measure):
                    print(f"  Komorniczak [{measure}] done — skipping.")
                    continue
 
                ba_list, f1_list, kappa_list = [], [], []
                for rep_id, rs in enumerate(RANDOM_STATES):
                    print(f"  [Komor {measure}] rep {rep_id+1}/{N_REPLICATIONS} (seed={rs})...")
                    X_komor, y_komor = extract_komor_metafeatures(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative, measure)
                    _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(X_komor, y_komor)
                    ba_list.append(tba); f1_list.append(tf1); kappa_list.append(tk)
                    print(f"    [Preq Komor {measure} rep{rep_id+1}] " + "  ".join(
                        f"{n}={tba[-1, i]:.3f}"
                        for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))
 
                save(np.array(ba_list),    f'preq_komor_{measure}_ba',    tag)
                save(np.array(f1_list),    f'preq_komor_{measure}_f1',    tag)
                save(np.array(kappa_list), f'preq_komor_{measure}_kappa', tag)
                print(f"  Saved Komorniczak [{measure}] for {tag}")
 
            # ---- combined heatmap ----
            plot_combined_heatmap(
                tag              = tag,
                drift_type       = drift_type,
                n_concepts       = n_concepts,
                abfs_results_dict = abfs_arrays,
            )