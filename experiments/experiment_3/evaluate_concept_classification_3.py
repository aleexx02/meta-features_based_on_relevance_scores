# evaluate_concept_classification_3.py
# ==============================================================================
# Experiment 3 Approach 1: Annotated Real-World Stream Evaluation
#
# Self-contained pipeline: loads streams, extracts ABFS meta-features,
# extracts Komorniczak pymfe meta-features inline, runs prequential
# evaluation. No separate preprocessing step required.
#
# All stream files sourced from the USP DS Repository (Souza et al., 2020)
# https://sites.google.com/view/uspdsrepository
# Prepared by streams/generate_real_streams.py.
#
# Streams (all with ground truth drift annotations):
#   INSECTS-abrupt_imbalanced    33 features,  2 concepts, 1184 chunks
#   INSECTS-gradual_imbalanced   33 features,  6 concepts,  477 chunks
#   INSECTS-incremental_imbalanced 33 features, 6 concepts, 1506 chunks
#   INSECTS-abrupt_balanced      33 features,  2 concepts,  176 chunks
#   INSECTS-gradual_balanced     33 features,  6 concepts,   80 chunks
#   INSECTS-incremental_balanced 33 features,  6 concepts,  190 chunks
#   electricity                   8 features,  6 concepts,  151 chunks
#   covtype                      54 features,  7 concepts, 1936 chunks
#
# chunk_size = 200 — consistent with Experiments 1a-1c and Experiment 2
# baseline configuration.
#
# ABFS versions evaluated:
#   v1.1 aggstats    : 8-dim aggregate statistics
#   v2.0 raw scores  : n_features-dim normalized relevance vector
#   v2.1 raw+temporal: v2.0 + delta_mean + cosine_sim
#
# Komorniczak: all 9 pymfe measure groups extracted inline and cached.
#
# Outputs saved to results/experiment_3/:
#   abfs_y_{stream}.npy                     (n_windows,)
#   preq_abfs_{version}_ba_{stream}.npy     (n_windows, n_clfs)
#   preq_abfs_{version}_f1_{stream}.npy     (n_windows, n_clfs)
#   preq_abfs_{version}_kappa_{stream}.npy  (n_windows, n_clfs)
#   preq_komor_{measure}_ba_{stream}.npy    (n_windows, n_clfs)
#   preq_komor_{measure}_f1_{stream}.npy    (n_windows, n_clfs)
#   preq_komor_{measure}_kappa_{stream}.npy (n_windows, n_clfs)
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
from pymfe.mfe import MFE
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

REAL_STREAM_DIR   = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')
REAL_GT_DIR       = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_gt')
KOMOR_CACHE_DIR   = os.path.join(PROJECT_ROOT, 'external', 'komorniczak', 'results', 'real')
RESULTS_DIR       = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
FIGURES_DIR       = os.path.join(PROJECT_ROOT, 'results', 'experiment_3', 'figures')
os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs(FIGURES_DIR,     exist_ok=True)
os.makedirs(KOMOR_CACHE_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================

# chunk_size = 200: consistent with Experiments 1a-1c and the Experiment 2
# baseline configuration. Fixed throughout to isolate the effect of stream
# type and meta-feature version from chunk size effects, which are already
# characterised in Experiment 2.
CHUNK_SIZE     = 200
WARMUP_WINDOWS = 0

REAL_STREAMS = [
    'INSECTS-abrupt_imbalanced',
    'INSECTS-gradual_imbalanced',
    'INSECTS-incremental_imbalanced',
    'INSECTS-abrupt_balanced',
    'INSECTS-gradual_balanced',
    'INSECTS-incremental_balanced',
    'electricity',
    'covtype',
]

N_FEATURES = {
    'INSECTS-abrupt_imbalanced':       33,
    'INSECTS-gradual_imbalanced':      33,
    'INSECTS-incremental_imbalanced':  33,
    'INSECTS-abrupt_balanced':         33,
    'INSECTS-gradual_balanced':        33,
    'INSECTS-incremental_balanced':    33,
    'electricity':                      8,
    'covtype':                         54,
}

N_CONCEPTS = {
    s: len(np.load(os.path.join(REAL_GT_DIR, f'{s}.npy'))) + 1
    for s in REAL_STREAMS
}

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


# ============================================================
#  HELPERS
# ============================================================

def load_gt(stream_name):
    return np.load(os.path.join(REAL_GT_DIR, f'{stream_name}.npy'))


def already_done_abfs(stream_name, version):
    prefixes = [f'preq_abfs_{version}_ba', f'preq_abfs_{version}_f1',
                f'preq_abfs_{version}_kappa']
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_N_CLFS:
            print(f"  WARNING: {path} shape {arr.shape} — will rerun.")
            return False
    if not os.path.exists(os.path.join(RESULTS_DIR,
                                       f'abfs_y_{stream_name}.npy')):
        return False
    return True


def already_done_komor(stream_name, measure):
    prefixes = [f'preq_komor_{measure}_ba', f'preq_komor_{measure}_f1',
                f'preq_komor_{measure}_kappa']
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_N_CLFS:
            print(f"  WARNING: {path} shape {arr.shape} — will rerun.")
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
# ============================================================

def plot_combined_heatmap(stream_name, n_concepts, tba_versions):
    clf_names = [n for n, _ in BASE_CLFS_PREQUENTIAL]
    n_clfs    = len(clf_names)

    komor_matrix = np.full((len(MEASURES), n_clfs), np.nan)
    for m_id, measure in enumerate(MEASURES):
        path = os.path.join(RESULTS_DIR,
                            f'preq_komor_{measure}_ba_{stream_name}.npy')
        if not os.path.exists(path):
            print(f"  Heatmap: missing {measure} — skipping.")
            return
        komor_matrix[m_id, :] = np.load(path)[-1, :]

    abfs_matrix = np.full((len(ABFS_VERSIONS), n_clfs), np.nan)
    for v_id, version in enumerate(ABFS_VERSIONS):
        if version in tba_versions:
            abfs_matrix[v_id, :] = tba_versions[version][-1, :]
        else:
            path = os.path.join(RESULTS_DIR,
                                f'preq_abfs_{version}_ba_{stream_name}.npy')
            if not os.path.exists(path):
                print(f"  Heatmap: missing ABFS {version} — skipping.")
                return
            abfs_matrix[v_id, :] = np.load(path)[-1, :]

    random_baseline = 1.0 / n_concepts
    fig, axes = plt.subplots(
        1, 2, figsize=(26, max(5, len(MEASURES) * 0.75)),
        gridspec_kw={'width_ratios': [3, 1.5]})

    ax = axes[0]
    ax.imshow(komor_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i, measure in enumerate(MEASURES):
        for j in range(n_clfs):
            val = komor_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if val > 0.6 else 'black')
    ax.set_xticks(range(n_clfs)); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES, fontsize=10)
    ax.set_title('Komorniczak meta-features — balanced accuracy', fontsize=12)

    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i, version in enumerate(ABFS_VERSIONS):
        for j in range(n_clfs):
            val = abfs_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if val > 0.6 else 'black')
    ax.set_xticks(range(n_clfs)); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(ABFS_VERSIONS)))
    ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
    ax.set_title('ABFS meta-features — balanced accuracy', fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f'Komorniczak vs ABFS — {stream_name}\n'
        f'Prequential  |  chunk_size=200  |  '
        f'random baseline = {random_baseline:.3f}',
        fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        f'heatmap_comparison_komorniczak_ABFS_preq_exp3_{stream_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {out_path}")


# ============================================================
#  ABFS EXTRACTION — all 3 versions in one pass
# ============================================================

def extract_abfs_metafeatures(stream_name, drift_chunks):
    n_features  = N_FEATURES[stream_name]
    stream_path = os.path.join(REAL_STREAM_DIR, f'{stream_name}.npy')
    stream = sl.streams.NPYParser(stream_path, chunk_size=CHUNK_SIZE,
                                  n_chunks=100000)
    abfs = ABFS_match(n_features=n_features, categorical_features=[],
                      accuracy_window_size=CHUNK_SIZE,
                      class_window_size=CHUNK_SIZE)

    mf_aggstats, mf_raw, mf_raw_temporal = [], [], []
    concept_labels = []
    wt_prev = None

    for chunk_idx in range(100000):
        concept = int(np.sum(drift_chunks <= chunk_idx))
        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break
        if len(np.unique(y_chunk)) < 2:
            continue
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()
        t_since     = abfs.time_since_drift

        mf_aggstats.append(extract_metafeatures(wt, wt_prev, drift_count, t_since))
        mf_raw.append(extract_metafeatures_raw(wt))
        mf_raw_temporal.append(extract_metafeatures_raw_temporal(wt, wt_prev))
        concept_labels.append(concept)
        wt_prev = wt

    def clean(arr):
        a = np.array(arr, dtype=float)
        a[np.isnan(a)] = 0; a[np.isinf(a)] = 0
        return a

    return (clean(mf_aggstats), clean(mf_raw), clean(mf_raw_temporal),
            np.array(concept_labels))


# ============================================================
#  KOMORNICZAK EXTRACTION — inline, results cached to disk
# ============================================================

def extract_komor_metafeatures(stream_name, measure, drift_chunks):
    """
    Extract pymfe meta-features for one measure group from one stream.
    Results cached to KOMOR_CACHE_DIR — skipped on re-run.
    Output shape: (n_valid_chunks, n_mf + 1) — last column = concept label.
    """
    cache_path = os.path.join(KOMOR_CACHE_DIR,
                              f'komor_real_{stream_name}_{measure}.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)

    stream_path = os.path.join(REAL_STREAM_DIR, f'{stream_name}.npy')
    stream = sl.streams.NPYParser(stream_path, chunk_size=CHUNK_SIZE,
                                  n_chunks=100000)
    mfe = MFE(groups=[measure], suppress_warnings=True)
    out = []

    for chunk_idx in range(100000):
        concept = int(np.sum(drift_chunks <= chunk_idx))
        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break
        if len(np.unique(y_chunk)) < 2:
            continue
        try:
            mfe.fit(X_chunk, y_chunk)
            _, ft = mfe.extract(suppress_warnings=True)
            ft = np.array(ft, dtype=float)
            ft[np.isnan(ft)] = 0; ft[np.isinf(ft)] = 0
        except Exception as e:
            print(f"    chunk {chunk_idx}: pymfe failed ({e}) — skipping.")
            continue
        out.append(np.append(ft, concept))

    result = np.array(out)
    np.save(cache_path, result)
    print(f"  [{measure}] cached: shape={result.shape}")
    return result


# ============================================================
#  MAIN SWEEP
# ============================================================

for stream_name in REAL_STREAMS:
    n_features = N_FEATURES[stream_name]
    n_concepts = N_CONCEPTS[stream_name]

    print(f"\n{'='*70}")
    print(f"Stream   : {stream_name}")
    print(f"Features : {n_features}  |  Concepts: {n_concepts}  |  "
          f"Random baseline: {1/n_concepts:.3f}")
    print(f"{'='*70}")

    drift_chunks = load_gt(stream_name)
    print(f"  Drift chunks: {drift_chunks}")

    # ---- ABFS ----
    versions_needed = [v for v in ABFS_VERSIONS
                       if not already_done_abfs(stream_name, v)]
    tba_versions = {}

    if not versions_needed:
        print("  All ABFS versions done — loading from disk.")
        y_abfs = np.load(os.path.join(RESULTS_DIR,
                                      f'abfs_y_{stream_name}.npy'))
        print_label_dist('ABFS (loaded)', y_abfs)
        for version in ABFS_VERSIONS:
            tba_versions[version] = np.load(
                os.path.join(RESULTS_DIR,
                             f'preq_abfs_{version}_ba_{stream_name}.npy'))
    else:
        print(f"\n  Extracting ABFS (all 3 versions, one pass)...")
        X_agg, X_raw, X_rt, y_abfs = extract_abfs_metafeatures(
            stream_name, drift_chunks)
        print(f"  aggstats     : {X_agg.shape}")
        print(f"  raw          : {X_raw.shape}")
        print(f"  raw_temporal : {X_rt.shape}")
        print_label_dist('ABFS', y_abfs)
        save(y_abfs, 'abfs_y', stream_name)

        X_by_version = {'aggstats': X_agg, 'raw': X_raw,
                        'raw_temporal': X_rt}

        for version in ABFS_VERSIONS:
            if already_done_abfs(stream_name, version):
                print(f"  ABFS [{version}] done — loading.")
                tba_versions[version] = np.load(
                    os.path.join(RESULTS_DIR,
                                 f'preq_abfs_{version}_ba_{stream_name}.npy'))
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

    # ---- Komorniczak ----
    for measure in MEASURES:
        print(f"\n  --- Measure: {measure} ---")
        if already_done_komor(stream_name, measure):
            print(f"  Komorniczak [{measure}] done — skipping.")
            continue

        print(f"  Extracting Komorniczak [{measure}]...")
        komor_data = extract_komor_metafeatures(stream_name, measure,
                                                drift_chunks)
        X_komor = komor_data[:, :-1]
        y_komor = komor_data[:, -1].astype(int)
        X_komor[np.isnan(X_komor)] = 0; X_komor[np.isinf(X_komor)] = 0

        print(f"  Komor: {X_komor.shape}  concepts: {np.unique(y_komor)}")
        y_komor_dist = print_label_dist(f'Komor [{measure}]', y_komor)
        y_abfs_dist  = print_label_dist('ABFS', y_abfs)
        if y_komor_dist != y_abfs_dist:
            print(f"  WARNING: label distributions differ.")

        print(f"  Running prequential...")
        _, _, tba_k, _, _, tf1_k, _, _, tk_k = run_prequential_sweep(
            X_komor, y_komor)
        save(tba_k, f'preq_komor_{measure}_ba',    stream_name)
        save(tf1_k, f'preq_komor_{measure}_f1',    stream_name)
        save(tk_k,  f'preq_komor_{measure}_kappa', stream_name)
        print(f"  [Preq Komor {measure}] " + "  ".join(
            f"{n}={tba_k[-1, i]:.3f}"
            for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    # ---- heatmap ----
    plot_combined_heatmap(stream_name, n_concepts, tba_versions)

print("\nExperiment 3 complete.")