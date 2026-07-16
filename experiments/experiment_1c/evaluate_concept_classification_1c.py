# evaluate_concept_classification_1c.py

# ============================================================
# Experiment 1c: Prequential evaluation of ABFS-based
# meta-features using River classifiers (test-then-train).
#
# Same streams, same concept labelling as 1a/1b.
# Evaluation protocol: prequential (test-then-train per window).
# ABFS warmup: first WARMUP_WINDOWS windows skipped.
#
# Metrics: cumulative balanced accuracy, macro F1, Cohen's Kappa
# per window per classifier per replication.
#
# Output: ABFS classifier-sweep results (clf_ba/f1/kappa_*.npy) and
# the comparison heatmap -> results/experiment_1c/. This script never
# touches Komorniczak's features; that happens in komor_concept_classification_1c.py.

# It generates 18 .npy files in results/experiment_1c (9 per drift type):
#   clf_ba_aggstats_sudden.npy, clf_ba_aggstats_gradual.npy
#   clf_ba_raw_sudden.npy, clf_ba_raw_gradual.npy
#   clf_ba_raw_temporal_sudden.npy, clf_ba_raw_temporal_gradual.npy
#   clf_f1_*.npy (same pattern, 6 files)
#   clf_kappa_*.npy (same pattern, 6 files)
# Each file has shape (n_replications, n_windows, n_clfs)
# where n_windows = N_CHUNKS - WARMUP_WINDOWS = 4990

# And 2 figures in results/experiment_1c/figures (1 per drift type):
#   heatmap_comparison_komorniczak_ABFS_sudden.png
#   heatmap_comparison_komorniczak_ABFS_gradual.png
# Note: run komor_concept_classification_1c.py first to generate
# the Komorniczak baseline files needed for the comparison plots.
# ============================================================

import numpy as np
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import warnings
warnings.filterwarnings('ignore')

from strlearn.streams import StreamGenerator

from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal)
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from plot_results import plot_heatmap_balanced_accuracy_comparison_exp1


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1c')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1c', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_CHUNKS = 5000
CHUNK_SIZE = 200
N_FEATURES = 10
WARMUP_WINDOWS = 10
N_REPLICATIONS = 5

MF_CONFIGS = [('aggstats', 'Aggregate stats (v1.1)', 8), ('raw', 'Raw scores (v2.0)', 10),('raw_temporal', 'Raw + temporal (v2.1)', 12)]

DRIFT_CONFIGS = [('sudden', 20, 9999), ('gradual', 6, 5)]

clf_names = [name for name, _ in BASE_CLFS_PREQUENTIAL]

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

MEASURES = ['clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical']


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


def make_extract_mf(mf_type):
    if mf_type == 'aggstats':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures(wt=wt, wt_prev=wt_prev,drift_count=drift_count, time_since_drift=time_since_drift)
    elif mf_type == 'raw':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw(wt)
    elif mf_type == 'raw_temporal':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw_temporal(wt, wt_prev=wt_prev)
    else:
        raise ValueError(f"Unknown mf_type: {mf_type}")
    return extract_mf


def extract_metafeatures_for_stream(random_state, mf_type, drift_type,n_drifts, concept_sigmoid_spacing):
    """
    Run ABFS on one stream and extract meta-features.

    Returns
    -------
    X: np.ndarray, shape (n_windows, n_features)
    y: np.ndarray, shape (n_windows,)
    """
    config = {
        'n_drifts': n_drifts,
        'n_chunks': N_CHUNKS,
        'chunk_size': CHUNK_SIZE,
        'n_features': N_FEATURES,
        'n_informative': N_FEATURES,
        'n_redundant': 0,
        'n_repeated': 0,
        'concept_sigmoid_spacing': concept_sigmoid_spacing,
        'random_state': random_state
    }
    stream = StreamGenerator(**config)
    extract_mf = make_extract_mf(mf_type)

    # pass 1: concept labels
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
        accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

    if drift_type == 'sudden':
        concept_selector_saved = stream.concept_selector.copy()
        concept_labels_all = np.array([
            int(np.bincount(concept_selector_saved[
                i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]).argmax())
            for i in range(N_CHUNKS)])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, CHUNK_SIZE)

    # pass 2: extract meta-features
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)

    meta_features = []
    concept_labels = []
    wt_prev = None
    window_counter = 0

    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()

        if window_counter >= WARMUP_WINDOWS:
            mf = extract_mf(wt, wt_prev, drift_count, abfs.time_since_drift)
            meta_features.append(mf)
            concept_labels.append(concept_labels_all[window_counter])

        wt_prev = wt
        window_counter += 1

    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y


# ============================================================
#  MAIN
# ============================================================
for drift_type, n_drifts, concept_sigmoid_spacing in DRIFT_CONFIGS:
    n_concepts = 25 if drift_type == 'gradual' else n_drifts + 1
    random_baseline = 1 / n_concepts

    print(f"\n{'#'*60}")
    print(f"DRIFT TYPE: {drift_type} ({n_concepts} concepts)")
    print(f"{'#'*60}")

    all_mean_ba = {}
    all_std_ba  = {}
    all_median_ba = {}

    for mf_type, mf_label, n_mf in MF_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Meta-features: {mf_label} ({n_mf}) | Drift: {drift_type}")
        print(f"{'='*60}")

        all_ba  = []
        all_f1  = []
        all_kap = []
        out_file = os.path.join(
            RESULTS_DIR,
            f'clf_ba_{mf_type}_{drift_type}.npy'
        )

        if os.path.exists(out_file):
            print(f"Loading existing results for {mf_type} ({drift_type})")

            all_ba = np.load(out_file)

            all_mean_ba[mf_type] = np.mean(all_ba[:, -1, :], axis=0)
            all_std_ba[mf_type] = np.std(all_ba[:, -1, :], axis=0)
            all_median_ba[mf_type] = np.median(all_ba[:, -1, :], axis=0)

            continue

        
        for rep_id, rs in enumerate(RANDOM_STATES):
            print(f"Rep {rep_id+1}/{N_REPLICATIONS} (seed={rs})...")

            X, y = extract_metafeatures_for_stream(rs, mf_type, drift_type, n_drifts, concept_sigmoid_spacing)
            mean_ba, std_ba, traj_ba, mean_f1, std_f1, traj_f1, mean_kappa, std_kappa, traj_kappa = run_prequential_sweep(X, y)

            all_ba.append(traj_ba)
            all_f1.append(traj_f1)
            all_kap.append(traj_kappa)

            print(f"{'Clf':<6s} {'Final BA':>10s} {'Final F1':>10s} {'Final K':>10s}")
            for clf_id, name in enumerate(clf_names):
                print(f"{name:<6s} {traj_ba[-1, clf_id]:>10.4f} "
                      f"{traj_f1[-1, clf_id]:>10.4f} "
                      f"{traj_kappa[-1, clf_id]:>10.4f}")

        # shape: (n_replications, n_windows, n_clfs)
        all_ba  = np.array(all_ba)
        all_f1  = np.array(all_f1)
        all_kap = np.array(all_kap)

        np.save(os.path.join(RESULTS_DIR, f'clf_ba_{mf_type}_{drift_type}.npy'),    all_ba)
        np.save(os.path.join(RESULTS_DIR, f'clf_f1_{mf_type}_{drift_type}.npy'),    all_f1)
        np.save(os.path.join(RESULTS_DIR, f'clf_kappa_{mf_type}_{drift_type}.npy'), all_kap)
        print(f"\nSaved to {RESULTS_DIR}")

        all_mean_ba[mf_type] = np.mean(all_ba[:, -1, :], axis=0)
        all_std_ba[mf_type]  = np.std(all_ba[:, -1, :],  axis=0)
        all_median_ba[mf_type] = np.median(all_ba[:, -1, :], axis=0)                           

        print(f"\nMean final BA across replications:")
        print(f"{'Clf':<6s} {'Mean BA':>10s}")
        for clf_id, name in enumerate(clf_names):
            print(f"{name:<6s} {all_mean_ba[mf_type][clf_id]:>10.4f}")

    # ============================================================
    #  COMPARISON - our ABFS meta-features vs komor_concept_classif
    # ============================================================
    rc_path = os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_ba_{drift_type}.npy')

    if os.path.exists(rc_path):
        rc_raw = np.load(rc_path)  # shape: (n_measures, n_replications, n_windows, n_clfs)
        plot_heatmap_balanced_accuracy_comparison_exp1(all_mean_ba, all_std_ba, all_median_ba, rc_raw, MEASURES, BASE_CLFS_PREQUENTIAL,drift_type, n_concepts, FIGURES_DIR,
            exp_label='1c',filename=f'heatmap_comparison_komorniczak_ABFS_{drift_type}.png')
    else:
        print(f"\nWarning: {rc_path} not found - run komor_concept_classification_1c.py first.")