# evaluate_concept_classification_1a.py

# ============================================================
# Evaluation of our ABFS-based meta-features using the same
# experimental setup as Komorniczak et al. (2024):
#   - Same streams: same StreamGenerator configuration and seeds
#   - Same concept labelling: majority vote (sudden) and sigmoid
#     threshold method (gradual)
#   - Same evaluation protocol: classifier_sweep_komor.py

# The only difference with respect to their pipeline is the
# meta-features: instead of statistical descriptors computed
# by pymfe directly from the raw instances, we use relevance
# scores produced by ABFS, encoding which features are
# currently predictive and how that relevance is evolving
# over time.

# By controlling for everything except the meta-features,
# any difference in results observed when comparing
# against replication_check.py can be attributed solely to
# the meta-features themselves.

# Metrics computed: balanced accuracy, macro F1, Cohen's Kappa.
# Results saved as clf_ba_*.npy, clf_f1_*.npy, clf_kappa_*.npy

# Steps:
#   1. Generate a synthetic stream using StreamGenerator
#   2. Run ABFS to compute per-feature relevance scores
#   3. Extract meta-feature vectors from the relevance scores
#   4. Assign concept labels to each window
#   5. Run the classifier sweep (classifier_sweep_komor.py)
#   6. Compare output against replication_check.py results: their meta-features evaluated with our protocol
# ============================================================
 

# RUN: our meta-features evaluated with our protocol (classifier_sweep_komor.py)
# COMPARISON: our meta-features evaluated with our protocol VS. their meta-features evaluated with our protocol
# COMPARE: meta-features (our ABFS-based meta-features vs their statistical meta-features) - using the same evaluation protocol (classifier_sweep_komor.py)


# It generates 18 .npy files in results/experiment_1a (9 per drift type):
    # clf_ba_aggstats_sudden.npy, clf_ba_aggstats_gradual.npy
    # clf_ba_raw_sudden.npy, clf_ba_raw_gradual.npy
    # clf_ba_raw_temporal_sudden.npy, clf_ba_raw_temporal_gradual.npy
    # clf_f1_*.npy (same pattern, 6 files)
    # clf_kappa_*.npy (same pattern, 6 files)
# Each file has shape (n_replications, n_folds, n_clfs) and contains the raw results of the classifier sweep for each replication, fold, and classifier.

# And 2 figures in results/experiment_1a/figures (1 per drift type):
    # heatmap_comparison_komorniczak_ABFS_sudden.png
    # heatmap_comparison_komorniczak_ABFS_gradual.png


import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
import warnings
import os
import sys
sys.path.append('..') # points to experiments/ where classifier_sweep_komor.py is
sys.path.append('../..') # points to project root where plot_results.py is
warnings.filterwarnings('ignore')

from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
    extract_metafeatures_raw_delta,
    extract_metafeatures_raw_cosine)

from classifier_sweep_komor import run_classifier_sweep, BASE_CLFS
from plot_results import print_summary_table_experiment1, plot_heatmap_balanced_accuracy_comparison


# path to results folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) # go up two levels to project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results/experiment_1a')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results/experiment_1a', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================
#  CONFIGURATION (matching Komorniczak et al.)
# =============================================

#  Sudden drift:
#    concept_sigmoid_spacing=9999, n_drifts=20 -> 21 concepts
#    label assignment: majority vote (equivalent to their
#    threshold method since sigmoid is a step function)
#
#  Gradual drift:
#    concept_sigmoid_spacing=5, n_drifts=6 -> 25 concepts
#    label assignment: sigmoid threshold method from
#    E1_extract_synthetic.py (Komorniczak)
#    4 stages per transition:
#      concept%4==0: static   (e >= 0.9 / e <= 0.1)
#      concept%4==1: early    (e >= 0.75 / e >= 0.1)
#      concept%4==2: central  (e >= 0.25 / e >= 0.25)
#      concept%4==3: late     (e >= 0.1  / e >= 0.75)
# ============================================================


N_CHUNKS = 5000
CHUNK_SIZE = 200
N_FEATURES = 10
WARMUP_WINDOWS = 10
N_REPLICATIONS = 5

# rows of the heatmap — one per meta-feature set
MF_CONFIGS = [('aggstats', 'Aggregate stats (v1.1)', 8), ('raw', 'Raw scores (v2.0)', 10),('raw_temporal', 'Raw + temporal (v2.1)', 12),]

# MF_CONFIGS = [
#     ('raw',          'Raw only (v2.0)',        10),
#     ('raw_delta',    'Raw + delta_mean',        11),
#     ('raw_cosine',   'Raw + cosine_sim',        11),
#     ('raw_temporal', 'Raw + both (v2.1)',       12),
# ]

DRIFT_CONFIGS = [
    ('sudden', 20, 9999), # 20 drifts, so 21 concepts
    ('gradual', 6, 5), # 6*4 = 24 drifts, so 25 concepts
]

MEASURES = ['clustering', 'complexity', 'concept', 'general', 'info-theory', 'itemset', 'landmarking', 'model-based', 'statistical']

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

clf_names = [name for name, _ in BASE_CLFS]


# ============================================================
#  HELPER - build extract_mf for a given MF_TYPE
# ============================================================
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
    elif mf_type == 'raw_delta':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw_delta(wt, wt_prev=wt_prev)
    elif mf_type == 'raw_cosine':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw_cosine(wt, wt_prev=wt_prev)
    else:
        raise ValueError(f"Unknown MF_TYPE: '{mf_type}'")
    return extract_mf


# ============================================================
#  HELPER - sigmoid threshold label assignment (gradual drift)
# ============================================================
 
def assign_labels_gradual(stream, config):
    """
    Assign concept labels using the sigmoid threshold method
    from Komorniczak et al.
    Produces 25 concepts for n_drifts=6, n_chunks=5000.
 
    Returns
    -------
    labels : list of int, length n_chunks
        Concept label for each chunk.
    """
    # sigmoid value per chunk: same as their e[chunk]
    e = stream._sigmoid(stream.concept_sigmoid_spacing, stream.n_drifts)[1][::config['chunk_size']]
 
    concept = 0
    decreasing = True
    labels = []
 
    for chunk in range(config['n_chunks']):
        # threshold logic from Komorniczak et al.
        if decreasing:
            if concept % 4 == 0:
                if e[chunk] < 0.9:
                    concept += 1
            if concept % 4 == 1:
                if e[chunk] < 0.75:
                    concept += 1
            if concept % 4 == 2:
                if e[chunk] < 0.25:
                    concept += 1
            if concept % 4 == 3:
                if e[chunk] < 0.1:
                    concept += 1
                    decreasing = False
        else:
            if concept % 4 == 0:
                if e[chunk] > 0.1:
                    concept += 1
            if concept % 4 == 1:
                if e[chunk] > 0.25:
                    concept += 1
            if concept % 4 == 2:
                if e[chunk] > 0.75:
                    concept += 1
            if concept % 4 == 3:
                if e[chunk] > 0.9:
                    concept += 1
                    decreasing = True
 
        labels.append(concept)
 
    return labels


# ============================================================
#  HELPER - extract meta-features for one stream
# ============================================================

def extract_metafeatures_for_stream(random_state, extract_mf, drift_type, n_drifts, concept_sigmoid_spacing):
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

    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[], accuracy_window_size=CHUNK_SIZE,
        class_window_size=CHUNK_SIZE)

    # pass 1: run ABFS, save concept_selector
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

    # concept labels: method depends on drift type
    if drift_type == 'sudden':
        # majority vote from concept_selector
        # equivalent to their threshold method for sudden drift
        concept_selector_saved = stream.concept_selector.copy()
 
    elif drift_type == 'gradual':
        # sigmoid threshold method
        all_chunk_labels = assign_labels_gradual(stream, config)

    # pass 2: extract meta-features
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[], accuracy_window_size=CHUNK_SIZE,
        class_window_size=CHUNK_SIZE)

    meta_features  = []
    concept_labels = []
    window_indices = []
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
            window_indices.append(window_counter)

        wt_prev = wt
        window_counter += 1

    # assign labels after extraction
    for idx in window_indices:
        if drift_type == 'sudden':
            chunk_start = idx * CHUNK_SIZE
            chunk_end = min((idx + 1) * CHUNK_SIZE, len(concept_selector_saved))
            chunk_concepts = concept_selector_saved[chunk_start:chunk_end]
            concept_labels.append(int(np.bincount(chunk_concepts).argmax()))
        elif drift_type == 'gradual':
            concept_labels.append(all_chunk_labels[idx])

    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y


# ============================================================
#  MAIN - sweep across all MF_TYPES
# ============================================================
np.random.seed(1233)  # for shuffles in classifier sweep
for drift_type, n_drifts, concept_sigmoid_spacing in DRIFT_CONFIGS:
    n_concepts = 25 if drift_type == 'gradual' else n_drifts + 1
    random_baseline = 1 / n_concepts

    print(f"\n{'#'*60}")
    print(f"DRIFT TYPE: {drift_type} ({n_concepts} concepts)")
    print(f"{'#'*60}")

    all_mean_ba = {}
    all_std_ba = {}
    all_median_ba = {}
    all_mean_f1 = {}
    all_std_f1 = {}
    all_mean_kappa = {}
    all_std_kappa = {}

    for mf_type, mf_label, n_mf in MF_CONFIGS:

        print(f"\n{'='*60}")
        print(f"Meta-features: {mf_label} ({n_mf}) | Drift: {drift_type}")
        print(f"{'='*60}")

        extract_mf = make_extract_mf(mf_type)

        all_clf_res_ba = []
        all_clf_res_f1 = []
        all_clf_res_kappa = []

        for rep_id, rs in enumerate(RANDOM_STATES):
            print(f"Replication {rep_id+1}/{N_REPLICATIONS} (seed={rs})...")
            X, y = extract_metafeatures_for_stream(rs, extract_mf, drift_type, n_drifts, concept_sigmoid_spacing)

            mean_ba, std_ba, clf_res_ba, mean_f1, std_f1, clf_res_f1, mean_kappa, std_kappa, clf_res_kappa = run_classifier_sweep(X, y, shuffle_seed=None)
            all_clf_res_ba.append(clf_res_ba)
            all_clf_res_f1.append(clf_res_f1)
            all_clf_res_kappa.append(clf_res_kappa)

            print(f"{'Clf':<6s} {'Mean BA':>8s} {'Mean F1':>8s} {'Mean K':>8s}")
            for clf_id, (name, _) in enumerate(BASE_CLFS):
                print(f"{name:<6s} {mean_ba[clf_id]:>8.4f} {mean_f1[clf_id]:>8.4f} {mean_kappa[clf_id]:>8.4f}")

        all_clf_res_ba = np.array(all_clf_res_ba)
        all_clf_res_f1 = np.array(all_clf_res_f1)
        all_clf_res_kappa = np.array(all_clf_res_kappa)

        np.save(os.path.join(RESULTS_DIR, f'clf_ba_{mf_type}_{drift_type}.npy'), all_clf_res_ba)
        np.save(os.path.join(RESULTS_DIR, f'clf_f1_{mf_type}_{drift_type}.npy'), all_clf_res_f1)
        np.save(os.path.join(RESULTS_DIR, f'clf_kappa_{mf_type}_{drift_type}.npy'), all_clf_res_kappa)
        print(f"Saved to {RESULTS_DIR}")

        all_mean_ba[mf_type] = np.mean(all_clf_res_ba, axis=(0, 1))
        all_std_ba[mf_type] = np.std(all_clf_res_ba, axis=(0, 1))
        all_median_ba[mf_type] = np.median(all_clf_res_ba, axis=(0, 1))
        all_mean_f1[mf_type] = np.mean(all_clf_res_f1, axis=(0, 1))
        all_std_f1[mf_type] = np.std(all_clf_res_f1, axis=(0, 1))
        all_mean_kappa[mf_type] = np.mean(all_clf_res_kappa, axis=(0, 1))
        all_std_kappa[mf_type] = np.std(all_clf_res_kappa, axis=(0, 1))


    # ================
    #  SUMMARY TABLE
    # ================
    print_summary_table_experiment1(all_mean_ba, MF_CONFIGS, BASE_CLFS, drift_type, n_concepts, random_baseline)


    # ============================================================
    #  COMPARISON - our ABFS meta-features vs replication_check_1a
    # ============================================================
    rc_path = os.path.join(RESULTS_DIR, f'clf_replication_ba_{drift_type}.npy')

    if os.path.exists(rc_path):
        rc_raw = np.load(rc_path) # shape: (n_measures, n_replications, n_folds, n_clfs)
        plot_heatmap_balanced_accuracy_comparison(all_mean_ba, all_std_ba, all_median_ba, rc_raw, MEASURES, BASE_CLFS,
        drift_type, n_concepts, FIGURES_DIR,exp_label='1a',filename=f'heatmap_comparison_komorniczak_ABFS_{drift_type}.png')
    else:
        print(f"\nWarning: {rc_path} not found - run replication_check.py first.")