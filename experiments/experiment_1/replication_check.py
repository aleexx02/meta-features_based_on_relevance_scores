# Verification that our evaluation pipeline correctly replicates
# the results of Komorniczak et al. (2024).
#
# Uses the same streams (same StreamGenerator configuration and
# seeds), the same concept labelling, and the same evaluation
# protocol (classifier_sweep_komor.py) as our own pipeline in
# evaluate_concept_classification.py. The only difference is
# the meta-features: here we load their pre-extracted statistical
# meta-features (produced by E1_extract_synthetic.py) instead
# of computing ABFS-based relevance scores.
#
# If the output matches Figure 12 of their paper, the comparison
# between their meta-features and ours in
# evaluate_concept_classification.py is fully controlled —
# any difference in balanced accuracy is due solely to the
# meta-features.
#
# Steps:
#   1. Load their .npy files from the results/ folder
#   2. Extract meta-feature vectors and concept labels
#   3. Run the classifier sweep (classifier_sweep_komor.py)
#   4. Compare output against Figure 12 of their paper

import numpy as np
import sys
sys.path.append('..') # points to experiments/ where classifier_sweep_komor.py is
import os
# path to results folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) # go up two levels to project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


from classifier_sweep_komor import run_classifier_sweep, print_results, BASE_CLFS
from plot_results import plot_heatmap_balanced_accuracy

# ============================================================
#  CONFIGURATION
# ============================================================

# absolute path to their results folder
THEIR_RESULTS_PATH = '/home/ptr/code_komor/results' # path in cluster


# all measure groups produced by E1_extract_synthetic.py
MEASURES = [
    'clustering',
    'complexity',
    'concept',
    'general',
    'info-theory',
    'itemset',
    'landmarking',
    'model-based',
    'statistical'
]

# drift types: sudden + gradual
DRIFT_TYPES = [
    (0, 'sudden',  21),
    (1, 'gradual', 25)
]

clf_names = [name for name, _ in BASE_CLFS]


# ============================================================
#  LOAD AND EVALUATE
# ============================================================

for drift_idx, drift_type, n_concepts in DRIFT_TYPES:
    print(f"\n{'='*60}")
    print(f"Drift type: {drift_type}")
    print(f"{'='*60}")

    all_mean_ba = {}
    all_std_ba  = {}

    for m_idx, measure in enumerate(MEASURES):
        print(f"\n  Measure group: {measure}")
        print(f"  {'-'*56}")

        data = np.load(f'{THEIR_RESULTS_PATH}/{measure}.npy')
        # shape: (n_drift_types, n_replications, n_chunks, n_metafeatures + 1)

        data_drift = data[drift_idx]
        # shape: (n_replications, n_chunks, n_metafeatures + 1)

        rep_mean_ba = []

        for rep_id, rep_data in enumerate(data_drift):
            X = rep_data[:, :-1].astype(float)  # meta-feature vectors
            y = rep_data[:, -1].astype(int)     # concept labels

            mean_ba, std_ba, clf_res = run_classifier_sweep(X, y)
            rep_mean_ba.append(mean_ba)

        overall_mean = np.mean(rep_mean_ba, axis=0)
        overall_std  = np.std(rep_mean_ba, axis=0)
        all_mean_ba[measure] = overall_mean
        all_std_ba[measure] = overall_std


        print_results(overall_mean, overall_std, label=f"  {drift_type} / {measure}")



    # ============================================================
    #  HEATMAP
    # ============================================================
    MEASURE_CONFIGS = [(m, m, None) for m in MEASURES]
    plot_heatmap_balanced_accuracy(all_mean_ba=all_mean_ba, all_std_ba=all_std_ba, MF_CONFIGS=MEASURE_CONFIGS, BASE_CLFS=BASE_CLFS,
    drift_type=drift_type, n_concepts=n_concepts, FIGURES_DIR=FIGURES_DIR,
    title_prefix='Replication check — Komorniczak et al. ', filename=f'heatmap_replication_{drift_type}.png', figsize=(10, 5))
