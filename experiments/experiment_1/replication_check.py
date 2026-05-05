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
from classifier_sweep_komor import run_classifier_sweep, print_results, BASE_CLFS


# ============================================================
#  CONFIGURATION
# ============================================================

# absolute path to their results folder
THEIR_RESULTS_PATH = '/home/ptr/code_komor/results'


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
    (0, 'sudden'),
    (1, 'gradual')
]


# ============================================================
#  LOAD AND EVALUATE
# ============================================================

for drift_idx, drift_type in DRIFT_TYPES:
    print(f"\n{'='*60}")
    print(f"Drift type: {drift_type}")
    print(f"{'='*60}")

    for measure in MEASURES:
        print(f"\n  Measure group: {measure}")
        print(f"  {'-'*56}")

        data = np.load(f'{THEIR_RESULTS_PATH}/{measure}.npy')
        # shape: (n_drift_types, n_replications, n_chunks, n_metafeatures + 1)

        data_drift = data[drift_idx]
        # shape: (n_replications, n_chunks, n_metafeatures + 1)

        all_mean_ba = []

        for rep_id, rep_data in enumerate(data_drift):
            X = rep_data[:, :-1].astype(float)  # meta-feature vectors
            y = rep_data[:, -1].astype(int)     # concept labels

            mean_ba, std_ba, clf_res = run_classifier_sweep(X, y)
            all_mean_ba.append(mean_ba)

        overall_mean = np.mean(all_mean_ba, axis=0)
        overall_std  = np.std(all_mean_ba, axis=0)
        print_results(overall_mean, overall_std,
                      label=f"  {drift_type} / {measure}")