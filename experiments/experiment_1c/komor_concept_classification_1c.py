# Evaluation of Komorniczak et al. meta-features using our
# prequential evaluation protocol (Experiment 1c).
#
# Loads their pre-extracted statistical meta-features
# (produced by E1_extract_synthetic.py) and evaluates them
# using the same prequential (test-then-train) protocol as
# evaluate_concept_classification_1c.py. The first
# WARMUP_WINDOWS windows are skipped to align with our
# ABFS-based meta-features.
#
# Unlike replication_check_1a.py, this is NOT a replication
# of their original results — the protocol differs (prequential
# vs their CV-based E2). The purpose is to provide a fair
# within-1c comparison: their features vs our features under
# the same prequential protocol.

# Metrics computed: balanced accuracy, macro F1, Cohen's Kappa.
# Results saved as clf_komor_concept_classif_ba_*.npy, clf_komor_concept_classif_f1_*.npy,
# clf_komor_concept_classif_kappa_*.npy

# Steps:
#   1. Load their .npy files from the results/ folder
#   2. Extract meta-feature vectors and concept labels
#   3. Run the classifier sweep for prequential evaluation
#   4. Save the raw results (balanced accuracy, F1, Kappa) for each replication, window, and classifier in results/experiment_1c

# RUN: their meta-features evaluated with our prequential protocol
# COMPARISON: their meta-features (prequential) vs our meta-features (prequential)
# COMPARE: meta-features (their static vs our ABFS-based)

# Input: Komorniczak's raw pre-extracted meta-features, written once by
#          E1_extract_synthetic.py (external/komorniczak/results/synthetic/),
#          not regenerated here.
# Output: classifier-sweep results (clf_komor_concept_classif_*.npy) -> results/experiment_1c/

# It generates 6 .npy files in results/experiment_1c (3 per drift type):
#   clf_komor_concept_classif_ba_sudden.npy
#   clf_komor_concept_classif_f1_sudden.npy
#   clf_komor_concept_classif_kappa_sudden.npy
#   clf_komor_concept_classif_ba_gradual.npy
#   clf_komor_concept_classif_f1_gradual.npy
#   clf_komor_concept_classif_kappa_gradual.npy

# Each file has shape (n_measures, n_replications, n_windows, n_clfs)
# where n_windows = N_CHUNKS - WARMUP_WINDOWS = 4990


import numpy as np
import os
import sys
sys.path.append('..')
sys.path.append('../..')

from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1c')
THEIR_RESULTS_PATH = os.path.join(PROJECT_ROOT, 'external', 'komorniczak', 'results', 'synthetic')

os.makedirs(RESULTS_DIR, exist_ok=True)

WARMUP_WINDOWS = 10

MEASURES = ['clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical']

DRIFT_TYPES = [(0, 'sudden',  21), (1, 'gradual', 25)]

clf_names = [name for name, _ in BASE_CLFS_PREQUENTIAL]


for drift_idx, drift_type, n_concepts in DRIFT_TYPES:
    print(f"\n{'='*60}")
    print(f"Drift type: {drift_type} ({n_concepts} concepts)")
    print(f"{'='*60}")

    # shape: (n_measures, n_replications, n_windows, n_clfs)
    all_ba = []
    all_f1 = []
    all_kap = []

    for m_idx, measure in enumerate(MEASURES):
        print(f"\nMeasure: {measure}")

        data = np.load(f'{THEIR_RESULTS_PATH}/{measure}.npy') # shape: (n_drift_types, n_replications, n_windows, n_features+1)
        data_drift = data[drift_idx]  # (n_replications, 5000, n_features+1)

        rep_ba = []
        rep_f1 = []
        rep_kap = []

        for rep_id, rep_data in enumerate(data_drift):
            rep_data = rep_data[WARMUP_WINDOWS:] # skip first WARMUP_WINDOWS to align with our meta-features
            X = rep_data[:, :-1].astype(float)
            y = rep_data[:, -1].astype(int)

            X[np.isnan(X)] = 1
            X[np.isinf(X)] = 1

            print(f"Rep {rep_id+1}: X={X.shape}, concepts={np.unique(y)}")

            mean_ba, std_ba, traj_ba, mean_f1, std_f1, traj_f1, mean_kappa, std_kappa, traj_kappa = run_prequential_sweep(X, y)
            
            rep_ba.append(traj_ba)
            rep_f1.append(traj_f1)
            rep_kap.append(traj_kappa)

            print(f"{'Clf':<6s} {'Final BA':>10s} {'Final F1':>10s} {'Final K':>10s}")
            for clf_id, name in enumerate(clf_names):
                print(f"{name:<6s} {traj_ba[-1, clf_id]:>10.4f} "
                      f"{traj_f1[-1, clf_id]:>10.4f} "
                      f"{traj_kappa[-1, clf_id]:>10.4f}")

        all_ba.append(rep_ba)
        all_f1.append(rep_f1)
        all_kap.append(rep_kap)

    # shape: (n_measures, n_replications, n_windows, n_clfs)
    rc_ba = np.array(all_ba)
    rc_f1 = np.array(all_f1)
    rc_kap = np.array(all_kap)

    np.save(os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_ba_{drift_type}.npy'),    rc_ba)
    np.save(os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_f1_{drift_type}.npy'),    rc_f1)
    np.save(os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_kappa_{drift_type}.npy'), rc_kap)
    print(f"\nSaved to {RESULTS_DIR}")