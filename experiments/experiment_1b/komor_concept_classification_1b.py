# Evaluation of Komorniczak meta-features using our
# no-shuffle CV protocol (Experiment 1b).
#
# Loads their pre-extracted statistical meta-features
# (produced by E1_extract_synthetic.py) and evaluates them
# using the same no-shuffle RepeatedStratifiedKFold protocol
# as evaluate_concept_classification_1b.py.
#
# Unlike replication_check_1a.py, this is NOT a replication
# of their original results — the protocol differs (no-shuffle
# CV vs their shuffled CV-based E2). The purpose is to provide
# a fair within-1b comparison: their features vs our features
# under the same no-shuffle protocol.


# Metrics computed: balanced accuracy, macro F1, Cohen's Kappa.
# Results saved as clf_komor_concept_classif_ba_*.npy, clf_komor_concept_classif_f1_*.npy,
# clf_komor_concept_classif_kappa_*.npy

# Steps:
#   1. Load their .npy files from the results/ folder
#   2. Extract meta-feature vectors and concept labels
#   3. Run the classifier sweep (classifier_sweep_komor.py)
#   4. Save the raw results (balanced accuracy, F1, Kappa) for each replication, fold, and classifier in results/experiment_1b

# RUN: their meta-features evaluated with our no-shuffle CV protocol
# COMPARISON: their meta-features (no-shuffle) vs our meta-features (no-shuffle)
# COMPARE: meta-features (their statistical vs our ABFS-based)


# This is to verify whether we get the same results as in the comparison.py when running our evaluation protocol on their meta-features.
# If we do, then we can be confident that any difference in results between their meta-features and ours is due solely to the meta-features and not to a difference in the evaluation protocol.

# It generates 6 .npy files in results/experiment_1b (3 per drift type):
#   clf_komor_concept_classif_ba_sudden.npy
#   clf_komor_concept_classif_f1_sudden.npy
#   clf_komor_concept_classif_kappa_sudden.npy
#   clf_komor_concept_classif_ba_gradual.npy
#   clf_komor_concept_classif_f1_gradual.npy
#   clf_komor_concept_classif_kappa_gradual.npy

# Each file has shape (n_measures, n_replications, n_folds, n_clfs)




import numpy as np
import sys
sys.path.append('..') # points to experiments/ where classifier_sweep_komor.py is
sys.path.append('../..') # points to project root where plot_results.py is
import os
# path to results folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) # go up two levels to project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results/experiment_1b')

os.makedirs(RESULTS_DIR, exist_ok=True)

from classifier_sweep_komor import run_classifier_sweep, BASE_CLFS
from plot_results import print_summary_table_experiment1

# ============================================================
#  CONFIGURATION
# ============================================================

# absolute path to their results folder
THEIR_RESULTS_PATH = os.path.join(PROJECT_ROOT, 'external', 'komorniczak', 'results')


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


# ========================================
#  LOAD AND EVALUATE THEIR META-FEATURES
# ========================================

for drift_idx, drift_type, n_concepts in DRIFT_TYPES:
    print(f"\n{'='*60}")
    print(f"Drift type: {drift_type}")
    print(f"{'='*60}")

    all_mean_ba = {}
    all_std_ba = {}

    measure_reps_ba = {m: [] for m in MEASURES}
    measure_reps_f1 = {m: [] for m in MEASURES}
    measure_reps_kappa = {m: [] for m in MEASURES}

    for m_idx, measure in enumerate(MEASURES):
        print(f"\nMeasure group: {measure}")
        print(f"{'-'*56}")

        data = np.load(f'{THEIR_RESULTS_PATH}/{measure}.npy')

        data_drift = data[drift_idx]

        rep_mean_ba = []

        for rep_id, rep_data in enumerate(data_drift):
            X = rep_data[:, :-1].astype(float) # meta-feature vectors
            y = rep_data[:, -1].astype(int) # concept labels

            mean_ba, std_ba, clf_res_ba, mean_f1, std_f1, clf_res_f1, mean_kappa, std_kappa, clf_res_kappa = run_classifier_sweep(X, y, shuffle = False) # no shuffle
            
            rep_mean_ba.append(mean_ba)
            measure_reps_ba[measure].append(clf_res_ba)
            measure_reps_f1[measure].append(clf_res_f1)
            measure_reps_kappa[measure].append(clf_res_kappa)

        overall_mean = np.mean(rep_mean_ba, axis=0)
        overall_std = np.std(rep_mean_ba, axis=0)
        all_mean_ba[measure] = overall_mean
        all_std_ba[measure] = overall_std



    # save raw results for this drift type
    # shape: (n_measures, n_replications, n_folds, n_clfs)
    rc_raw_ba = np.array([measure_reps_ba[m] for m in MEASURES])
    rc_raw_f1 = np.array([measure_reps_f1[m] for m in MEASURES])
    rc_raw_kappa = np.array([measure_reps_kappa[m] for m in MEASURES])

    np.save(os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_ba_{drift_type}.npy'),    rc_raw_ba)
    np.save(os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_f1_{drift_type}.npy'),    rc_raw_f1)
    np.save(os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_kappa_{drift_type}.npy'), rc_raw_kappa)
    print(f"Saved to {RESULTS_DIR}")

    MEASURE_CONFIGS = [(m, m, None) for m in MEASURES]
    
    # ============================================================
    #  SUMMARY TABLE
    # ============================================================
    print_summary_table_experiment1(all_mean_ba=all_mean_ba, MF_CONFIGS=MEASURE_CONFIGS, BASE_CLFS=BASE_CLFS,
        drift_type=drift_type, n_concepts=n_concepts, random_baseline=1/n_concepts)