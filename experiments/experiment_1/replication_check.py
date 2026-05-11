# Verification that our evaluation pipeline correctly replicates
# the results of Komorniczak et al.
#
# Uses the same streams (same StreamGenerator configuration and
# seeds), the same concept labelling, and the same evaluation
# protocol (classifier_sweep_komor.py) as our own pipeline in
# evaluate_concept_classification.py. The only difference is
# the meta-features: here we load their pre-extracted statistical
# meta-features (produced by E1_extract_synthetic.py) instead
# of computing ABFS-based relevance scores.
#
# Two comparisons are produced:
#   1. Heatmap of our evaluation protocol results (classifier_sweep_komor.py)
#      applied to their meta-features. One row per measure group.
#   2. Side-by-side comparison of our evaluation protocol results vs E2
#      (their original evaluation script). This confirms that any difference
#      in results between their meta-features and ours is due solely to the meta-features
#      and not to a difference in the evaluation protocol.
#
# Steps:
#   1. Load their .npy files from the results/ folder
#   2. Extract meta-feature vectors and concept labels
#   3. Run the classifier sweep (classifier_sweep_komor.py)
#   4. Compare output against E2 (their original evaluation script)

# This is to verify whether we get the same results as in the comparison.py when running our evaluation protocol on their meta-features.
# If we do, then we can be confident that any difference in results between their meta-features and ours is due solely to the meta-features and not to a difference in the evaluation protocol.

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..') # points to experiments/ where classifier_sweep_komor.py is
import os
# path to results folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) # go up two levels to project root
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results/experiment_1')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results/experiment_1', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


from classifier_sweep_komor import run_classifier_sweep, BASE_CLFS
from plot_results import print_summary_table_experiment1, plot_heatmap_balanced_accuracy

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

# ========================================================
#  LOAD E2 RESULTS (external/komorniczak/results/clf.npy)
# ========================================================
clf_res_e2 = np.load(os.path.join(THEIR_RESULTS_PATH, 'clf.npy'))


# ========================================
#  LOAD AND EVALUATE THEIR META-FEATURES
# ========================================
np.random.seed(1233)
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

        data_drift = data[drift_idx]

        rep_mean_ba = []

        for rep_id, rep_data in enumerate(data_drift):
            X = rep_data[:, :-1].astype(float) # meta-feature vectors
            y = rep_data[:, -1].astype(int) # concept labels

            mean_ba, std_ba, clf_res = run_classifier_sweep(X, y, shuffle_seed=None)
            rep_mean_ba.append(mean_ba)

        overall_mean = np.mean(rep_mean_ba, axis=0)
        overall_std  = np.std(rep_mean_ba, axis=0)
        all_mean_ba[measure] = overall_mean
        all_std_ba[measure] = overall_std

    # save replication results for this drift type
    rc_matrix = np.array([all_mean_ba[m] for m in MEASURES])
    rc_std_matrix = np.array([all_std_ba[m] for m in MEASURES])
    np.save(os.path.join(RESULTS_DIR, f'clf_replication_{drift_type}.npy'), rc_matrix)
    np.save(os.path.join(RESULTS_DIR, f'clf_replication_std_{drift_type}.npy'), rc_std_matrix)

    MEASURE_CONFIGS = [(m, m, None) for m in MEASURES]
    
    # ============================================================
    #  SUMMARY TABLE
    # ============================================================
    print_summary_table_experiment1(all_mean_ba=all_mean_ba, MF_CONFIGS=MEASURE_CONFIGS, BASE_CLFS=BASE_CLFS,
        drift_type=drift_type, n_concepts=n_concepts, random_baseline=1/n_concepts)

    # ================================================================
    #  HEATMAP: our evaluation protocol applied to their meta-features
    # ================================================================
    plot_heatmap_balanced_accuracy(all_mean_ba=all_mean_ba, all_std_ba=all_std_ba, MF_CONFIGS=MEASURE_CONFIGS, BASE_CLFS=BASE_CLFS,
    drift_type=drift_type, n_concepts=n_concepts, FIGURES_DIR=FIGURES_DIR,
    title_prefix='Replication check - Komorniczak et al. ', filename=f'heatmap_replication_{drift_type}.png', figsize=(10, 5))


    # ============================================================
    #  COMPARISON - our protocol vs E2
    # ============================================================
    e2_matrix = np.zeros((len(MEASURES), len(BASE_CLFS)))
    for m_idx, measure in enumerate(MEASURES):
        e2_matrix[m_idx] = np.mean(clf_res_e2[m_idx, drift_idx], axis=(0, 1))

    rc_matrix = np.array([all_mean_ba[m] for m in MEASURES])
    diff = rc_matrix - e2_matrix

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, matrix, title in [
        (axes[0], e2_matrix, f'E2 output ({drift_type})'),
        (axes[1], rc_matrix, f'Replication check ({drift_type})'),
        (axes[2], diff, f'Difference: Our protocol - E2 ({drift_type})')
    ]:
        vmin, vmax = (0.0, 1.0) if 'Difference' not in title else (-0.1, 0.1)
        cmap = 'Blues' if 'Difference' not in title else 'RdYlGn'
        ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        for i in range(len(MEASURES)):
            for j in range(len(BASE_CLFS)):
                val = matrix[i, j]
                txt_color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=txt_color)
        ax.set_xticks(range(len(BASE_CLFS)))
        ax.set_xticklabels(clf_names, fontsize=9)
        ax.set_yticks(range(len(MEASURES)))
        ax.set_yticklabels(MEASURES, fontsize=9)
        ax.set_title(title, fontsize=10)

    fig.suptitle(f'Replication check - Our protocol vs E2 ({drift_type})', fontsize=13)
    plt.tight_layout()
    comp_path = os.path.join(FIGURES_DIR, f'compare_our_protocol_vs_e2_{drift_type}.png')
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {comp_path}")
