# generate_missing_heatmaps.py
# ============================================================
# Generates comparison heatmaps for grid cells that were
# computed by the old script (before the heatmap call was
# added to evaluate_concept_classification_2.py).
#
# Loads the already-saved .npy files from results/experiment_2/
# and generates the missing heatmap figures.
#
# Run from experiments/experiment_2/:
#   python generate_missing_heatmaps.py
# ============================================================
 
import numpy as np
import os
import sys
 
sys.path.append('..')       # experiments/
sys.path.append('../..')    # project root
 
from classifier_sweep_komor import BASE_CLFS
from plot_results import plot_heatmap_balanced_accuracy_comparison_exp2
 
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
 
# cells that have .npy files but are missing heatmap figures
# add any further missing cells here if needed
MISSING = [
    ('chunk100_ninf3_sudden',  'sudden',  21),
    ('chunk100_ninf5_sudden',  'sudden',  21),
]
 
for tag, drift_type, n_concepts in MISSING:
    cv_abfs_path  = os.path.join(RESULTS_DIR, f'cv_abfs_ba_{tag}.npy')
    cv_komor_path = os.path.join(RESULTS_DIR, f'cv_komor_ba_{tag}.npy')
    fig_path      = os.path.join(FIGURES_DIR,
                        f'heatmap_comparison_komorniczak_ABFS_{tag}.png')
 
    if os.path.exists(fig_path):
        print(f"Already exists, skipping: {fig_path}")
        continue
 
    if not os.path.exists(cv_abfs_path) or not os.path.exists(cv_komor_path):
        print(f"Missing .npy files for {tag}, skipping.")
        continue
 
    cv_abfs  = np.load(cv_abfs_path)   # (n_reps, n_folds, n_clfs)
    cv_komor = np.load(cv_komor_path)
 
    plot_heatmap_balanced_accuracy_comparison_exp2(
        mean_ba_abfs    = np.mean(cv_abfs,    axis=(0, 1)),
        std_ba_abfs     = np.std(cv_abfs,     axis=(0, 1)),
        median_ba_abfs  = np.median(cv_abfs,  axis=(0, 1)),
        mean_ba_komor   = np.mean(cv_komor,   axis=(0, 1)),
        std_ba_komor    = np.std(cv_komor,    axis=(0, 1)),
        median_ba_komor = np.median(cv_komor, axis=(0, 1)),
        BASE_CLFS       = BASE_CLFS,
        drift_type      = drift_type,
        n_concepts      = n_concepts,
        tag             = tag,
        FIGURES_DIR     = FIGURES_DIR,
    )
    print(f"Generated: heatmap_comparison_komorniczak_ABFS_{tag}.png")
 
print("\nDone.")
