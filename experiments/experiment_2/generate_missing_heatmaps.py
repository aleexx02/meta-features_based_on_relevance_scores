# generate_missing_heatmaps.py
# ============================================================
# Generates CV and prequential comparison heatmaps for all
# grid cells that have .npy files in results/experiment_2/.
#
# Skips cells where the heatmap figure already exists.
#
# Run from experiments/experiment_2/:
#   python generate_missing_heatmaps.py
# ============================================================
 
import numpy as np
import os
import sys
 
sys.path.append('..')
sys.path.append('../..')
 
from classifier_sweep_komor import BASE_CLFS
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
from plot_results import plot_heatmap_balanced_accuracy_comparison_exp2
 
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
 
# all cells that have .npy files
MISSING = [
    ('chunk500_ninf3_gradual',   'gradual', 25),
    ('chunk500_ninf5_gradual',   'gradual', 25),
    ('chunk500_ninf10_gradual',  'gradual', 25),
    ('chunk500_ninf15_gradual',  'gradual', 25),
    ('chunk1000_ninf3_gradual',  'gradual', 25),
    ('chunk1000_ninf5_gradual',  'gradual', 25),
    ('chunk1000_ninf10_gradual', 'gradual', 25),
    ('chunk1000_ninf15_gradual', 'gradual', 25),
    ('chunk1000_ninf15_sudden',  'sudden',  21),
]
 
for tag, drift_type, n_concepts in MISSING:
 
    # --- CV heatmap ---
    cv_fig = os.path.join(FIGURES_DIR,
                 f'heatmap_comparison_komorniczak_ABFS_cv_{tag}.png')
    if os.path.exists(cv_fig):
        print(f"CV heatmap already exists, skipping: cv_{tag}")
    else:
        cv_abfs_path  = os.path.join(RESULTS_DIR, f'cv_abfs_ba_{tag}.npy')
        cv_komor_path = os.path.join(RESULTS_DIR, f'cv_komor_ba_{tag}.npy')
        if not os.path.exists(cv_abfs_path) or not os.path.exists(cv_komor_path):
            print(f"Missing .npy files for cv_{tag}, skipping.")
        else:
            cv_abfs  = np.load(cv_abfs_path)
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
                tag             = f'cv_{tag}',
                FIGURES_DIR     = FIGURES_DIR,
            )
            print(f"Generated: heatmap_comparison_komorniczak_ABFS_cv_{tag}.png")
 
    # --- Prequential heatmap ---
    preq_fig = os.path.join(FIGURES_DIR,
                  f'heatmap_comparison_komorniczak_ABFS_preq_{tag}.png')
    if os.path.exists(preq_fig):
        print(f"Preq heatmap already exists, skipping: preq_{tag}")
    else:
        pr_abfs_path  = os.path.join(RESULTS_DIR, f'preq_abfs_ba_{tag}.npy')
        pr_komor_path = os.path.join(RESULTS_DIR, f'preq_komor_ba_{tag}.npy')
        if not os.path.exists(pr_abfs_path) or not os.path.exists(pr_komor_path):
            print(f"Missing .npy files for preq_{tag}, skipping.")
        else:
            pr_abfs  = np.load(pr_abfs_path)
            pr_komor = np.load(pr_komor_path)
            # shape: (n_reps, n_windows, n_clfs) — take final window
            plot_heatmap_balanced_accuracy_comparison_exp2(
                mean_ba_abfs    = np.mean(pr_abfs[:, -1, :],    axis=0),
                std_ba_abfs     = np.std(pr_abfs[:, -1, :],     axis=0),
                median_ba_abfs  = np.median(pr_abfs[:, -1, :],  axis=0),
                mean_ba_komor   = np.mean(pr_komor[:, -1, :],   axis=0),
                std_ba_komor    = np.std(pr_komor[:, -1, :],    axis=0),
                median_ba_komor = np.median(pr_komor[:, -1, :], axis=0),
                BASE_CLFS       = BASE_CLFS_PREQUENTIAL,
                drift_type      = drift_type,
                n_concepts      = n_concepts,
                tag             = f'preq_{tag}',
                FIGURES_DIR     = FIGURES_DIR,
            )
            print(f"Generated: heatmap_comparison_komorniczak_ABFS_preq_{tag}.png")
 
print("\nDone.")