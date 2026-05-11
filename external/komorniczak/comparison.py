
# Compare E2 clf.npy results against Figure 12 of Komorniczak paper
# Produces a side-by-side heatmap for sudden and gradual drift.
# Is their reported performance of their meta-features (Figure 12) consistent with the results we get when we run their code?

import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results') # folder where the results from their code are stored (after running E2)
SAVE_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'results', 'experiment_1', 'figures') # folder where the comparison figure will be saved
os.makedirs(SAVE_PATH, exist_ok=True)


# =============================
#  EXPECTED VALUES (from paper)
# =============================

measures = ["clustering", "complexity", "concept", "general", "info-theory", "itemset", "landmarking", "model-based", "statistical"]

clf_names = ['GNB', 'KNN', 'SVM', 'DT', 'MLP']

expected_sudden = np.array([
    [0.295, 0.251, 0.223, 0.276, 0.282],  # clustering
    [0.445, 0.349, 0.307, 0.480, 0.360],  # complexity
    [0.156, 0.119, 0.079, 0.176, 0.064],  # concept
    [0.048, 0.048, 0.048, 0.049, 0.048],  # general
    [0.363, 0.153, 0.175, 0.309, 0.191],  # info-theory
    [0.117, 0.092, 0.115, 0.084, 0.048],  # itemset
    [0.334, 0.253, 0.309, 0.226, 0.276],  # landmarking
    [0.119, 0.083, 0.083, 0.117, 0.076],  # model-based
    [0.866, 0.506, 0.444, 0.752, 0.881],  # statistical
])

expected_gradual = np.array([
    [0.169, 0.138, 0.122, 0.152, 0.144],  # clustering
    [0.244, 0.210, 0.129, 0.262, 0.196],  # complexity
    [0.100, 0.079, 0.047, 0.115, 0.048],  # concept
    [0.040, 0.040, 0.040, 0.039, 0.040],  # general
    [0.229, 0.114, 0.119, 0.180, 0.126],  # info-theory
    [0.092, 0.078, 0.058, 0.073, 0.041],  # itemset
    [0.201, 0.156, 0.141, 0.143, 0.144],  # landmarking
    [0.096, 0.067, 0.060, 0.091, 0.061],  # model-based
    [0.557, 0.267, 0.159, 0.422, 0.515],  # statistical
])

# =================================================
#  LOAD E2 RESULTS (actual results from their code)
# =================================================

clf_res = np.load(os.path.join(RESULTS_DIR, 'clf.npy'))


# mean over reps and folds
actual_sudden  = np.mean(clf_res[:, 0], axis=(1, 2))
actual_gradual = np.mean(clf_res[:, 1], axis=(1, 2))

# ========
#  PLOT
# ========

def plot_comparison(paper, ours, drift_type, ax_paper, ax_ours, ax_diff):
    diff = ours - paper
    vmin, vmax = 0.0, 1.0

    im1 = ax_paper.imshow(paper, vmin=vmin, vmax=vmax, cmap='Blues', aspect='auto')
    im2 = ax_ours.imshow(ours,  vmin=vmin, vmax=vmax, cmap='Blues', aspect='auto')
    im3 = ax_diff.imshow(diff,  vmin=-0.1, vmax=0.1,  cmap='RdYlGn', aspect='auto')

    for ax, matrix in [(ax_paper, paper), (ax_ours, ours), (ax_diff, diff)]:
        for i in range(len(measures)):
            for j in range(len(clf_names)):
                val = matrix[i, j]
                txt_color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=txt_color)

        ax.set_xticks(range(len(clf_names)))
        ax.set_xticklabels(clf_names, fontsize=9)
        ax.set_yticks(range(len(measures)))
        ax.set_yticklabels(measures, fontsize=9)

    ax_paper.set_title(f'Expected values ({drift_type})', fontsize=10)
    ax_ours.set_title(f'Actual output ({drift_type})', fontsize=10)
    ax_diff.set_title(f'Difference: Actual - expected ({drift_type})', fontsize=10)

    return im1, im3


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_comparison(expected_sudden,  actual_sudden,  'sudden',  axes[0,0], axes[0,1], axes[0,2])
plot_comparison(expected_gradual, actual_gradual, 'gradual', axes[1,0], axes[1,1], axes[1,2])

fig.suptitle('Replication check - Actual vs Figure 12 (Komorniczak)', fontsize=13)
plt.tight_layout()

save_path = os.path.join(SAVE_PATH, 'compare_expected_vs_actual_results_komorniczak_synthetic.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved to {save_path}")