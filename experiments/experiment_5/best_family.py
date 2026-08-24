# Prints the best Komorniczak family per cell for Exp 5, using the final-window
# BA averaged over the 5 reps -- the same quantity the heatmap/summary report.
# Run from where results/experiment_5 is reachable; adjust RESULTS_DIR if needed.
import os, glob, numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_5')
MEASURES = ['clustering','complexity','concept','general','info-theory',
            'itemset','landmarking','model-based','statistical']
ABFS_VERSIONS = [
    'aggstats',
    'raw',
    'raw_temporal'
]
# discover cells from the vanilla files (one per cell)
cells = sorted(os.path.basename(p)[len('preq_vanilla_ba_'):-4]
               for p in glob.glob(os.path.join(RESULTS_DIR,'preq_vanilla_ba_*.npy')))

for cell in cells:
    best_fam, best_ba = None, -1.0
    for m in MEASURES:
        p = os.path.join(RESULTS_DIR, f'preq_komor_{m}_ba_{cell}.npy')
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        arr = np.squeeze(arr)            # kills any stray singleton axis
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected (n_windows, n_clfs), got {arr.shape}")
        ba = float(arr[-1, :].max())
        if ba > best_ba:
            best_fam, best_ba = m, ba
    print(f"{cell:32s} best family = {best_fam:12s}  BA={best_ba:.3f}")


print("\nBEST ReMF VARIANT PER CELL")
print("=" * 60)

for cell in cells:
    best_version, best_ba = None, -1.0

    for version in ['aggstats', 'raw', 'raw_temporal']:
        p = os.path.join(
            RESULTS_DIR,
            f'preq_abfs_{version}_ba_{cell}.npy'
        )

        if not os.path.exists(p):
            continue

        arr = np.load(p)   # (n_reps, n_windows, n_clfs)

        ba = float(
            np.mean(arr[:, -1, :], axis=0).max()
        )

        if ba > best_ba:
            best_version = version
            best_ba = ba

    print(
        f"{cell:32s} best ReMF = "
        f"{best_version:12s}  BA={best_ba:.3f}"
    )