# plot_results.py

# Shared file for printing the summary table and generating
# the heatmaps to compare results. Used by both
# evaluate_concept_classification.py and replication_check.py.

import numpy as np
import matplotlib.pyplot as plt
import os



def print_sanity_check_summary(stream_name, is_streamlearn, mf_type, mf_names, meta_features, concept_labels, raw_vectors, n_features):
    unique_concepts = np.unique(concept_labels)

    print(f"*** Sanity check summary: {stream_name} ***")
    print(f"{'-'*25}")
    print(f"Stream: {stream_name}")
    print(f"ABFS: {'ABFS_match' if is_streamlearn else 'ABFS_mismatch'}")
    print(f"Meta-features: {mf_type} ({len(mf_names)} features)")
    print(f"Total windows: {len(meta_features)}")
    print(f"Unique concepts: {len(unique_concepts)} {list(unique_concepts)}")

    print(f"\n\n*** Meta-feature means per concept: ***")
    print(f"{'':22s}", end='')
    for c in unique_concepts:
        print(f"{'concept '+str(c):>14s}", end='')
    # only show |Delta| for binary concepts (SEA/STAGGER)
    if len(unique_concepts) == 2:
        print(f"{'|Delta|':>14s}")
    else:
        print()
    print(f"{'-' * (20 + 14 * (len(unique_concepts) + (1 if len(unique_concepts) == 2 else 0)))}")


    for k, name in enumerate(mf_names):
        print(f"{name:<22s}", end='')
        vals = []
        for c in unique_concepts:
            mean_val = meta_features[concept_labels == c, k].mean()
            vals.append(mean_val)
            print(f"{mean_val:>14.4f}", end='')
        if len(unique_concepts) == 2:
            print(f"{abs(vals[-1] - vals[0]):>14.4f}")
        else:
            print()

    if raw_vectors.shape[0] > 0 and raw_vectors.ndim == 2:
        print(f"\n\n*** Mean raw relevance score per feature per concept: ***")
        print(f"{'':14s}", end='')
        for c in unique_concepts:
            print(f"{'concept '+str(c):>12s}", end='')
        if len(unique_concepts) == 2:
            print(f"{'|Delta|':>12s}")
        else:
            print()
        for j in range(n_features):
            print(f"  f{j+1:<10d}", end='')
            vals = []
            for c in unique_concepts:
                mean_val = raw_vectors[concept_labels == c, j].mean()
                vals.append(mean_val)
                print(f"{mean_val:>12.4f}", end='')
            if len(unique_concepts) == 2:
                print(f"{abs(vals[-1] - vals[0]):>12.4f}")
            else:
                print()
    else:
        print(f"\n\n*** Raw relevance vectors not available for {mf_type} meta-features. ***")



def print_summary_table_experiment1(all_mean_ba, MF_CONFIGS, BASE_CLFS, drift_type, n_concepts, random_baseline, benchmark_label='Komorniczak et al. (MLP, sudden)', benchmark_value=0.881):
    clf_names = [name for name, _ in BASE_CLFS]

    print(f"\n{'='*60}")
    print(f"Summary - {drift_type} drift")
    print(f"{'='*60}")
    print(f"\n{'Meta-features':<25s}", end='')
    for name in clf_names:
        print(f"{name:>10s}", end='')
    print()
    print('-' * (25 + 10 * len(clf_names)))
    for mf_type, mf_label, _ in MF_CONFIGS:
        print(f"{mf_label:<25s}", end='')
        for clf_id in range(len(BASE_CLFS)):
            print(f"{all_mean_ba[mf_type][clf_id]:>10.4f}", end='')
        print()
    print(f"\n{benchmark_label}: {benchmark_value:.4f}")
    print(f"Random baseline (1/{n_concepts}):  {random_baseline:.4f}")


def plot_heatmap_balanced_accuracy_comparison(all_mean_ba, all_std_ba, all_median_ba, rc_raw, MEASURES, BASE_CLFS,drift_type, n_concepts, FIGURES_DIR, exp_label='1a', filename=None):
    """
    One figure with two heatmaps side by side:
      - Left: Komorniczak results (their features, our protocol)
      - Right: ABFS results (aggstats v1.1, raw v2.0 and raw+temporal v2.1)

    Parameters
    ----------
    all_mean_ba: dict  {mf_type: np.ndarray (n_clfs,)}
    all_std_ba: dict  {mf_type: np.ndarray (n_clfs,)}
    all_median_ba: dict  {mf_type: np.ndarray (n_clfs,)}
    rc_raw: np.ndarray, shape (n_measures, n_replications, n_folds, n_clfs)
    MEASURES: list of str
    BASE_CLFS: list of (name, clf)
    drift_type: str
    n_concepts: int
    FIGURES_DIR: str
    exp_label: str  e.g. '1a', '1b', '1c'
    filename: str or None
    """
    clf_names = [name for name, _ in BASE_CLFS]
    n_clfs = len(BASE_CLFS)

    # Komorniczak: average over replications and folds
    rc_matrix = np.mean(rc_raw, axis=(1, 2))  # (n_measures, n_clfs)
    rc_std_matrix = np.std(rc_raw, axis=(1, 2))  # (n_measures, n_clfs)
    rc_median_matrix = np.median(rc_raw, axis=(1, 2))  # (n_measures, n_clfs)

    # ABFS 
    abfs_configs = [('aggstats', 'Aggregate stats (v1.1)'), ('raw', 'Raw scores (v2.0)'), ('raw_temporal', 'Raw + temporal (v2.1)')]

    abfs_matrix = np.array([all_mean_ba[mf_type] for mf_type, _ in abfs_configs])
    abfs_std_matrix = np.array([all_std_ba[mf_type] for mf_type, _ in abfs_configs])
    abfs_median_matrix = np.array([all_median_ba[mf_type] for mf_type, _ in abfs_configs])
    abfs_row_labels = [label for _, label in abfs_configs]

    n_measures = len(MEASURES)
    n_abfs = len(abfs_configs)

    fig, axes = plt.subplots(1, 2, figsize=(26, max(5, n_measures * 0.75)), gridspec_kw={'width_ratios': [3, 1.5]})

    # Left heatmap: Komorniczak 
    ax = axes[0]
    ax.imshow(rc_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(n_measures):
        for j in range(n_clfs):
            val = rc_matrix[i, j]
            std = rc_std_matrix[i, j]
            median = rc_median_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}\n(±{std:.3f})\nmed:{median:.3f}', ha='center', va='center', fontsize=11, color=txt_color, linespacing=1.4)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(n_measures))
    ax.set_yticklabels(MEASURES, fontsize=10)
    ax.set_title('Komorniczak meta-features - balanced accuracy', fontsize=12)

    # Right heatmap: ABFS
    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    ax.set_ylim(n_abfs - 0.5, -0.5)
    for i in range(n_abfs):
        for j in range(n_clfs):
            val = abfs_matrix[i, j]
            std = abfs_std_matrix[i, j]
            median = abfs_median_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}\n(±{std:.3f})\nmed:{median:.3f}',ha='center', va='center', fontsize=11,color=txt_color, linespacing=1.4)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(n_abfs))
    ax.set_yticklabels(abfs_row_labels, fontsize=10)
    ax.set_title('ABFS meta-features - balanced accuracy', fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f'Komorniczak vs ABFS - {drift_type} drift ({n_concepts} concepts) - experiment [{exp_label}]',fontsize=15)
    plt.tight_layout()

    if filename is None:
        filename = f'heatmap_comparison_komorniczak_ABFS_{drift_type}.png'
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparison heatmap saved to {path}")






def plot_heatmap_balanced_accuracy_comparison_exp2(
    mean_ba_abfs, std_ba_abfs, median_ba_abfs,
    mean_ba_komor, std_ba_komor, median_ba_komor,
    BASE_CLFS, drift_type, n_concepts, tag, FIGURES_DIR):
    """
    Side-by-side balanced accuracy heatmap for Experiment 2.
    One row per meta-feature set (left: Komorniczak statistical,
    right: ABFS raw scores v2.0).

    Parameters
    ----------
    mean_ba_abfs   : np.ndarray, shape (n_clfs,)
    std_ba_abfs    : np.ndarray, shape (n_clfs,)
    median_ba_abfs : np.ndarray, shape (n_clfs,)
    mean_ba_komor  : np.ndarray, shape (n_clfs,)
    std_ba_komor   : np.ndarray, shape (n_clfs,)
    median_ba_komor: np.ndarray, shape (n_clfs,)
    BASE_CLFS      : list of (name, clf)
    drift_type     : str
    n_concepts     : int
    tag            : str  e.g. 'chunk200_ninf10_sudden'
    FIGURES_DIR    : str
    """
    clf_names = [name for name, _ in BASE_CLFS]
    n_clfs    = len(BASE_CLFS)

    # shape (1, n_clfs) for imshow
    komor_matrix        = mean_ba_komor[np.newaxis, :]
    komor_std_matrix    = std_ba_komor[np.newaxis, :]
    komor_median_matrix = median_ba_komor[np.newaxis, :]

    abfs_matrix         = mean_ba_abfs[np.newaxis, :]
    abfs_std_matrix     = std_ba_abfs[np.newaxis, :]
    abfs_median_matrix  = median_ba_abfs[np.newaxis, :]

    fig, axes = plt.subplots(1, 2, figsize=(22, 3),
                             gridspec_kw={'width_ratios': [1, 1]})

    for ax, matrix, std_mat, med_mat, title, row_label in [
        (axes[0], komor_matrix, komor_std_matrix, komor_median_matrix,
         'Komorniczak meta-features - balanced accuracy', 'Statistical'),
        (axes[1], abfs_matrix,  abfs_std_matrix,  abfs_median_matrix,
         'ABFS meta-features - balanced accuracy',       'Raw scores (v2.0)'),
    ]:
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0,
                       cmap='Blues', aspect='auto')
        for j in range(n_clfs):
            val    = matrix[0, j]
            std    = std_mat[0, j]
            median = med_mat[0, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, 0,
                    f'{val:.3f}\n(±{std:.3f})\nmed:{median:.3f}',
                    ha='center', va='center', fontsize=11,
                    color=txt_color, linespacing=1.4)
        ax.set_xticks(range(n_clfs))
        ax.set_xticklabels(clf_names, fontsize=10)
        ax.set_yticks([0])
        ax.set_yticklabels([row_label], fontsize=10)
        ax.set_title(title, fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f'Komorniczak vs ABFS — {drift_type} drift '
        f'({n_concepts} concepts) — {tag}',
        fontsize=13)
    plt.tight_layout()

    filename = f'heatmap_comparison_komorniczak_ABFS_{tag}.png'
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved to {path}")