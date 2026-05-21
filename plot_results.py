
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


def plot_heatmap_balanced_accuracy_comparison(all_mean_ba, all_std_ba, rc_raw, MEASURES, BASE_CLFS,
        drift_type, n_concepts, FIGURES_DIR, exp_label='1a', filename=None):
    """
    One figure with two heatmaps side by side:
      - Left: Komorniczak results (their features, our protocol)
      - Right: ABFS results (raw v2.0 and raw+temporal v2.1 only)

    Parameters
    ----------
    all_mean_ba : dict  {mf_type: np.ndarray (n_clfs,)}
    all_std_ba  : dict  {mf_type: np.ndarray (n_clfs,)}
    rc_raw      : np.ndarray, shape (n_measures, n_replications, n_folds, n_clfs)
    MEASURES    : list of str
    BASE_CLFS   : list of (name, clf)
    drift_type  : str
    n_concepts  : int
    FIGURES_DIR : str
    exp_label   : str  e.g. '1a' or '1b'
    filename    : str or None
    """
    clf_names = [name for name, _ in BASE_CLFS]
    n_clfs    = len(BASE_CLFS)

    # Komorniczak: average over replications and folds
    rc_matrix     = np.mean(rc_raw, axis=(1, 2))  # (n_measures, n_clfs)
    rc_std_matrix = np.std(rc_raw,  axis=(1, 2))  # (n_measures, n_clfs)

    # ABFS: raw and raw_temporal only
    abfs_configs = [
        ('raw',          'Raw scores (v2.0)'),
        ('raw_temporal', 'Raw + temporal (v2.1)'),
    ]
    abfs_matrix     = np.array([all_mean_ba[mf_type] for mf_type, _ in abfs_configs])
    abfs_std_matrix = np.array([all_std_ba[mf_type]  for mf_type, _ in abfs_configs])

    # right panel y-axis labels include mean std
    abfs_row_labels = [
        f'{label}\n(±{abfs_std_matrix[i].mean():.3f})'
        for i, (_, label) in enumerate(abfs_configs)
    ]

    n_measures = len(MEASURES)
    n_abfs     = len(abfs_configs)

    fig, axes = plt.subplots(
        1, 2, figsize=(18, max(5, n_measures * 0.65)),
        gridspec_kw={'width_ratios': [3, 1]})

    # ── left: Komorniczak ──────────────────────────────────────
    ax = axes[0]
    ax.imshow(rc_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(n_measures):
        for j in range(n_clfs):
            val = rc_matrix[i, j]
            std = rc_std_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}\n(±{std:.3f})', ha='center', va='center',
                fontsize=8, color=txt_color, linespacing=1.4)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=9)
    ax.set_yticks(range(n_measures))
    ax.set_yticklabels(MEASURES, fontsize=9)
    ax.set_title('Komorniczak meta-features - balanced accuracy', fontsize=10)

    # ── right: ABFS ────────────────────────────────────────────
    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(n_abfs):
        for j in range(n_clfs):
            val = abfs_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=8, color=txt_color)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=9)
    ax.set_yticks(range(n_abfs))
    ax.set_yticklabels(abfs_row_labels, fontsize=9)
    ax.set_title('ABFS meta-features - balanced accuracy', fontsize=10)

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    fig.suptitle(
        f'Komorniczak vs ABFS - {drift_type} drift ({n_concepts} concepts) - experiment [{exp_label}]',
        fontsize=11)
    plt.tight_layout()

    if filename is None:
        filename = f'heatmap_comparison_komorniczak_ABFS_{drift_type}.png'
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparison heatmap saved to {path}")