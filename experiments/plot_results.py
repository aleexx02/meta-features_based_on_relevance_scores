
# Shared file for printing the summary table and generating
# the heatmaps to compare results. Used by both
# evaluate_concept_classification.py and replication_check.py.

import numpy as np
import matplotlib.pyplot as plt
import os



def print_sanity_check_summary(stream_name, is_streamlearn, mf_type, mf_names, meta_features, concept_labels, raw_vectors, n_features):
    unique_concepts = np.unique(concept_labels)

    print(f"\n{'='*60}")
    print(f"Sanity check summary")
    print(f"{'='*60}")
    print(f"Stream: {stream_name}")
    print(f"ABFS: {'ABFS_match' if is_streamlearn else 'ABFS_mismatch'}")
    print(f"Meta-features: {mf_type} ({len(mf_names)} features)")
    print(f"Total windows: {len(meta_features)}")
    print(f"Unique concepts: {len(unique_concepts)} {list(unique_concepts)}")

    print(f"\nMeta-feature means per concept:")
    print(f"{'':22s}", end='')
    for c in unique_concepts:
        print(f"{'concept '+str(c):>14s}", end='')
    print()
    for k, name in enumerate(mf_names):
        print(f"  {name:<22s}", end='')
        for c in unique_concepts:
            mean_val = meta_features[concept_labels == c, k].mean()
            print(f"{mean_val:>14.4f}", end='')
        print()

    if raw_vectors.shape[0] > 0 and raw_vectors.ndim == 2:
        print(f"\nMean raw relevance score per feature per concept:")
        print(f"{'':14s}", end='')
        for c in unique_concepts:
            print(f"{'concept '+str(c):>12s}", end='')
        print()
        for j in range(n_features):
            print(f"  f{j+1:<10d}", end='')
            for c in unique_concepts:
                mean_val = raw_vectors[concept_labels == c, j].mean()
                print(f"{mean_val:>12.4f}", end='')
            print()
    else:
        print(f"\nRaw relevance vectors not available for {mf_type} meta-features.")



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


def plot_heatmap_balanced_accuracy(all_mean_ba, all_std_ba, MF_CONFIGS, BASE_CLFS, drift_type, n_concepts, FIGURES_DIR, title_prefix='',
    filename=None, figsize=(8, 3.5)):
    clf_names = [name for name, _ in BASE_CLFS]
    n_mf_sets = len(MF_CONFIGS)
    n_clfs = len(BASE_CLFS)
    matrix = np.zeros((n_mf_sets, n_clfs))
    matrix_std = np.zeros((n_mf_sets, n_clfs))
    row_labels = [label for _, label, _ in MF_CONFIGS]

    for row_idx, (mf_type, _, _) in enumerate(MF_CONFIGS):
        matrix[row_idx] = all_mean_ba[mf_type]
        matrix_std[row_idx] = all_std_ba[mf_type]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix, vmin=0.05, vmax=1.0, cmap='Blues', aspect='auto')

    for i in range(n_mf_sets):
        for j in range(n_clfs):
            val = matrix[i, j]
            std = matrix_std[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}\n(±{std:.3f})', ha='center', va='center', fontsize=9, color=txt_color, linespacing=1.4)

    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=11)
    ax.set_yticks(range(n_mf_sets))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(f'{title_prefix}Balanced accuracy (mean ± std)\n'
    f'{drift_type} drift ({n_concepts} concepts)', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()

    if filename is None:
        filename = f'heatmap_{drift_type}.png'
    heatmap_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nHeatmap saved to {heatmap_path}")