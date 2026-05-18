# ============================================================
# Analysis of Experiment 1 results.
#
# Loads pre-computed results (.npy files) from results/experiment_1/ and
# produces the following analyses for each drift type and
# meta-feature set:
#
#   1. Sanity check plots (per replication × drift type):
#      - Relevance scores over time
#      - Meta-features over windows
#      - PCA projection of meta-feature vectors
#
#   2. Performance variance across replications:
#      - Mean balanced accuracy per replication per classifier
#      - One plot per meta-feature set per drift type
#
#   3. SHAP analysis:
#      - Mean absolute SHAP values per meta-feature (MLP)
#      - One plot per meta-feature set per drift type
#        averaged over replications
#
#   4. Additional metrics heatmaps (F1, Kappa):
#      - Same format as balanced accuracy heatmaps
#      - Includes Komorniczak (statistical) as baseline
#
# Inputs (from results/experiment_1/):
#   clf_ba_*.npy, clf_f1_*.npy, clf_kappa_*.npy
#   clf_replication_ba_*.npy, clf_replication_f1_*.npy,
#   clf_replication_kappa_*.npy
#
# Outputs saved to results/experiment_1/figures/analysis/
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import clone
import shap
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..') # experiments/ to access classifier_sweep_komor.py
sys.path.append('../..') # project root to access plot_results.py

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures_raw, extract_metafeatures_raw_temporal,
    MF_NAMES_RAW, MF_NAMES_RAW_TEMPORAL
)
from classifier_sweep_komor import BASE_CLFS
from plot_results import print_sanity_check_summary

# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)

# =================
#  CONFIGURATION
# =================
N_CHUNKS = 5000
CHUNK_SIZE = 200
N_FEATURES = 10
WARMUP_WINDOWS = 10
SCORE_INTERVAL = 100
N_REPLICATIONS = 5

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

MF_CONFIGS = [
    ('raw', 'Raw scores (v2.0)', MF_NAMES_RAW, 5),
    ('raw_temporal', 'Raw + temporal (v2.1)', MF_NAMES_RAW_TEMPORAL, 6),
]

DRIFT_CONFIGS = [
    ('sudden',  20, 9999, 21),
    ('gradual',  6, 5, 25),
]

ABFS_MF_CONFIGS_FULL = [
    ('aggstats', 'Aggregate stats (v1.1)'),
    ('raw', 'Raw scores (v2.0)'),
    ('raw_temporal', 'Raw + temporal (v2.1)'),
]

MEASURES = ['clustering', 'complexity', 'concept', 'general',
    'info-theory', 'itemset', 'landmarking', 'model-based', 'statistical']
STATISTICAL_IDX = MEASURES.index('statistical')

clf_names = [name for name, _ in BASE_CLFS]

palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', '#5e5151', '#08332b',
    '#000000', '#a9a9a9', '#ff69b4', '#00ced1', '#ff8c00'
]

# ============
#  HELPERS
# ============

def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing, stream.n_drifts)[1][::chunk_size]
    concept = 0
    decreasing = True
    labels = []
    for chunk in range(n_chunks):
        if decreasing:
            if concept % 4 == 0:
                if e[chunk] < 0.9: concept += 1
            if concept % 4 == 1:
                if e[chunk] < 0.75: concept += 1
            if concept % 4 == 2:
                if e[chunk] < 0.25: concept += 1
            if concept % 4 == 3:
                if e[chunk] < 0.1:
                    concept += 1
                    decreasing = False
        else:
            if concept % 4 == 0:
                if e[chunk] > 0.1: concept += 1
            if concept % 4 == 1:
                if e[chunk] > 0.25: concept += 1
            if concept % 4 == 2:
                if e[chunk] > 0.75: concept += 1
            if concept % 4 == 3:
                if e[chunk] > 0.9:
                    concept += 1
                    decreasing = True
        labels.append(concept)
    return np.array(labels)


def get_concept_boundaries(concept_labels_all, n_chunks):
    return [i for i in range(1, n_chunks) if concept_labels_all[i] != concept_labels_all[i-1]]


def extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing):
    """Run ABFS on one stream and extract meta-features for all MF types."""
    config = {
        'n_drifts': n_drifts,
        'n_chunks': N_CHUNKS,
        'chunk_size': CHUNK_SIZE,
        'n_features': N_FEATURES,
        'n_informative': N_FEATURES,
        'n_redundant': 0,
        'n_repeated': 0,
        'concept_sigmoid_spacing': concept_sigmoid_spacing,
        'random_state': rs
    }
    stream = StreamGenerator(**config)

    # pass 1: relevance scores over time
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
    scores_over_time = []
    instance_counter = 0
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
            if instance_counter % SCORE_INTERVAL == 0:
                scores_over_time.append(abfs.relevance_scores())
            instance_counter += 1
    concept_selector_saved = stream.concept_selector.copy()
    scores_over_time = np.array(scores_over_time)

    # concept labels
    if drift_type == 'sudden':
        concept_labels_all = np.array([
            int(np.bincount(concept_selector_saved[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]).argmax())
            for i in range(N_CHUNKS)])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, CHUNK_SIZE)

    boundaries = get_concept_boundaries(concept_labels_all, N_CHUNKS)

    # pass 2: meta-features per MF type
    results = {}
    for mf_type, mf_label, mf_names, _ in MF_CONFIGS:
        abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
        meta_features = []
        concept_labels = []
        wt_prev = None
        window_counter = 0

        stream.reset()
        for X_chunk, y_chunk in stream:
            for i in range(len(X_chunk)):
                abfs.update(X_chunk[i], y_chunk[i])
            wt = abfs.relevance_scores()
            if window_counter >= WARMUP_WINDOWS:
                if mf_type == 'raw':
                    mf = extract_metafeatures_raw(wt)
                else:
                    mf = extract_metafeatures_raw_temporal(wt=wt, wt_prev=wt_prev)
                meta_features.append(mf)
                concept_labels.append(concept_labels_all[window_counter])
            wt_prev = wt
            window_counter += 1

        X = np.array(meta_features, dtype=float)
        y = np.array(concept_labels)
        X[np.isnan(X)] = 1
        X[np.isinf(X)] = 1
        results[mf_type] = {'X': X, 'y': y}

    return scores_over_time, concept_labels_all, boundaries, results


# ==========================
#  1. SANITY CHECK PLOTS
# ==========================
print("\n" + "="*60)
print("1. SANITY CHECK PLOTS")
print("="*60)

for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
    for rep_id, rs in enumerate(RANDOM_STATES):
        print(f"\nDrift: {drift_type} | Rep {rep_id+1}/{N_REPLICATIONS} (seed={rs})")

        scores_over_time, concept_labels_all, boundaries, mf_results = extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing)

        # relevance scores over time
        fig, ax = plt.subplots(figsize=(14, 4))
        for j in range(N_FEATURES):
            ax.plot(scores_over_time[:, j], label=f'f{j+1}')
        for b in boundaries:
            ax.axvline(x=b * CHUNK_SIZE // SCORE_INTERVAL,color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axvline(x=-1, color='red', linestyle='--', linewidth=1.0,label='concept boundary')
        ax.set_xlabel('Time (x100 instances)')
        ax.set_ylabel('Relevance score')
        ax.set_title(f'ABFS relevance scores - {drift_type} drift (seed={rs})')
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        fname = os.path.join(FIGURES_DIR,f'relevance_scores_{drift_type}_rep{rep_id}.png')
        fig.savefig(fname, dpi=150)
        plt.close()
        print(f"\tRelevance scores saved at: '{fname}'")

        for mf_type, mf_label, mf_names, n_mf_cols in MF_CONFIGS:
            X = mf_results[mf_type]['X']
            y = mf_results[mf_type]['y']
            unique_concepts = np.unique(y)

            # meta-features over windows
            fig, axes = plt.subplots(2, n_mf_cols, figsize=(4 * n_mf_cols, 6))
            axes = axes.flatten()
            for k, name in enumerate(mf_names):
                axes[k].plot(X[:, k], color='steelblue')
                for b in boundaries:
                    drift_w = b - WARMUP_WINDOWS
                    if drift_w > 0:
                        axes[k].axvline(x=drift_w, color='red',linestyle='--', linewidth=1.0)
                axes[k].set_title(name, fontsize=9)
                axes[k].set_xlabel('Window')
                axes[k].set_ylabel('Value')
            fig.suptitle(
                f'Meta-features ({mf_type}) - {drift_type} drift (seed={rs})',
                fontsize=11)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR,
                f'metafeatures_{mf_type}_{drift_type}_rep{rep_id}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"\tMeta-features per window saved at: '{fname}'")

            # PCA
            colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_concepts)}
            pca = PCA(n_components=2)
            projected = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(8, 5))
            for c in unique_concepts:
                mask = y == c
                ax.scatter(projected[mask, 0], projected[mask, 1],color=colors[c], label=f'concept {c}',alpha=0.6, edgecolors='none', s=30)
            ax.set_xlabel(
                f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
            ax.set_ylabel(
                f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
            ax.set_title(
                f'PCA ({mf_type}) - {drift_type} drift (seed={rs})')
            ax.legend(ncol=4, fontsize=8)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR,
                f'pca_{mf_type}_{drift_type}_rep{rep_id}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"\tPCA plot saved at: '{fname}'")

            print_sanity_check_summary(f'{drift_type} drift (seed={rs})', True,mf_type, mf_names, X, y,X[:, :N_FEATURES], N_FEATURES)


# ===============================================
#  2. PERFORMANCE VARIANCE ACROSS REPLICATIONS
# ================================================
print("\n" + "="*60)
print("2. PERFORMANCE VARIANCE ACROSS REPLICATIONS")
print("="*60)

for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
    for mf_type, mf_label, mf_names, _ in MF_CONFIGS:

        path = os.path.join(RESULTS_DIR, f'clf_ba_{mf_type}_{drift_type}.npy')
        if not os.path.exists(path):
            print(f"\tWarning: {path} not found, skipping.")
            continue

        clf_res = np.load(path)
        # shape: (n_replications, n_folds, n_clfs)
        rep_means = np.mean(clf_res, axis=1)  # (n_replications, n_clfs)

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(N_REPLICATIONS)
        width = 0.15
        for clf_id, (name, _) in enumerate(BASE_CLFS):
            ax.bar(x + clf_id * width, rep_means[:, clf_id],width=width, label=name, alpha=0.8)
        ax.axhline(y=1/n_concepts, color='red', linestyle='--',linewidth=1.0, label='random baseline')
        ax.set_xlabel('Replication')
        ax.set_ylabel('Mean balanced accuracy')
        ax.set_title(f'Performance variance - {mf_label} - {drift_type} drift')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f'Rep {i+1}\n(seed={RANDOM_STATES[i]})' for i in range(N_REPLICATIONS)],fontsize=8)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fname = os.path.join(FIGURES_DIR,f'variance_{mf_type}_{drift_type}.png')
        fig.savefig(fname, dpi=150)
        plt.close()
        print(f"\tPerformance variance plot saved at: '{fname}'")


# ============================================================
#  3. SHAP ANALYSIS
# ============================================================
print("\n" + "="*60)
print("3. SHAP ANALYSIS")
print("="*60)

for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
    for mf_type, mf_label, mf_names, _ in MF_CONFIGS:
        print(f"\nSHAP: {mf_label} - {drift_type} drift")

        all_X, all_y = [], []
        for rep_id, rs in enumerate(RANDOM_STATES):
            _, _, _, mf_results = extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing)
            all_X.append(mf_results[mf_type]['X'])
            all_y.append(mf_results[mf_type]['y'])

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        X_all[np.isnan(X_all)] = 1
        X_all[np.isinf(X_all)] = 1

        # train MLP on full dataset
        mlp = clone(dict(BASE_CLFS)['MLP'])
        mlp.fit(X_all, y_all)

        # SHAP
        explainer  = shap.KernelExplainer(mlp.predict_proba, shap.sample(X_all, 100))
        shap_values = explainer.shap_values(shap.sample(X_all, 200), nsamples=100)

        # mean absolute SHAP per feature across all classes
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        mean_abs_shap = np.mean(mean_abs_shap, axis=0)

        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(mf_names)), mean_abs_shap[sorted_idx],color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(mf_names)))
        ax.set_xticklabels([mf_names[i] for i in sorted_idx],rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean absolute SHAP value')
        ax.set_title(f'SHAP feature importance - {mf_label} - {drift_type} drift\n(MLP, averaged over {N_REPLICATIONS} replications)')
        fig.tight_layout()
        fname = os.path.join(FIGURES_DIR,f'shap_{mf_type}_{drift_type}.png')
        fig.savefig(fname, dpi=150)
        plt.close()
        print(f"\tSHAP plot saved: {fname}")


# ==============================================
#  4. ADDITIONAL METRICS HEATMAPS (F1, KAPPA)
# ==============================================
print("\n" + "="*60)
print("4. ADDITIONAL METRICS HEATMAPS")
print("="*60)

for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
    for metric, metric_label in [('f1', 'F1'), ('kappa', "Kappa")]:

        # load Komorniczak baseline (statistical measure group)
        rc_path = os.path.join(RESULTS_DIR,f'clf_replication_{metric}_{drift_type}.npy')
        if os.path.exists(rc_path):
            rc_raw = np.load(rc_path)
            # shape: (n_measures, n_replications, n_folds, n_clfs)
            rc_mean = np.mean(rc_raw[STATISTICAL_IDX], axis=(0, 1))
            rc_std = np.std(rc_raw[STATISTICAL_IDX],  axis=(0, 1))
        else:
            rc_mean = None
            print(f"\tWarning: {rc_path} not found")

        # collect ABFS rows
        all_rows = []
        for mf_type, mf_label in ABFS_MF_CONFIGS_FULL:
            path = os.path.join(RESULTS_DIR,f'clf_{metric}_{mf_type}_{drift_type}.npy')
            if not os.path.exists(path):
                print(f"\tWarning: {path} not found, skipping.")
                continue
            raw = np.load(path)
            mean_vals = np.mean(raw, axis=(0, 1))
            std_vals = np.std(raw,  axis=(0, 1))
            all_rows.append((mf_label, mean_vals, std_vals))

        if rc_mean is not None:
            all_rows.append(('Komorniczak (statistical)', rc_mean, rc_std))

        if not all_rows:
            continue

        matrix = np.array([r[1] for r in all_rows])
        matrix_std = np.array([r[2] for r in all_rows])
        row_labels = [r[0] for r in all_rows]

        # print summary
        print(f"\n\t{metric_label} - {drift_type} drift")
        print(f"\t\t{'Meta-features':<25s}", end='')
        for name in clf_names:
            print(f"{name:>10s}", end='')
        print()
        print(f"\t\t{'-' * (25 + 10 * len(clf_names))}")
        for label, mean_vals, _ in all_rows:
            print(f"\t\t{label:<25s}", end='')
            for v in mean_vals:
                print(f"{v:>10.3f}", end='')
            print()

        # heatmap
        fig, ax = plt.subplots(
            figsize=(10, max(3, len(all_rows) * 0.9)))
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0,cmap='Blues', aspect='auto')
        for i in range(len(all_rows)):
            for j in range(len(BASE_CLFS)):
                val = matrix[i, j]
                std = matrix_std[i, j]
                txt_color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}\n(±{std:.3f})',ha='center', va='center', fontsize=8,
                    color=txt_color, linespacing=1.4)
        ax.set_xticks(range(len(BASE_CLFS)))
        ax.set_xticklabels(clf_names, fontsize=9)
        ax.set_yticks(range(len(all_rows)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(f'{metric_label} - {drift_type} drift ({n_concepts} concepts)',fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        fig.tight_layout()
        fname = os.path.join(FIGURES_DIR, f'heatmap_{metric}_{drift_type}.png')
        fig.savefig(fname, dpi=150)
        plt.close()
        print(f"\tHeatmap for metric {metric_label} saved at: '{fname}'")