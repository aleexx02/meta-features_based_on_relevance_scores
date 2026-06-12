# analysis_3.py
# ============================================================
# Analysis of Experiment 3 results (annotated real-world streams).
#
# Usage:
#   python experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics
#
# Flags:
#   --sanity      : relevance scores, meta-features per version, PCA per version
#   --performance : cumulative BA trajectory per stream
#   --shap        : SHAP — all 4 classifiers, per stream per ABFS version
#   --metrics     : F1 and Kappa heatmaps per stream
# ============================================================
#
# Inputs (from results/experiment_3/):
#   preq_abfs_{version}_ba_{stream}.npy     shape: (n_windows, n_clfs)
#   preq_komor_{measure}_ba_{stream}.npy    shape: (n_windows, n_clfs)
#   abfs_y_{stream}.npy                     shape: (n_windows,)
#
# Outputs saved to results/experiment_3/figures/analysis/
# ============================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))

import strlearn as sl
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures, extract_metafeatures_raw,
    extract_metafeatures_raw_temporal, MF_NAMES_AGGSTATS,
)
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone as skclone
from sklearn.decomposition import PCA
import shap


# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sanity',      action='store_true')
parser.add_argument('--performance', action='store_true')
parser.add_argument('--shap',        action='store_true')
parser.add_argument('--metrics',     action='store_true')
args = parser.parse_args()

RUN_SANITY      = args.sanity
RUN_PERFORMANCE = args.performance
RUN_SHAP        = args.shap
RUN_METRICS     = args.metrics

print(f"\nRunning analysis for Experiment 3")
print(f"Sanity      : {RUN_SANITY}")
print(f"Performance : {RUN_PERFORMANCE}")
print(f"SHAP        : {RUN_SHAP}")
print(f"Metrics     : {RUN_METRICS}")


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

REAL_STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')
REAL_GT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_gt')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
FIGURES_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_3', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================
CHUNK_SIZE = 300

REAL_STREAMS = [
    'INSECTS-abrupt_imbalanced_norm',
    'INSECTS-gradual_imbalanced_norm',
    'INSECTS-incremental_imbalanced_norm',
    'poker-lsn-1-2vsAll-pruned',
]

N_FEATURES = {
    'INSECTS-abrupt_imbalanced_norm':       33,
    'INSECTS-gradual_imbalanced_norm':      33,
    'INSECTS-incremental_imbalanced_norm':  33,
    'poker-lsn-1-2vsAll-pruned':            10,
}

N_CONCEPTS = {
    s: len(np.load(os.path.join(REAL_GT_DIR, f'{s}.npy'))) + 1
    for s in REAL_STREAMS
}

MEASURES = [
    'clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical',
]

ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {
    'aggstats':     'Aggstats (v1.1)',
    'raw':          'Raw scores (v2.0)',
    'raw_temporal': 'Raw + temporal (v2.1)',
}

CLF_NAMES = [n for n, _ in BASE_CLFS_PREQUENTIAL]
N_CLFS    = len(CLF_NAMES)

CLF_COLORS = {
    'GNB': '#e6194b', 'KNN': '#3cb44b',
    'HT':  '#f032e6', 'MLP': '#911eb4',
}

PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#469990', '#9a6324',
    '#000075', '#800000', '#a9a9a9', '#000000', '#556b2f',
    '#d2691e', '#5e5151', '#08332b', '#2ecc71', '#ff69b4',
    '#00ced1', '#ff8c00', '#c9a0dc', '#7b3f91', '#e6ac00',
    '#808000', '#42d4f4', '#f032e6', '#469990', '#9a6324',
    '#000075', '#800000',
]


SHAP_CLFS = [
    ('GNB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('HT',  DecisionTreeClassifier(random_state=11313)),
    ('MLP', MLPClassifier(random_state=11313)),
]


# ============================================================
#  HELPERS
# ============================================================

def load(prefix, stream_name, optional=False):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)


def load_gt(stream_name):
    return np.load(os.path.join(REAL_GT_DIR, f'{stream_name}.npy'))


def feat_names_for(version, n_features):
    if version == 'aggstats':
        return MF_NAMES_AGGSTATS
    elif version == 'raw':
        return [f'r_f{j+1}' for j in range(n_features)]
    else:
        return [f'r_f{j+1}' for j in range(n_features)] + ['delta_mean', 'cosine_sim']


def extract_stream_features(stream_name, drift_chunks):
    """Single pass: relevance scores + all 3 meta-feature versions."""
    n_features  = N_FEATURES[stream_name]
    stream_path = os.path.join(REAL_STREAM_DIR, f'{stream_name}.npy')
    stream = sl.streams.NPYParser(stream_path,
                                  chunk_size=CHUNK_SIZE, n_chunks=100000)
    abfs = ABFS_match(n_features=n_features, categorical_features=[],
                      accuracy_window_size=CHUNK_SIZE,
                      class_window_size=CHUNK_SIZE)

    scores_over_time = []
    mf = {'aggstats': [], 'raw': [], 'raw_temporal': []}
    concept_labels   = []
    wt_prev          = None

    for chunk_idx in range(100000):
        concept = int(np.sum(drift_chunks <= chunk_idx))
        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break
        if len(np.unique(y_chunk)) < 2:
            continue
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()
        t_since     = abfs.time_since_drift

        scores_over_time.append(wt)
        mf['aggstats'].append(extract_metafeatures(wt, wt_prev, drift_count, t_since))
        mf['raw'].append(extract_metafeatures_raw(wt))
        mf['raw_temporal'].append(extract_metafeatures_raw_temporal(wt, wt_prev))
        concept_labels.append(concept)
        wt_prev = wt

    def clean(arr):
        a = np.array(arr, dtype=float)
        a[np.isnan(a)] = 0; a[np.isinf(a)] = 0
        return a

    X_by_version = {v: clean(mf[v]) for v in ABFS_VERSIONS}
    return np.array(scores_over_time), X_by_version, np.array(concept_labels)


# ============================================================
#  0. SANITY CHECK
# ============================================================
if RUN_SANITY:
    print("\n" + "="*60)
    print("0. SANITY CHECK PLOTS")
    print("="*60)

    for stream_name in REAL_STREAMS:
        n_features      = N_FEATURES[stream_name]
        n_concepts      = N_CONCEPTS[stream_name]
        random_baseline = 1.0 / n_concepts
        drift_chunks    = load_gt(stream_name)
        boundaries      = list(drift_chunks)
        print(f"\n  {stream_name}")

        scores_over_time, X_by_version, y = \
            extract_stream_features(stream_name, drift_chunks)
        unique_concepts = np.unique(y)

        # relevance scores - one per stream
        fname = os.path.join(FIGURES_DIR, f'relevance_scores_{stream_name}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for j in range(n_features):
                ax.plot(scores_over_time[:, j], label=f'f{j+1}',
                        linewidth=0.7, alpha=0.8)
            for b in boundaries:
                ax.axvline(x=b, color='red', linestyle='--',
                           linewidth=0.8, alpha=0.7)
            ax.axvline(x=-1, color='red', linestyle='--',
                       linewidth=0.8, label='concept boundary')
            ax.set_xlabel('Window'); ax.set_ylabel('Relevance score')
            ax.set_title(f'Relevance scores — {stream_name}\n'
                         f'({n_concepts} concepts, '
                         f'random baseline={random_baseline:.3f})')
            ax.legend(ncol=6, fontsize=6, loc='upper right')
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        # meta-features and PCA - one per version
        for version in ABFS_VERSIONS:
            X     = X_by_version[version]
            names = feat_names_for(version, n_features)
            n_f   = len(names)

            fname = os.path.join(FIGURES_DIR,
                                 f'metafeatures_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                n_cols = 5
                n_rows = (n_f + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(4*n_cols, 3*n_rows))
                axes_flat = axes.flatten()
                for k in range(n_f):
                    axes_flat[k].plot(X[:, k], color='steelblue', linewidth=0.8)
                    for b in boundaries:
                        axes_flat[k].axvline(x=b, color='red', linestyle='--',
                                             linewidth=0.8, alpha=0.7)
                    axes_flat[k].set_title(names[k], fontsize=8)
                    axes_flat[k].set_xlabel('Window', fontsize=7)
                for k in range(n_f, len(axes_flat)):
                    axes_flat[k].set_visible(False)
                fig.suptitle(f'Meta-features ({ABFS_LABELS[version]}) — '
                             f'{stream_name}', fontsize=10)
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")
            else:
                print(f"  Exists: {fname}")

            fname = os.path.join(FIGURES_DIR,
                                 f'pca_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                colors = {c: PALETTE[i % len(PALETTE)]
                          for i, c in enumerate(unique_concepts)}
                pca = PCA(n_components=2)
                projected = pca.fit_transform(X)
                fig, ax = plt.subplots(figsize=(8, 5))
                for c in unique_concepts:
                    mask = y == c
                    ax.scatter(projected[mask, 0], projected[mask, 1],
                               color=colors[c], label=f'concept {c}',
                               alpha=0.6, edgecolors='none', s=20)
                ax.set_xlabel(
                    f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(
                    f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title(f'PCA — {ABFS_LABELS[version]}\n{stream_name}')
                ax.legend(ncol=4, fontsize=7)
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")
            else:
                print(f"  Exists: {fname}")


# ============================================================
#  1. PERFORMANCE TRAJECTORIES
# ============================================================
if RUN_PERFORMANCE:
    print("\n" + "="*60)
    print("1. PERFORMANCE TRAJECTORIES")
    print("="*60)

    for stream_name in REAL_STREAMS:
        n_concepts      = N_CONCEPTS[stream_name]
        random_baseline = 1.0 / n_concepts
        drift_chunks    = load_gt(stream_name)
        boundaries      = list(drift_chunks)
        print(f"\n  {stream_name}")

        # ABFS: all 3 stacked
        fname = os.path.join(FIGURES_DIR, f'trajectory_abfs_{stream_name}.png')
        if not os.path.exists(fname):
            fig, axes = plt.subplots(len(ABFS_VERSIONS), 1,
                                     figsize=(14, 4*len(ABFS_VERSIONS)),
                                     sharex=True)
            for ax, version in zip(axes, ABFS_VERSIONS):
                data = load(f'preq_abfs_{version}_ba', stream_name)
                if data is None:
                    ax.set_title(f'{ABFS_LABELS[version]} — no data')
                    continue
                x_axis = np.arange(data.shape[0])
                for clf_id, name in enumerate(CLF_NAMES):
                    color = CLF_COLORS.get(name, f'C{clf_id}')
                    ax.plot(x_axis, data[:, clf_id], label=name,
                            color=color, linewidth=1.5)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--',
                               linewidth=0.8, alpha=0.7)
                ax.axhline(y=random_baseline, color='red', linestyle='--',
                           linewidth=1.0, label='random baseline')
                ax.set_ylabel('Cumulative BA', fontsize=10)
                ax.set_title(ABFS_LABELS[version], fontsize=11)
                ax.legend(fontsize=9, ncol=4)
                ax.set_ylim(0, 1)
            axes[-1].set_xlabel('Window', fontsize=10)
            fig.suptitle(f'ABFS trajectories — {stream_name}\n'
                         f'({n_concepts} concepts | '
                         f'random baseline={random_baseline:.3f})',
                         fontsize=12)
            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        # Komorniczak: 3x3 grid
        fname = os.path.join(FIGURES_DIR, f'trajectory_komor_{stream_name}.png')
        if not os.path.exists(fname):
            n_cols = 3
            n_rows = (len(MEASURES) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(6*n_cols, 3.5*n_rows),
                                     sharex=True, sharey=True)
            axes_flat = axes.flatten()
            for ax_id, measure in enumerate(MEASURES):
                ax   = axes_flat[ax_id]
                data = load(f'preq_komor_{measure}_ba', stream_name)
                if data is None:
                    ax.set_title(f'{measure} — no data'); continue
                x_axis = np.arange(data.shape[0])
                for clf_id, name in enumerate(CLF_NAMES):
                    color = CLF_COLORS.get(name, f'C{clf_id}')
                    ax.plot(x_axis, data[:, clf_id], label=name,
                            color=color, linewidth=1.2)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--',
                               linewidth=0.6, alpha=0.6)
                ax.axhline(y=random_baseline, color='red',
                           linestyle='--', linewidth=0.8)
                ax.set_title(measure, fontsize=10)
                ax.set_ylim(0, 1)
                if ax_id == 0:
                    ax.legend(fontsize=7, ncol=2)
            for ax_id in range(len(MEASURES), len(axes_flat)):
                axes_flat[ax_id].set_visible(False)
            fig.suptitle(f'Komorniczak trajectories — {stream_name}',
                         fontsize=12)
            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")


# ============================================================
#  2. SHAP — all 4 classifiers, all 3 ABFS versions
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("2. SHAP ANALYSIS")
    print("="*60)

    for stream_name in REAL_STREAMS:
        n_features = N_FEATURES[stream_name]
        print(f"\n  {stream_name}")

        y = load('abfs_y', stream_name)
        if y is None:
            print(f"  y labels not found - skipping."); continue

        all_done = all(
            os.path.exists(os.path.join(
                FIGURES_DIR, f'shap_all_clfs_{v}_{stream_name}.png'))
            for v in ABFS_VERSIONS
        )
        if all_done:
            print(f"  All SHAP figures exist - skipping."); continue

        drift_chunks = load_gt(stream_name)
        _, X_by_version, _ = extract_stream_features(stream_name, drift_chunks)

        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR,
                                 f'shap_all_clfs_{version}_{stream_name}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue

            X          = X_by_version[version]
            feat_names = feat_names_for(version, n_features)
            print(f"  SHAP [{version}]: X={X.shape}, y={y.shape}")

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes_flat  = axes.flatten()

            for clf_idx, (clf_name, clf_proto) in enumerate(SHAP_CLFS):
                ax  = axes_flat[clf_idx]
                clf = skclone(clf_proto)
                clf.fit(X, y)

                explainer   = shap.KernelExplainer(
                    clf.predict_proba, shap.sample(X, 50))
                shap_values = explainer.shap_values(
                    shap.sample(X, 100), nsamples=50)

                shap_array    = np.array(shap_values)
                mean_abs_shap = (np.mean(np.abs(shap_array), axis=(0, 2))
                                 if shap_array.ndim == 3
                                 else np.mean(np.abs(shap_array), axis=0))

                sorted_idx = np.argsort(mean_abs_shap)[::-1]
                ax.bar(range(len(feat_names)), mean_abs_shap[sorted_idx],
                       color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(feat_names)))
                ax.set_xticklabels([feat_names[i] for i in sorted_idx],
                                   rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('Mean |SHAP|', fontsize=9)
                ax.set_title(clf_name, fontsize=11)

            fig.suptitle(f'SHAP — {ABFS_LABELS[version]}\n{stream_name}',
                         fontsize=12)
            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  3. METRICS HEATMAPS (F1, KAPPA)
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60)
    print("3. METRICS HEATMAPS")
    print("="*60)

    for stream_name in REAL_STREAMS:
        n_concepts = N_CONCEPTS[stream_name]
        print(f"\n  {stream_name}")

        for metric in ['f1', 'kappa']:
            fname = os.path.join(FIGURES_DIR,
                                 f'heatmap_{metric}_{stream_name}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue

            komor_matrix = np.full((len(MEASURES), N_CLFS), np.nan)
            for m_id, measure in enumerate(MEASURES):
                data = load(f'preq_komor_{measure}_{metric}',
                            stream_name, optional=True)
                if data is not None:
                    komor_matrix[m_id, :] = data[-1, :]

            abfs_matrix = np.full((len(ABFS_VERSIONS), N_CLFS), np.nan)
            for v_id, version in enumerate(ABFS_VERSIONS):
                data = load(f'preq_abfs_{version}_{metric}',
                            stream_name, optional=True)
                if data is not None:
                    abfs_matrix[v_id, :] = data[-1, :]

            fig, axes = plt.subplots(
                1, 2, figsize=(22, max(5, len(MEASURES) * 0.75)),
                gridspec_kw={'width_ratios': [3, 1.5]})
            metric_label = (metric.upper() if metric == 'f1'
                            else "Cohen's Kappa")

            ax = axes[0]
            ax.imshow(komor_matrix, vmin=0.0, vmax=1.0,
                      cmap='Blues', aspect='auto')
            for i, measure in enumerate(MEASURES):
                for j in range(N_CLFS):
                    val = komor_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                fontsize=10,
                                color='white' if val > 0.6 else 'black')
            ax.set_xticks(range(N_CLFS))
            ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(MEASURES)))
            ax.set_yticklabels(MEASURES, fontsize=10)
            ax.set_title(f'Komorniczak — {metric_label}', fontsize=12)

            ax = axes[1]
            im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0,
                           cmap='Blues', aspect='auto')
            for i, version in enumerate(ABFS_VERSIONS):
                for j in range(N_CLFS):
                    val = abfs_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                fontsize=10,
                                color='white' if val > 0.6 else 'black')
            ax.set_xticks(range(N_CLFS))
            ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(ABFS_VERSIONS)))
            ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS],
                               fontsize=10)
            ax.set_title(f'ABFS — {metric_label}', fontsize=12)

            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle(
                f'{metric_label} — {stream_name}\n'
                f'Prequential | final window | '
                f'random baseline = {1/n_concepts:.3f}',
                fontsize=12)
            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")

print("\nAnalysis 3 complete.")