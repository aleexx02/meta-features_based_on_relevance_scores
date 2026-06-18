# analysis_4.py
# ============================================================
# Analysis of Experiment 4 results (semi-synthetic injected-drift streams).
#
# Streams analysed here: electricity and covtype, both with
# artificially injected drift (see generate_semi_synthetic_streams.py
# and evaluate_concept_classification_4.py for the full rationale —
# neither dataset has a published natural drift ground truth).
#
# I load the pre-computed prequential results from
# results/experiment_4/ and generate the following figures
# per stream:
#
# --sanity
#   relevance_scores_{stream}.png
#     ABFS relevance scores over time for all features,
#     with concept drift boundaries marked in red.
#
#   metafeatures_{version}_{stream}.png  (one per ABFS version)
#     Each meta-feature dimension plotted over windows,
#     with drift boundaries marked.
#
#   pca_{version}_{stream}.png  (one per ABFS version)
#     2D PCA projection of the meta-feature space,
#     coloured by concept label.
#
# --performance
#   trajectory_abfs_{stream}.png
#     Cumulative balanced accuracy over windows for all
#     3 ABFS versions stacked vertically, with drift
#     boundaries and random baseline marked.
#
#   trajectory_komor_{stream}.png
#     Same but for all 9 Komorniczak measure groups
#     arranged in a 3×3 grid.
#
# --shap
#   shap_all_clfs_{version}_{stream}.png  (one per ABFS version)
#     Mean |SHAP| feature importance for all 4 classifiers
#     arranged in a 2×2 subplot.
#
# --metrics
#   heatmap_f1_{stream}.png
#   heatmap_kappa_{stream}.png
#     Side-by-side heatmaps of final F1 / Kappa values
#     for Komorniczak (left) and ABFS (right).
#
# Usage:
#   python experiments/experiment_4/analysis_4.py --sanity --performance --shap --metrics
#
# Inputs (from results/experiment_4/):
#   concept_labels_{stream}.npy -> concept label per window
#   preq_abfs_{version}_ba_{stream}.npy -> cumulative BA trajectory
#   preq_komor_{measure}_ba_{stream}.npy -> same for each Komorniczak group
#   (same pattern for f1, kappa)
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
parser.add_argument('--stream_analysis', action='store_true')

args = parser.parse_args()
 
RUN_SANITY      = args.sanity
RUN_PERFORMANCE = args.performance
RUN_SHAP        = args.shap
RUN_METRICS     = args.metrics
RUN_STREAM_ANALYSIS = args.stream_analysis
 
print(f"\nRunning analysis for Experiment 4 (semi-synthetic streams):")
print(f"Sanity      : {RUN_SANITY}")
print(f"Performance : {RUN_PERFORMANCE}")
print(f"SHAP        : {RUN_SHAP}")
print(f"Metrics     : {RUN_METRICS}")
print(f"Stream analysis : {RUN_STREAM_ANALYSIS}")
 
 
# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
 
SEMI_SYN_STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'semi_synthetic', 'streams')
SEMI_SYN_GT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'semi_synthetic', 'streams_gt')
SEMI_SYN_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'data', 'semi_synthetic', 'analysis')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
FIGURES_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_4', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)
 
 
# ============================================================
#  CONFIGURATION
# ============================================================
# chunk_size = 200, same as evaluate_concept_classification_4.py
CHUNK_SIZE = 200
 
SEMI_SYN_STREAMS = [
    'electricity',
    'covtype',
]
 
N_FEATURES = {
    'electricity': 8,
    'covtype':     54,
}
 
N_CONCEPTS = {
    s: len(np.load(os.path.join(SEMI_SYN_GT_DIR, f'{s}.npy'))) + 1
    for s in SEMI_SYN_STREAMS
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
 
# one colour per concept — enough for up to 32 concepts
PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#469990', '#9a6324',
    '#000075', '#800000', '#a9a9a9', '#000000', '#556b2f',
    '#d2691e', '#5e5151', '#08332b', '#2ecc71', '#ff69b4',
    '#00ced1', '#ff8c00', '#c9a0dc', '#7b3f91', '#e6ac00',
]
 
# sklearn classifiers used for SHAP (proxies for the River online classifiers)
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
    """Load a result .npy file from results/experiment_4/."""
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)
 
 
def load_gt(stream_name):
    """Load the drift chunk indices for this stream."""
    return np.load(os.path.join(SEMI_SYN_GT_DIR, f'{stream_name}.npy'))
 
 
def feat_names_for(version, n_features):
    """Return human-readable feature names for a given ABFS version."""
    if version == 'aggstats':
        return MF_NAMES_AGGSTATS
    elif version == 'raw':
        return [f'r_f{j+1}' for j in range(n_features)]
    else:
        return [f'r_f{j+1}' for j in range(n_features)] + ['delta_mean', 'cosine_sim']
 
 
def re_extract_stream(stream_name, drift_chunks):
    """
    Re-run ABFS on the stream to get relevance scores and all 3 meta-feature
    versions for sanity and SHAP plots. This avoids having to store large
    intermediate arrays to disk — only the prequential results are saved by
    evaluate_concept_classification_4.py.
    """
    n_features  = N_FEATURES[stream_name]
    stream_path = os.path.join(SEMI_SYN_STREAM_DIR, f'{stream_name}.npy')
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
        if len(X_chunk) == 0:
            break
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
#  SANITY CHECK
#  relevance_scores_{stream}.png
#  metafeatures_{version}_{stream}.png  x3
#  pca_{version}_{stream}.png           x3
# ============================================================
if RUN_SANITY:
    print("\n" + "="*60)
    print("SANITY CHECK PLOTS")
    print("="*60)
 
    for stream_name in SEMI_SYN_STREAMS:
        n_features      = N_FEATURES[stream_name]
        n_concepts      = N_CONCEPTS[stream_name]
        random_baseline = 1.0 / n_concepts
        drift_chunks    = load_gt(stream_name)
        boundaries      = list(drift_chunks)
        print(f"\n  {stream_name}")
 
        scores_over_time, X_by_version, y = re_extract_stream(
            stream_name, drift_chunks)
        unique_concepts = np.unique(y)
 
        # ---- relevance scores over time ----
        fname = os.path.join(FIGURES_DIR, f'relevance_scores_{stream_name}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for j in range(n_features):
                ax.plot(scores_over_time[:, j], linewidth=0.7, alpha=0.8,
                        label=f'f{j+1}')
            for b in boundaries:
                ax.axvline(x=b, color='red', linestyle='--',
                           linewidth=0.8, alpha=0.7)
            ax.axvline(x=-1, color='red', linestyle='--',
                       linewidth=0.8, label='concept boundary')
            ax.set_xlabel('Window')
            ax.set_ylabel('Relevance score')
            ax.set_title(f'Relevance scores — {stream_name}\n'
                         f'({n_concepts} concepts, '
                         f'random baseline={random_baseline:.3f})')
            ax.legend(ncol=6, fontsize=6, loc='upper right')
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")
 
        # ---- meta-features over time + PCA — one per version ----
        for version in ABFS_VERSIONS:
            X     = X_by_version[version]
            names = feat_names_for(version, n_features)
            n_f   = len(names)
 
            # meta-feature trajectories
            fname = os.path.join(FIGURES_DIR,
                                 f'metafeatures_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                n_cols = 5
                n_rows = (n_f + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(4*n_cols, 3*n_rows))
                axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
                for k in range(n_f):
                    axes_flat[k].plot(X[:, k], color='steelblue', linewidth=0.8)
                    for b in boundaries:
                        axes_flat[k].axvline(x=b, color='red',
                                             linestyle='--', linewidth=0.8, alpha=0.7)
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
 
            # PCA projection coloured by concept
            fname = os.path.join(FIGURES_DIR,
                                 f'pca_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                colors = {c: PALETTE[i % len(PALETTE)]
                          for i, c in enumerate(unique_concepts)}
                pca       = PCA(n_components=2)
                projected = pca.fit_transform(X)
                fig, ax   = plt.subplots(figsize=(8, 5))
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
#  PERFORMANCE TRAJECTORIES
#  trajectory_abfs_{stream}.png
#  trajectory_komor_{stream}.png
# ============================================================
if RUN_PERFORMANCE:
    print("\n" + "="*60)
    print("PERFORMANCE TRAJECTORIES")
    print("="*60)
 
    for stream_name in SEMI_SYN_STREAMS:
        n_concepts      = N_CONCEPTS[stream_name]
        random_baseline = 1.0 / n_concepts
        drift_chunks    = load_gt(stream_name)
        boundaries      = list(drift_chunks)
        print(f"\n  {stream_name}")
 
        # all 3 ABFS versions stacked vertically
        fname = os.path.join(FIGURES_DIR, f'trajectory_abfs_{stream_name}.png')
        if not os.path.exists(fname):
            fig, axes = plt.subplots(len(ABFS_VERSIONS), 1,
                                     figsize=(14, 4*len(ABFS_VERSIONS)),
                                     sharex=True)
            for ax, version in zip(axes, ABFS_VERSIONS):
                data = load(f'preq_abfs_{version}_ba', stream_name)
                if data is None:
                    ax.set_title(f'{ABFS_LABELS[version]} — no data'); continue
                x_axis = np.arange(data.shape[0])
                for clf_id, name in enumerate(CLF_NAMES):
                    ax.plot(x_axis, data[:, clf_id], label=name,
                            color=CLF_COLORS.get(name, f'C{clf_id}'),
                            linewidth=1.5)
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
 
        # all 9 Komorniczak groups in a 3×3 grid
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
                    ax.plot(x_axis, data[:, clf_id], label=name,
                            color=CLF_COLORS.get(name, f'C{clf_id}'),
                            linewidth=1.2)
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
#  SHAP
#  shap_all_clfs_{version}_{stream}.png  x3
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
 
    for stream_name in SEMI_SYN_STREAMS:
        n_features = N_FEATURES[stream_name]
        print(f"\n  {stream_name}")
 
        # concept_labels_{stream}.npy contains the ground truth concept label
        # per window, derived from the known drift boundaries (not from ABFS).
        # It was saved by evaluate_concept_classification_4.py.
        y = load('concept_labels', stream_name)
        if y is None:
            print(f"  concept labels not found — skipping."); continue
 
        all_done = all(
            os.path.exists(os.path.join(
                FIGURES_DIR, f'shap_all_clfs_{v}_{stream_name}.png'))
            for v in ABFS_VERSIONS)
        if all_done:
            print(f"  All SHAP figures exist — skipping."); continue
 
        drift_chunks = load_gt(stream_name)
        _, X_by_version, _ = re_extract_stream(stream_name, drift_chunks)
 
        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR,
                                 f'shap_all_clfs_{version}_{stream_name}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue
 
            X          = X_by_version[version]
            feat_names = feat_names_for(version, n_features)
            print(f"  SHAP [{version}]: X={X.shape}")
 
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
#  METRICS HEATMAPS
#  heatmap_f1_{stream}.png
#  heatmap_kappa_{stream}.png
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60)
    print("METRICS HEATMAPS")
    print("="*60)
 
    for stream_name in SEMI_SYN_STREAMS:
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
 
            metric_label = metric.upper() if metric == 'f1' else "Cohen's Kappa"
            fig, axes = plt.subplots(
                1, 2, figsize=(22, max(5, len(MEASURES) * 0.75)),
                gridspec_kw={'width_ratios': [3, 1.5]})
 
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
            ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES, fontsize=10)
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
            ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(ABFS_VERSIONS)))
            ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
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

# ============================================================
# STREAM ANALYSIS PLOTS
# ============================================================
if RUN_STREAM_ANALYSIS:
    print("\n" + "="*60)
    print("STREAM ANALYSIS PLOTS")
    print("="*60)

    for stream_name in SEMI_SYN_STREAMS:
        print(f"\n  {stream_name}")

        drift_chunks = load_gt(stream_name)
        boundaries   = list(drift_chunks)

        # ---- load analysis data ----
        try:
            drift_intensity = np.load(os.path.join(
                SEMI_SYN_ANALYSIS_DIR, f'{stream_name}_drift_intensity.npy'))

            class_dist = np.load(os.path.join(
                SEMI_SYN_ANALYSIS_DIR, f'{stream_name}_class_distribution.npy'))

            entropy_vals = np.load(os.path.join(
                SEMI_SYN_ANALYSIS_DIR, f'{stream_name}_label_entropy.npy'))
            # ---- ABFS RELEVANCE DYNAMICS ----
            scores_over_time, _, _ = re_extract_stream(stream_name, drift_chunks)

            delta_relevance = np.linalg.norm(
                scores_over_time[1:] - scores_over_time[:-1],
                axis=1
            )
            delta_relevance = np.concatenate([[0], delta_relevance])

            # normalize
            delta_relevance = delta_relevance / (np.max(delta_relevance) + 1e-10)

        except Exception as e:
            print(f"  Missing analysis files for {stream_name} — skipping.")
            continue

        n_chunks = len(drift_intensity)

        # ========================================================
        # Drift + entropy + ABFS dynamics
        # ========================================================
        fname = os.path.join(FIGURES_DIR,
                            f'stream_drift_entropy_{stream_name}.png')
        if not os.path.exists(fname):

            # normalize drift
            drift_intensity = drift_intensity / (np.max(drift_intensity) + 1e-10)

            fig, ax1 = plt.subplots(figsize=(14, 4))

            # ---- data drift ----
            ax1.plot(drift_intensity, color='steelblue',
                    label='Drift intensity', linewidth=1.5)

            # ---- ABFS dynamics ----
            ax1.plot(delta_relevance, color='purple',
                    label='ABFS relevance change', linewidth=1.2, alpha=0.7)

            ax1.set_ylabel('Normalized value')
            ax1.set_xlabel('Window')

            # ---- entropy ----
            ax2 = ax1.twinx()
            ax2.plot(entropy_vals, color='darkorange',
                    label='Label entropy', alpha=0.7)
            ax2.set_ylabel('Entropy', color='darkorange')

            # ---- boundaries ----
            for b in boundaries:
                ax1.axvline(x=b, color='red', linestyle='--',
                            linewidth=0.8, alpha=0.7)

            # ---- legend ----
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            ax1.set_title(f'Drift vs ABFS dynamics — {stream_name}')

            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        # ========================================================
        # Class distribution
        # ========================================================
        fname = os.path.join(FIGURES_DIR,
                             f'class_distribution_{stream_name}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))

            for c in range(class_dist.shape[1]):
                ax.plot(class_dist[:, c],
                        label=f'class {c}', linewidth=1.2)

            for b in boundaries:
                ax.axvline(x=b, color='grey', linestyle='--',
                           linewidth=0.7, alpha=0.6)

            ax.set_xlabel('Window')
            ax.set_ylabel('Proportion')
            ax.set_title(f'Class distribution over time — {stream_name}')
            ax.legend(ncol=4, fontsize=8)

            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")
 
print("\nAnalysis 4 complete.")