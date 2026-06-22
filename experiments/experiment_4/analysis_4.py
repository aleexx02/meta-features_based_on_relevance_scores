# analysis_4.py
# ============================================================
# Analysis of Experiment 4 results.
#
# Streams analysed here: RECURRING-concept streams built from SEA and
# STAGGER (river synthetic; each concept cycles twice so it reappears
# with the same generative id). The stream set + per-stream metadata
# come from streams.generate_synthetic_streams.exp4_specs(), so this
# list stays in sync automatically with the evaluate script and the
# generator. 4 streams: recurring_{sea,stagger}_fixed_{sudden,gradual}.
#
# The concept label per chunk is the GENERATIVE concept id
# (concept_per_chunk on disk), so a recurring concept carries the SAME
# label on each appearance -- that repetition is the entire point of
# Experiment 4 (it lets the PCA / relevance-score plots show whether a
# recurring concept lands back in the same region of meta-feature
# space, i.e. whether ABFS RECOGNISES it rather than just registering
# that something changed).
#
# Two differences from analysis_5.py (identical to analysis_3.py):
#  1. Single realization for per-stream plots: the on-disk MASTER_SEED
#     stream (= replication 0), loaded from data/synthetic/streams[/_gt].
#     Performance / metrics / gap figures average over the rep axis.
#  2. Stream-analysis diagnostics computed INLINE (generate_synthetic_
#     streams.py saves only the stream + concept_per_chunk, not
#     pre-computed diagnostics).
#
# Figures (results/experiment_4/figures/analysis/):
# --sanity           relevance_scores / metafeatures_{version} / pca_{version}
#                    (feature counts are small: 3 for both SEA and STAGGER,
#                     so no feature capping needed)
# --performance      trajectory_abfs / trajectory_komor  (mean over reps)
# --shap             shap_all_clfs_{version}
# --metrics          heatmap_f1 / heatmap_kappa           (mean over reps)
# --stream_analysis  stream_drift_entropy / class_distribution (inline)
# --gap              gap_heatmap_preq_exp4_{stream}       (mean over reps)
#
# Usage:
#   python experiments/experiment_4/analysis_4.py --sanity --performance --shap --metrics --stream_analysis --gap
# ============================================================

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import entropy as scipy_entropy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))

from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures, extract_metafeatures_raw,
    extract_metafeatures_raw_temporal, MF_NAMES_AGGSTATS,
)
from streams.generate_synthetic_streams import exp4_specs, CHUNK_SIZE
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
parser.add_argument('--sanity',          action='store_true')
parser.add_argument('--performance',     action='store_true')
parser.add_argument('--shap',            action='store_true')
parser.add_argument('--metrics',         action='store_true')
parser.add_argument('--gap',             action='store_true')
parser.add_argument('--stream_analysis', action='store_true')
args = parser.parse_args()

RUN_SANITY          = args.sanity
RUN_PERFORMANCE     = args.performance
RUN_SHAP            = args.shap
RUN_METRICS         = args.metrics
RUN_GAP             = args.gap
RUN_STREAM_ANALYSIS = args.stream_analysis

EXP_TAG = 'exp4'
print(f"\nRunning analysis for Experiment 4 (recurring SEA/STAGGER synthetic streams):")
print(f"Sanity={RUN_SANITY} Performance={RUN_PERFORMANCE} SHAP={RUN_SHAP} "
      f"Metrics={RUN_METRICS} Gap={RUN_GAP} StreamAnalysis={RUN_STREAM_ANALYSIS}")


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

SYNTH_STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'streams')
SYNTH_GT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'streams_gt')
RESULTS_DIR      = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
FIGURES_DIR      = os.path.join(PROJECT_ROOT, 'results', 'experiment_4', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================
SPECS      = exp4_specs()
SPEC_BY_NAME = {s['name']: s for s in SPECS}
STREAMS    = [s['name'] for s in SPECS]

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
CLF_COLORS = {'GNB': '#e6194b', 'KNN': '#3cb44b', 'HT': '#f032e6', 'MLP': '#911eb4'}

PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#469990', '#9a6324',
    '#000075', '#800000', '#a9a9a9', '#000000', '#556b2f',
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
    """Load a result .npy from results/experiment_4/. These are
    (n_reps, n_windows, n_clfs); callers average over axis 0 as needed."""
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)


def load_disk_stream(stream_name):
    """Load the on-disk MASTER_SEED realization (= rep 0) and its
    concept_per_chunk ground truth."""
    data = np.load(os.path.join(SYNTH_STREAM_DIR, f'{stream_name}.npy'))
    cpc  = np.load(os.path.join(SYNTH_GT_DIR,     f'{stream_name}.npy'))
    return data, cpc


def boundaries_from_cpc(cpc):
    """Drift chunk indices = where the generative concept id changes."""
    return [int(i) for i in np.where(np.diff(cpc) != 0)[0] + 1]


def feat_names_for(version, n_features):
    if version == 'aggstats':
        return MF_NAMES_AGGSTATS
    elif version == 'raw':
        return [f'r_f{j+1}' for j in range(n_features)]
    else:
        return [f'r_f{j+1}' for j in range(n_features)] + ['delta_mean', 'cosine_sim']


def re_extract_stream(stream_name):
    """Re-run ABFS on the on-disk realization to get relevance scores +
    all 3 meta-feature versions for sanity / SHAP / stream-analysis.
    Returns scores_over_time, {version: X}, concept_labels."""
    spec       = SPEC_BY_NAME[stream_name]
    n_features = spec['n_features']
    data, cpc  = load_disk_stream(stream_name)
    X_full = data[:, :-1]
    y_full = data[:, -1]
    n_chunks = len(cpc)

    abfs = ABFS_match(n_features=n_features, categorical_features=[],
                      accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
    scores_over_time = []
    mf = {'aggstats': [], 'raw': [], 'raw_temporal': []}
    concept_labels = []
    wt_prev = None

    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE; e = s + CHUNK_SIZE
        X_chunk, y_chunk = X_full[s:e], y_full[s:e]
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()
        t_since     = abfs.time_since_drift

        scores_over_time.append(wt)
        mf['aggstats'].append(extract_metafeatures(wt, wt_prev, drift_count, t_since))
        mf['raw'].append(extract_metafeatures_raw(wt))
        mf['raw_temporal'].append(extract_metafeatures_raw_temporal(wt, wt_prev))
        concept_labels.append(int(cpc[ci]))
        wt_prev = wt

    def clean(arr):
        a = np.array(arr, dtype=float)
        a[np.isnan(a)] = 0; a[np.isinf(a)] = 0
        return a

    X_by_version = {v: clean(mf[v]) for v in ABFS_VERSIONS}
    return np.array(scores_over_time), X_by_version, np.array(concept_labels)


def compute_stream_diagnostics(stream_name):
    """Inline per-chunk diagnostics from the on-disk realization
    (generate_synthetic_streams.py doesn't pre-save these, unlike
    generate_real_streams.py): drift intensity (feature-mean shift),
    class distribution of the real target, label entropy."""
    data, cpc = load_disk_stream(stream_name)
    X_full = data[:, :-1]
    y_full = data[:, -1].astype(int)
    n_chunks  = len(cpc)
    n_classes = int(np.max(y_full)) + 1

    drift_intensity, class_dist, label_entropy = [], [], []
    prev_mean = None
    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE; e = s + CHUNK_SIZE
        Xc = X_full[s:e]; yc = y_full[s:e]
        mean = np.mean(Xc, axis=0)
        drift_intensity.append(0.0 if prev_mean is None
                               else float(np.linalg.norm(mean - prev_mean)))
        prev_mean = mean
        counts = np.bincount(yc, minlength=n_classes)
        probs  = counts / max(1, counts.sum())
        class_dist.append(probs)
        label_entropy.append(float(scipy_entropy(probs + 1e-10)))

    return (np.array(drift_intensity), np.array(class_dist),
            np.array(label_entropy))


# ============================================================
#  SANITY
# ============================================================
if RUN_SANITY:
    print("\n" + "="*60); print("SANITY CHECK PLOTS"); print("="*60)
    for stream_name in STREAMS:
        spec       = SPEC_BY_NAME[stream_name]
        n_features = spec['n_features']
        _, cpc     = load_disk_stream(stream_name)
        n_concepts = len(np.unique(cpc))
        random_baseline = 1.0 / n_concepts
        boundaries = boundaries_from_cpc(cpc)
        print(f"\n  {stream_name}")

        scores_over_time, X_by_version, y = re_extract_stream(stream_name)
        unique_concepts = np.unique(y)

        # relevance scores over time
        fname = os.path.join(FIGURES_DIR, f'relevance_scores_{stream_name}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for j in range(n_features):
                ax.plot(scores_over_time[:, j], linewidth=0.7, alpha=0.8, label=f'f{j+1}')
            for b in boundaries:
                ax.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axvline(x=-1, color='red', linestyle='--', linewidth=0.8, label='concept boundary')
            ax.set_xlabel('Window'); ax.set_ylabel('Relevance score')
            ax.set_title(f'Relevance scores -- {stream_name}\n'
                         f'({n_concepts} concepts, random baseline={random_baseline:.3f})')
            ax.legend(ncol=6, fontsize=6, loc='upper right')
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        for version in ABFS_VERSIONS:
            X     = X_by_version[version]
            names = feat_names_for(version, n_features)
            n_f   = len(names)

            fname = os.path.join(FIGURES_DIR, f'metafeatures_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                n_cols = 5
                n_rows = (n_f + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
                axes_flat = np.array(axes).flatten()
                for k in range(n_f):
                    axes_flat[k].plot(X[:, k], color='steelblue', linewidth=0.8)
                    for b in boundaries:
                        axes_flat[k].axvline(x=b, color='red', linestyle='--',
                                             linewidth=0.8, alpha=0.7)
                    axes_flat[k].set_title(names[k], fontsize=8)
                    axes_flat[k].set_xlabel('Window', fontsize=7)
                for k in range(n_f, len(axes_flat)):
                    axes_flat[k].set_visible(False)
                fig.suptitle(f'Meta-features ({ABFS_LABELS[version]}) -- {stream_name}', fontsize=10)
                fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")
            else:
                print(f"  Exists: {fname}")

            fname = os.path.join(FIGURES_DIR, f'pca_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique_concepts)}
                pca = PCA(n_components=2); projected = pca.fit_transform(X)
                fig, ax = plt.subplots(figsize=(8, 5))
                for c in unique_concepts:
                    mask = y == c
                    ax.scatter(projected[mask, 0], projected[mask, 1],
                               color=colors[c], label=f'concept {c}', alpha=0.6,
                               edgecolors='none', s=20)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title(f'PCA -- {ABFS_LABELS[version]}\n{stream_name}')
                ax.legend(ncol=4, fontsize=7)
                fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")
            else:
                print(f"  Exists: {fname}")


# ============================================================
#  PERFORMANCE TRAJECTORIES  (mean over reps)
# ============================================================
if RUN_PERFORMANCE:
    print("\n" + "="*60); print("PERFORMANCE TRAJECTORIES"); print("="*60)
    for stream_name in STREAMS:
        _, cpc     = load_disk_stream(stream_name)
        n_concepts = len(np.unique(cpc))
        random_baseline = 1.0 / n_concepts
        boundaries = boundaries_from_cpc(cpc)
        print(f"\n  {stream_name}")

        fname = os.path.join(FIGURES_DIR, f'trajectory_abfs_{stream_name}.png')
        if not os.path.exists(fname):
            fig, axes = plt.subplots(len(ABFS_VERSIONS), 1,
                                     figsize=(14, 4*len(ABFS_VERSIONS)), sharex=True)
            for ax, version in zip(axes, ABFS_VERSIONS):
                data = load(f'preq_abfs_{version}_ba', stream_name)
                if data is None:
                    ax.set_title(f'{ABFS_LABELS[version]} -- no data'); continue
                mean_traj = np.mean(data, axis=0)   # (n_windows, n_clfs)
                x_axis = np.arange(mean_traj.shape[0])
                for clf_id, name in enumerate(CLF_NAMES):
                    ax.plot(x_axis, mean_traj[:, clf_id], label=name,
                            color=CLF_COLORS.get(name, f'C{clf_id}'), linewidth=1.5)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.axhline(y=random_baseline, color='red', linestyle='--',
                           linewidth=1.0, label='random baseline')
                ax.set_ylabel('Cumulative BA'); ax.set_title(ABFS_LABELS[version])
                ax.legend(fontsize=9, ncol=4); ax.set_ylim(0, 1)
            axes[-1].set_xlabel('Window')
            fig.suptitle(f'ABFS trajectories (mean over reps) -- {stream_name}\n'
                         f'({n_concepts} concepts | random baseline={random_baseline:.3f})',
                         fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        fname = os.path.join(FIGURES_DIR, f'trajectory_komor_{stream_name}.png')
        if not os.path.exists(fname):
            n_cols = 3; n_rows = (len(MEASURES) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3.5*n_rows),
                                     sharex=True, sharey=True)
            axes_flat = axes.flatten()
            for ax_id, measure in enumerate(MEASURES):
                ax = axes_flat[ax_id]
                data = load(f'preq_komor_{measure}_ba', stream_name)
                if data is None:
                    ax.set_title(f'{measure} -- no data'); continue
                mean_traj = np.mean(data, axis=0)
                x_axis = np.arange(mean_traj.shape[0])
                for clf_id, name in enumerate(CLF_NAMES):
                    ax.plot(x_axis, mean_traj[:, clf_id], label=name,
                            color=CLF_COLORS.get(name, f'C{clf_id}'), linewidth=1.2)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.6, alpha=0.6)
                ax.axhline(y=random_baseline, color='red', linestyle='--', linewidth=0.8)
                ax.set_title(measure, fontsize=10); ax.set_ylim(0, 1)
                if ax_id == 0:
                    ax.legend(fontsize=7, ncol=2)
            for ax_id in range(len(MEASURES), len(axes_flat)):
                axes_flat[ax_id].set_visible(False)
            fig.suptitle(f'Komorniczak trajectories (mean over reps) -- {stream_name}', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")


# ============================================================
#  SHAP
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60); print("SHAP ANALYSIS"); print("="*60)
    for stream_name in STREAMS:
        spec       = SPEC_BY_NAME[stream_name]
        n_features = spec['n_features']
        print(f"\n  {stream_name}")

        y = load('concept_labels', stream_name)
        if y is None:
            print(f"  concept labels not found -- skipping."); continue

        all_done = all(os.path.exists(os.path.join(
            FIGURES_DIR, f'shap_all_clfs_{v}_{stream_name}.png')) for v in ABFS_VERSIONS)
        if all_done:
            print(f"  All SHAP figures exist -- skipping."); continue

        _, X_by_version, _ = re_extract_stream(stream_name)

        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR, f'shap_all_clfs_{version}_{stream_name}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue
            X          = X_by_version[version]
            feat_names = feat_names_for(version, n_features)
            print(f"  SHAP [{version}]: X={X.shape}")

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes_flat = axes.flatten()
            for clf_idx, (clf_name, clf_proto) in enumerate(SHAP_CLFS):
                ax  = axes_flat[clf_idx]
                clf = skclone(clf_proto); clf.fit(X, y)
                explainer   = shap.KernelExplainer(clf.predict_proba, shap.sample(X, 50))
                shap_values = explainer.shap_values(shap.sample(X, 100), nsamples=50)
                shap_array  = np.array(shap_values)
                mean_abs    = (np.mean(np.abs(shap_array), axis=(0, 2))
                               if shap_array.ndim == 3
                               else np.mean(np.abs(shap_array), axis=0))
                order = np.argsort(mean_abs)[::-1]
                ax.bar(range(len(feat_names)), mean_abs[order], color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(feat_names)))
                ax.set_xticklabels([feat_names[i] for i in order],
                                   rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('Mean |SHAP|', fontsize=9); ax.set_title(clf_name, fontsize=11)
            fig.suptitle(f'SHAP -- {ABFS_LABELS[version]}\n{stream_name}', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  METRICS HEATMAPS  (mean over reps)
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60); print("METRICS HEATMAPS"); print("="*60)
    for stream_name in STREAMS:
        _, cpc     = load_disk_stream(stream_name)
        n_concepts = len(np.unique(cpc))
        print(f"\n  {stream_name}")
        for metric in ['f1', 'kappa']:
            fname = os.path.join(FIGURES_DIR, f'heatmap_{metric}_{stream_name}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue

            komor_matrix = np.full((len(MEASURES), N_CLFS), np.nan)
            for m_id, measure in enumerate(MEASURES):
                data = load(f'preq_komor_{measure}_{metric}', stream_name, optional=True)
                if data is not None:
                    komor_matrix[m_id, :] = np.mean(data[:, -1, :], axis=0)

            abfs_matrix = np.full((len(ABFS_VERSIONS), N_CLFS), np.nan)
            for v_id, version in enumerate(ABFS_VERSIONS):
                data = load(f'preq_abfs_{version}_{metric}', stream_name, optional=True)
                if data is not None:
                    abfs_matrix[v_id, :] = np.mean(data[:, -1, :], axis=0)

            metric_label = metric.upper() if metric == 'f1' else "Cohen's Kappa"
            fig, axes = plt.subplots(1, 2, figsize=(22, max(5, len(MEASURES) * 0.75)),
                                     gridspec_kw={'width_ratios': [3, 1.5]})
            ax = axes[0]
            ax.imshow(komor_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
            for i in range(len(MEASURES)):
                for j in range(N_CLFS):
                    val = komor_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                fontsize=10, color='white' if val > 0.6 else 'black')
            ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES, fontsize=10)
            ax.set_title(f'Komorniczak -- {metric_label}', fontsize=12)

            ax = axes[1]
            im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
            for i in range(len(ABFS_VERSIONS)):
                for j in range(N_CLFS):
                    val = abfs_matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                fontsize=10, color='white' if val > 0.6 else 'black')
            ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(ABFS_VERSIONS)))
            ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
            ax.set_title(f'ABFS -- {metric_label}', fontsize=12)

            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle(f'{metric_label} (mean over reps) -- {stream_name}\n'
                         f'Prequential | final window | random baseline = {1/n_concepts:.3f}',
                         fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  STREAM ANALYSIS  (diagnostics computed inline)
# ============================================================
if RUN_STREAM_ANALYSIS:
    print("\n" + "="*60); print("STREAM ANALYSIS PLOTS"); print("="*60)
    for stream_name in STREAMS:
        print(f"\n  {stream_name}")
        _, cpc     = load_disk_stream(stream_name)
        boundaries = boundaries_from_cpc(cpc)

        drift_intensity, class_dist, entropy_vals = compute_stream_diagnostics(stream_name)
        scores_over_time, _, _ = re_extract_stream(stream_name)
        delta_relevance = np.linalg.norm(
            scores_over_time[1:] - scores_over_time[:-1], axis=1)
        delta_relevance = np.concatenate([[0], delta_relevance])
        delta_relevance = delta_relevance / (np.max(delta_relevance) + 1e-10)

        fname = os.path.join(FIGURES_DIR, f'stream_drift_entropy_{stream_name}.png')
        if not os.path.exists(fname):
            di = drift_intensity / (np.max(drift_intensity) + 1e-10)
            fig, ax1 = plt.subplots(figsize=(14, 4))
            ax1.plot(di, color='steelblue', label='Drift intensity', linewidth=1.5)
            ax1.plot(delta_relevance, color='purple', label='ABFS relevance change',
                     linewidth=1.2, alpha=0.7)
            ax1.set_ylabel('Normalized value')
            ax2 = ax1.twinx()
            ax2.plot(entropy_vals, color='darkorange', label='Label entropy', alpha=0.7)
            ax2.set_ylabel('Entropy', color='darkorange')
            for b in boundaries:
                ax1.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax1.set_xlabel('Window')
            l1, lab1 = ax1.get_legend_handles_labels()
            l2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lab1 + lab2, loc='upper right')
            ax1.set_title(f'Drift vs ABFS dynamics -- {stream_name}')
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        fname = os.path.join(FIGURES_DIR, f'class_distribution_{stream_name}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for c in range(class_dist.shape[1]):
                ax.plot(class_dist[:, c], label=f'class {c}', linewidth=1.2)
            for b in boundaries:
                ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.set_xlabel('Window'); ax.set_ylabel('Proportion')
            ax.set_title(f'Class distribution over time -- {stream_name}')
            ax.legend(ncol=4, fontsize=8)
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")


# ============================================================
#  GAP HEATMAP  -- ABFS raw v2.0 vs Komorniczak best-of-9, mean over reps
# ============================================================
if RUN_GAP:
    print("\n" + "="*60)
    print("GAP HEATMAP -- ABFS raw v2.0 vs Komorniczak best-of-9"); print("="*60)
    for stream_name in STREAMS:
        fname = os.path.join(FIGURES_DIR, f'gap_heatmap_preq_{EXP_TAG}_{stream_name}.png')
        if os.path.exists(fname):
            print(f"  Exists: {fname}"); continue

        pr_abfs = load('preq_abfs_raw_ba', stream_name, optional=True)
        if pr_abfs is None:
            print(f"  {stream_name}: missing ABFS raw -- skipping."); continue
        abfs_final = np.mean(pr_abfs[:, -1, :], axis=0)   # (n_clfs,)

        komor_best = None
        for measure in MEASURES:
            data = load(f'preq_komor_{measure}_ba', stream_name, optional=True)
            if data is None:
                continue
            final = np.mean(data[:, -1, :], axis=0)
            if komor_best is None or np.max(final) > np.max(komor_best):
                komor_best = final
        if komor_best is None:
            print(f"  {stream_name}: missing Komorniczak -- skipping."); continue

        gap_row = abfs_final - komor_best
        vmax = np.max(np.abs(gap_row)) if np.any(~np.isnan(gap_row)) else 1.0

        fig, ax = plt.subplots(figsize=(max(4, N_CLFS * 1.4), 2.4))
        im = ax.imshow(gap_row.reshape(1, -1), vmin=-vmax, vmax=vmax, cmap='RdBu', aspect='auto')
        for j in range(N_CLFS):
            val = gap_row[j]
            ax.text(j, 0, f'{val:+.3f}', ha='center', va='center', fontsize=11,
                    color='white' if abs(val) > vmax * 0.6 else 'black')
        ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
        ax.set_yticks([0]); ax.set_yticklabels([stream_name], fontsize=10)
        ax.set_xlabel('Classifier', fontsize=11)
        ax.set_title(f'Gap (ABFS raw v2.0 minus Komorniczak best, mean over reps) -- {stream_name}',
                     fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.1, orientation='horizontal')
        fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(); print(f"  Gap heatmap saved: {fname}")


print("\nAnalysis 4 complete.")