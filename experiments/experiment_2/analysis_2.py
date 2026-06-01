# analysis_2.py
# ============================================================
# Analysis of Experiment 2 results (stream configuration sensitivity).
#
# Usage:
#   python analysis_2.py --sanity --variance --performance --shap --metrics --grid
#
# Loads pre-computed results (.npy files) from
# results/experiment_2/ and produces:
#
# We compare the Komorniczak vs ABFS raw score meta-features (v2.0) across
# the 4x4 grid of stream configurations (chunk_size x n_informative) and 2 drift types, for both CV and prequential protocols.
#   Per grid cell (for each chunk_size x n_informative x drift_type):
#   1. Sanity check plots (per replication) - but ONLY FIRST REPLICATION PER CONFIGURATION TO AVOID TOO MANY PLOTS:
#      - Relevance scores over time
#      - Meta-features over windows
#      - PCA projection of meta-feature vectors
#      - 16 cells x 2 drift types x 2 feature sets = 64 figures per plot type

#   2. Performance variance across replications (CV protocol): mean balanced accuracy per replication per classifier
#      - performance_variance_{abfs|komor}_{tag}.png
#      - 16 cells x 2 drift types x 2 feature sets = 64 figures
#
#   3. Performance trajectory over time (prequential protocol): cumulative balanced accuracy per window per classifier
#      - performance_over_time_{abfs|komor}_{tag}.png
#      - 16 cells x 2 drift types x 2 feature sets = 64 figures
#
#   4. SHAP analysis:
#      - Mean absolute SHAP values per meta-feature (MLP)
#
#   5. Additional metrics heatmaps (F1, Kappa):
#      - Mean, std, median across replications
#      - ABFS vs Komorniczak
#      - 16 cells x 2 drift types x 2 protocols x 2 metrics = 128 figures
#
#   Across the full 4x4 grid:
#   6. Gap heatmap:
#      - ABFS minus Komorniczak balanced accuracy for the best-performing
#        classifier per cell, shown as a 4x4 grid (chunk_size x n_informative)
#      - Positive values: ABFS is better; negative values: Komorniczak is better
#      - One heatmap per drift type x protocol = 4 heatmaps total:
#          gap_heatmap_cv_sudden.png, gap_heatmap_cv_gradual.png
#          gap_heatmap_preq_sudden.png, gap_heatmap_preq_gradual.png
#
#   7. Sensitivity curves:
#      - BA vs chunk_size (fixing n_informative=10, our baseline):
#        shows how performance changes as chunk_size increases,
#        one line per classifier
#      - BA vs n_informative (fixing chunk_size=200, our baseline):
#        shows how performance changes as n_informative increases,
#        same format as above
#      - One plot per axis x drift type x protocol = 8 plots total:
#          sensitivity_chunk_cv_sudden.png, sensitivity_chunk_cv_gradual.png
#          sensitivity_chunk_preq_sudden.png, sensitivity_chunk_preq_gradual.png
#          sensitivity_ninf_cv_sudden.png, sensitivity_ninf_cv_gradual.png
#          sensitivity_ninf_preq_sudden.png, sensitivity_ninf_preq_gradual.png
#
# Inputs (from results/experiment_2/):
#   cv_abfs_ba_chunk{cs}_ninf{ni}_{drift}.npy  shape: (n_reps, n_folds, n_clfs)
#   preq_abfs_ba_chunk{cs}_ninf{ni}_{drift}.npy shape: (n_reps, n_windows, n_clfs)
#   (same pattern for komor, f1, kappa)
#
# Outputs saved to results/experiment_2/figures/analysis/
# ============================================================
 
import argparse
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
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
 
from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (extract_metafeatures_raw, MF_NAMES_RAW)
from classifier_sweep_komor import BASE_CLFS
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
from plot_results import print_sanity_check_summary
 
 
# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description='Analysis for Experiment 2.')
parser.add_argument('--sanity', action='store_true', help='Sanity check plots per cell')
parser.add_argument('--variance', action='store_true', help='Performance variance plots (CV)')
parser.add_argument('--performance', action='store_true', help='Trajectory plots (prequential)')
parser.add_argument('--shap', action='store_true', help='SHAP analysis per cell')
parser.add_argument('--metrics', action='store_true', help='F1 and Kappa heatmaps per cell')
parser.add_argument('--grid', action='store_true', help='Gap heatmap + sensitivity curves')
args = parser.parse_args()
 
RUN_SANITY = args.sanity
RUN_VARIANCE = args.variance
RUN_PERFORMANCE = args.performance
RUN_SHAP = args.shap
RUN_METRICS = args.metrics
RUN_GRID = args.grid
 
print(f"\nRunning analysis for Experiment 2")
print(f"Sanity check: {RUN_SANITY}")
print(f"Variance: {RUN_VARIANCE}")
print(f"Performance: {RUN_PERFORMANCE}")
print(f"SHAP: {RUN_SHAP}")
print(f"Metrics: {RUN_METRICS}")
print(f"Grid analyses: {RUN_GRID}")
 
 
# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)
 
 
# ============================================================
#  CONFIGURATION
# ============================================================
N_CHUNKS = 5000
N_FEATURES = 20
WARMUP_WINDOWS = 10
SCORE_INTERVAL = 100
N_REPLICATIONS = 5
 
CHUNK_SIZES = [100, 200, 500, 1000]
N_INFORMATIVES = [3, 5, 10, 15]
 
np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")
 
DRIFT_CONFIGS = [('sudden', 20, 9999, 21), ('gradual', 6, 5, 25)]
 
MF_NAMES = [f'r_f{j+1}' for j in range(N_FEATURES)]  # 20 raw score features
 
clf_names_cv = [name for name, _ in BASE_CLFS]
clf_names_preq = [name for name, _ in BASE_CLFS_PREQUENTIAL]
 
CLF_COLORS = {
    'GNB': '#e6194b',
    'KNN': '#3cb44b',
    'SVM': '#4363d8',
    'DT':  '#f58231',
    'MLP': '#911eb4',
    'PAC': '#42d4f4',
    'HT':  '#f032e6',
}
 
palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', '#5e5151', '#08332b',
    '#000000', '#a9a9a9', '#ff69b4', '#00ced1', '#ff8c00'
]
 
 
# ============================================================
#  HELPERS
# ============================================================
 
def make_tag(chunk_size, n_informative, drift_type):
    return f"chunk{chunk_size}_ninf{n_informative}_{drift_type}"
 
 
def load(prefix, tag, optional=False):
    """Load a .npy file from RESULTS_DIR. Returns None if optional and missing."""
    path = os.path.join(RESULTS_DIR, f'{prefix}_{tag}.npy')
    if not os.path.exists(path):
        if optional:
            return None
        print(f"  Warning: {path} not found, skipping.")
        return None
    return np.load(path)
 
 
def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing,
                        stream.n_drifts)[1][::chunk_size]
    concept    = 0
    decreasing = True
    labels     = []
    for chunk in range(n_chunks):
        if decreasing:
            if concept % 4 == 0:
                if e[chunk] < 0.9:  concept += 1
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
                if e[chunk] > 0.1:  concept += 1
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
    return [i for i in range(1, n_chunks)
            if concept_labels_all[i] != concept_labels_all[i-1]]
 
 
def extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative):
    """
    Re-generate one stream and extract ABFS relevance scores and
    raw score meta-features (v2.0) for sanity check and SHAP.
    """
    config = dict(
        n_drifts                = n_drifts,
        n_chunks                = N_CHUNKS,
        chunk_size              = chunk_size,
        n_features              = N_FEATURES,
        n_informative           = n_informative,
        n_redundant             = 0,
        n_repeated              = 0,
        concept_sigmoid_spacing = concept_sigmoid_spacing,
        random_state            = rs,
    )
    stream = StreamGenerator(**config)
 
    # pass 1: relevance scores over time
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size,
                      class_window_size=chunk_size)
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
            int(np.bincount(
                concept_selector_saved[i*chunk_size:(i+1)*chunk_size]
            ).argmax())
            for i in range(N_CHUNKS)
        ])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, chunk_size)
 
    boundaries = get_concept_boundaries(concept_labels_all, N_CHUNKS)
 
    # pass 2: raw score meta-features (v2.0)
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size,
                      class_window_size=chunk_size)
    meta_features  = []
    concept_labels = []
    wt_prev        = None
    window_counter = 0
 
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt = abfs.relevance_scores()
        abfs.pop_drift_count()
        if window_counter >= WARMUP_WINDOWS:
            meta_features.append(extract_metafeatures_raw(wt))
            concept_labels.append(concept_labels_all[window_counter])
        wt_prev = wt
        window_counter += 1
 
    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
 
    return scores_over_time, concept_labels_all, boundaries, X, y
 
 
def get_stream_boundaries_meta(drift_type, n_drifts, concept_sigmoid_spacing,
                               chunk_size, n_informative):
    """Get concept boundaries in meta-window index space (after warmup)."""
    _, concept_labels_all, boundaries, _, _ = extract_stream_data(
        RANDOM_STATES[0], drift_type, n_drifts, concept_sigmoid_spacing,
        chunk_size, n_informative)
    boundaries_meta = [b - WARMUP_WINDOWS for b in boundaries
                       if b - WARMUP_WINDOWS > 0]
    return boundaries_meta
 
 
# ============================================================
#  1. SANITY CHECK PLOTS
# ============================================================
if RUN_SANITY:
    print("\n" + "="*60)
    print("1. SANITY CHECK PLOTS")
    print("="*60)
 
    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
                print(f"\n{tag}")

                # only run sanity check for the first replication (rep 0)
                for rep_id, rs in enumerate(RANDOM_STATES[:1]): 
                    scores_over_time, concept_labels_all, boundaries, X, y = \
                        extract_stream_data(rs, drift_type, n_drifts,
                                            concept_sigmoid_spacing,
                                            chunk_size, n_informative)
                    unique_concepts = np.unique(y)
 
                    # relevance scores over time
                    fig, ax = plt.subplots(figsize=(14, 4))
                    for j in range(N_FEATURES):
                        ax.plot(scores_over_time[:, j], label=f'f{j+1}',
                                linewidth=0.8)
                    for b in boundaries:
                        ax.axvline(x=b * chunk_size // SCORE_INTERVAL,
                                   color='red', linestyle='--',
                                   linewidth=0.8, alpha=0.6)
                    ax.axvline(x=-1, color='red', linestyle='--',
                               linewidth=0.8, label='concept boundary')
                    ax.set_xlabel('Time (x100 instances)')
                    ax.set_ylabel('Relevance score')
                    ax.set_title(f'Relevance scores - {tag} - rep{rep_id}')
                    ax.legend(ncol=5, fontsize=7)
                    fig.tight_layout()
                    fname = os.path.join(FIGURES_DIR,
                        f'relevance_scores_{tag}_rep{rep_id}.png')
                    if not os.path.exists(fname):
                        fig.savefig(fname, dpi=150)
                        print(f"  Saved: {fname}")
                    else:
                        print(f"File already exists: {fname}")
                    plt.close()
 
                    # meta-features over windows
                    n_cols = 5
                    n_rows = (N_FEATURES + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(n_rows, n_cols,
                                             figsize=(4*n_cols, 3*n_rows))
                    axes = axes.flatten()
                    for k in range(N_FEATURES):
                        axes[k].plot(X[:, k], color='steelblue', linewidth=0.8)
                        for b in boundaries:
                            drift_w = b - WARMUP_WINDOWS
                            if drift_w > 0:
                                axes[k].axvline(x=drift_w, color='red',
                                                linestyle='--', linewidth=0.8)
                        axes[k].set_title(MF_NAMES[k], fontsize=8)
                        axes[k].set_xlabel('Window', fontsize=7)
                    for k in range(N_FEATURES, len(axes)):
                        axes[k].set_visible(False)
                    fig.suptitle(f'Meta-features (raw v2.0) - {tag} - rep{rep_id}',
                                 fontsize=10)
                    fig.tight_layout()
                    fname = os.path.join(FIGURES_DIR,
                        f'metafeatures_{tag}_rep{rep_id}.png')
                    if not os.path.exists(fname):
                        fig.savefig(fname, dpi=150)
                        print(f"  Saved: {fname}")
                    else:
                        print(f"File already exists: {fname}")
                    plt.close()
 
                    # PCA
                    colors = {c: palette[i % len(palette)]
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
                    ax.set_title(f'PCA - {tag} - rep{rep_id}')
                    ax.legend(ncol=4, fontsize=7)
                    fig.tight_layout()
                    fname = os.path.join(FIGURES_DIR,
                        f'pca_{tag}_rep{rep_id}.png')
                    if not os.path.exists(fname):
                        fig.savefig(fname, dpi=150)
                        print(f"  Saved: {fname}")
                    else:
                        print(f"File already exists: {fname}")
                    plt.close()
 
                print(f"Sanity check plots saved for {tag}")
 
 
# ============================================================
#  2. PERFORMANCE VARIANCE ACROSS REPLICATIONS (CV)
# ============================================================
if RUN_VARIANCE:
    print("\n" + "="*60)
    print("2. PERFORMANCE VARIANCE (CV)")
    print("="*60)
 
    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
 
                for prefix, label, clf_names in [
                    ('cv_abfs_ba',  'ABFS raw v2.0',     clf_names_cv),
                    ('cv_komor_ba', 'Komorniczak',        clf_names_cv),
                ]:
                    data = load(prefix, tag)
                    if data is None:
                        continue
                    # shape: (n_reps, n_folds, n_clfs)
                    rep_means   = np.mean(data,   axis=1)
                    rep_stds    = np.std(data,    axis=1)
                    rep_medians = np.median(data, axis=1)
 
                    fig, ax = plt.subplots(figsize=(10, 4))
                    x     = np.arange(N_REPLICATIONS)
                    width = 0.15
                    for clf_id, name in enumerate(clf_names):
                        ax.bar(x + clf_id*width, rep_means[:, clf_id],
                               width=width, label=name, alpha=0.7)
                        ax.errorbar(x + clf_id*width, rep_means[:, clf_id],
                                    yerr=rep_stds[:, clf_id],
                                    fmt='none', color='black',
                                    capsize=3, linewidth=1)
                        ax.scatter(x + clf_id*width, rep_medians[:, clf_id],
                                   marker='_', color='black', s=100, zorder=5)
                    ax.axhline(y=1/n_concepts, color='red', linestyle='--',
                               linewidth=1.0, label='random baseline')
                    ax.set_xlabel('Replication')
                    ax.set_ylabel('Mean balanced accuracy')
                    ax.set_title(f'Performance variance across replications - {label} - {tag}')
                    ax.set_xticks(x + width * 2)
                    ax.set_xticklabels(
                        [f'Rep{i+1}\n(s={RANDOM_STATES[i]})'
                         for i in range(N_REPLICATIONS)], fontsize=8)
                    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1),
                              loc='upper left')
                    fig.tight_layout()
                    short = prefix.replace('cv_', '').replace('_ba', '')
                    fname = os.path.join(FIGURES_DIR,
                        f'performance_variance_{short}_{tag}.png')
                    if not os.path.exists(fname):
                        fig.savefig(fname, dpi=150)
                        print(f"  Saved: {fname}")
                    else:
                        print(f"File already exists: {fname}")
                    plt.close()
 
 
# ============================================================
#  3. PERFORMANCE TRAJECTORY OVER TIME (PREQUENTIAL)
# ============================================================
if RUN_PERFORMANCE:
    print("\n" + "="*60)
    print("3. PERFORMANCE TRAJECTORY (PREQUENTIAL)")
    print("="*60)
 
    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
                
                # check if all trajectory figures for this cell exist before
                shorts = ['abfs', 'komor']
                fnames = [os.path.join(FIGURES_DIR,
                    f'performance_over_time_{s}_{tag}.png') for s in shorts]
                if all(os.path.exists(f) for f in fnames):
                    print(f"  Skipping trajectory (all exist): {tag}")
                    continue

                boundaries_meta = get_stream_boundaries_meta(
                    drift_type, n_drifts, concept_sigmoid_spacing,
                    chunk_size, n_informative)
                if drift_type == 'gradual':
                    main_boundaries = boundaries_meta[::4]
                else:
                    main_boundaries = boundaries_meta
 
                random_baseline = 1 / n_concepts
 
                for prefix, label, clf_names in [
                    ('preq_abfs_ba',  'ABFS raw v2.0', clf_names_preq),
                    ('preq_komor_ba', 'Komorniczak',   clf_names_preq),
                ]:
                    data = load(prefix, tag)
                    if data is None:
                        continue
                    # shape: (n_reps, n_windows, n_clfs)
                    n_windows = data.shape[1]
                    x_axis    = np.arange(n_windows)

                    short = prefix.replace('preq_', '').replace('_ba', '')
                    fname = os.path.join(FIGURES_DIR,
                        f'performance_over_time_{short}_{tag}.png')
                    if os.path.exists(fname):
                        print(f"  Skipping (exists): {fname}")
                        continue

                    fig, ax = plt.subplots(figsize=(14, 4))
                    for clf_id, name in enumerate(clf_names):
                        mean_traj = np.mean(data[:, :, clf_id], axis=0)
                        std_traj  = np.std(data[:, :, clf_id],  axis=0)
                        color     = CLF_COLORS.get(name, f'C{clf_id}')
                        ax.plot(x_axis, mean_traj, label=name,
                                color=color, linewidth=1.5)
                        ax.fill_between(x_axis,
                            mean_traj - std_traj,
                            mean_traj + std_traj,
                            alpha=0.15, color=color)
                    for b in main_boundaries:
                        ax.axvline(x=b, color='grey', linestyle='--',
                                   linewidth=0.8, alpha=0.6)
                    ax.axhline(y=random_baseline, color='red',
                               linestyle='--', linewidth=1.0,
                               label='random baseline')
                    ax.set_xlabel('Window')
                    ax.set_ylabel('Cumulative balanced accuracy')
                    ax.set_title(
                        f'Performance trajectory over time - {label} - {tag}')
                    ax.legend(fontsize=9, ncol=3)
                    ax.set_xlim(0, n_windows)
                    ax.set_ylim(0, 1)
                    fig.tight_layout()
                    if not os.path.exists(fname):
                        fig.savefig(fname, dpi=150)
                        print(f"Trajectory plot saved: {fname}")
                    else:
                        print(f"File already exists: {fname}")
                    plt.close()
 
 
# ============================================================
#  4. SHAP ANALYSIS
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("4. SHAP ANALYSIS")
    print("="*60)
 
    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
                # skip if SHAP plot already exists
                fname = os.path.join(FIGURES_DIR, f'shap_{tag}.png')
                if os.path.exists(fname):
                    print(f"  Skipping SHAP (exists): {fname}")
                    continue
                print(f"\n  SHAP: {tag}")

                all_X, all_y = [], []
                for rs in RANDOM_STATES:
                    _, _, _, X, y = extract_stream_data(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)
                    all_X.append(X)
                    all_y.append(y)
 
                X_all = np.vstack(all_X)
                y_all = np.concatenate(all_y)
                X_all[np.isnan(X_all)] = 1
                X_all[np.isinf(X_all)] = 1
 
                mlp = clone(dict(BASE_CLFS)['MLP'])
                mlp.fit(X_all, y_all)
 
                explainer   = shap.KernelExplainer(
                    mlp.predict_proba, shap.sample(X_all, 100))
                shap_values = explainer.shap_values(
                    shap.sample(X_all, 200), nsamples=100)
 
                shap_array = np.array(shap_values)
                if shap_array.ndim == 3:
                    mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
                else:
                    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
 
                sorted_idx = np.argsort(mean_abs_shap)[::-1]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(MF_NAMES)), mean_abs_shap[sorted_idx],
                       color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(MF_NAMES)))
                ax.set_xticklabels([MF_NAMES[i] for i in sorted_idx],
                                   rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Mean absolute SHAP value')
                ax.set_title(f'SHAP - raw scores v2.0 - {tag}\n'
                             f'(MLP, {N_REPLICATIONS} replications combined)')
                fig.tight_layout()
                fname = os.path.join(FIGURES_DIR, f'shap_{tag}.png')
                if not os.path.exists(fname):
                    fig.savefig(fname, dpi=150)
                    print(f"SHAP saved: {fname}")
                else:
                    print(f"File already exists: {fname}")
                plt.close()
 
 
# ============================================================
#  5. ADDITIONAL METRICS HEATMAPS (F1, KAPPA)
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60)
    print("5. ADDITIONAL METRICS HEATMAPS")
    print("="*60)
 
    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
 
                for metric, metric_label in [('f1', 'F1'), ('kappa', 'Kappa')]:
 
                    for protocol, clf_names, agg in [
                        ('cv',   clf_names_cv,   lambda d: np.mean(d, axis=(0, 1))),
                        ('preq', clf_names_preq, lambda d: np.mean(d[:, -1, :], axis=0)),
                    ]:
                        abfs_data  = load(f'{protocol}_abfs_{metric}',  tag)
                        komor_data = load(f'{protocol}_komor_{metric}', tag)
                        if abfs_data is None or komor_data is None:
                            continue
 
                        if protocol == 'cv':
                            abfs_mean   = np.mean(abfs_data,   axis=(0, 1))
                            abfs_std    = np.std(abfs_data,    axis=(0, 1))
                            abfs_median = np.median(abfs_data, axis=(0, 1))
                            komor_mean   = np.mean(komor_data,   axis=(0, 1))
                            komor_std    = np.std(komor_data,    axis=(0, 1))
                            komor_median = np.median(komor_data, axis=(0, 1))
                        else:
                            # prequential: take final window
                            abfs_final  = abfs_data[:, -1, :]
                            komor_final = komor_data[:, -1, :]
                            abfs_mean   = np.mean(abfs_final,   axis=0)
                            abfs_std    = np.std(abfs_final,    axis=0)
                            abfs_median = np.median(abfs_final, axis=0)
                            komor_mean   = np.mean(komor_final,   axis=0)
                            komor_std    = np.std(komor_final,    axis=0)
                            komor_median = np.median(komor_final, axis=0)
 
                        rows = [
                            ('ABFS raw v2.0',    abfs_mean,  abfs_std,  abfs_median),
                            ('Komorniczak',       komor_mean, komor_std, komor_median),
                        ]
                        matrix        = np.array([r[1] for r in rows])
                        matrix_std    = np.array([r[2] for r in rows])
                        matrix_median = np.array([r[3] for r in rows])
                        row_labels    = [r[0] for r in rows]
 
                        fig, ax = plt.subplots(figsize=(10, 2.5))
                        im = ax.imshow(matrix, vmin=0.0, vmax=1.0,
                                       cmap='Blues', aspect='auto')
                        for i in range(len(rows)):
                            for j in range(len(clf_names)):
                                val    = matrix[i, j]
                                std    = matrix_std[i, j]
                                median = matrix_median[i, j]
                                txt_color = 'white' if val > 0.6 else 'black'
                                ax.text(j, i,
                                    f'{val:.3f}\n(±{std:.3f})\nmed:{median:.3f}',
                                    ha='center', va='center', fontsize=7,
                                    color=txt_color, linespacing=1.4)
                        ax.set_xticks(range(len(clf_names)))
                        ax.set_xticklabels(clf_names, fontsize=9)
                        ax.set_yticks(range(len(rows)))
                        ax.set_yticklabels(row_labels, fontsize=9)
                        ax.set_title(
                            f'{metric_label} ({protocol}) - {tag}',
                            fontsize=10)
                        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
                        fig.tight_layout()
                        fname = os.path.join(FIGURES_DIR,
                            f'heatmap_{metric}_{protocol}_{tag}.png')
                        if not os.path.exists(fname):
                            fig.savefig(fname, dpi=150)
                            print(f"{metric_label} heatmap saved: {fname}")
                        else:
                            print(f"File already exists: {fname}")
                        plt.close()
 
 
# ============================================================
#  6 + 7. GRID ANALYSES: GAP HEATMAP + SENSITIVITY CURVES
# ============================================================
if RUN_GRID:
    print("\n" + "="*60)
    print("6+7. GAP HEATMAP + SENSITIVITY CURVES")
    print("="*60)
 
    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
 
        # ---- build the full 4x4 grid of results ----
        # grid_cv[i, j]   = mean BA across reps/folds for best clf, CV protocol
        # grid_preq[i, j] = mean BA across reps at final window for best clf, preq
 
        grid_abfs_cv    = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
        grid_komor_cv   = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
        grid_abfs_preq  = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
        grid_komor_preq = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
 
        # also store per-classifier for sensitivity curves
        # shape: (n_chunk_sizes, n_informatives, n_clfs)
        grid_abfs_cv_clf  = np.full(
            (len(CHUNK_SIZES), len(N_INFORMATIVES), len(clf_names_cv)), np.nan)
        grid_komor_cv_clf = np.full(
            (len(CHUNK_SIZES), len(N_INFORMATIVES), len(clf_names_cv)), np.nan)
        grid_abfs_preq_clf  = np.full(
            (len(CHUNK_SIZES), len(N_INFORMATIVES), len(clf_names_preq)), np.nan)
        grid_komor_preq_clf = np.full(
            (len(CHUNK_SIZES), len(N_INFORMATIVES), len(clf_names_preq)), np.nan)
 
        for i, chunk_size in enumerate(CHUNK_SIZES):
            for j, n_informative in enumerate(N_INFORMATIVES):
                tag = make_tag(chunk_size, n_informative, drift_type)
 
                cv_abfs  = load('cv_abfs_ba',   tag, optional=True)
                cv_komor = load('cv_komor_ba',  tag, optional=True)
                pr_abfs  = load('preq_abfs_ba', tag, optional=True)
                pr_komor = load('preq_komor_ba',tag, optional=True)
 
                if cv_abfs is not None:
                    per_clf = np.mean(cv_abfs,  axis=(0, 1))  # (n_clfs,)
                    grid_abfs_cv[i, j]      = np.max(per_clf)
                    grid_abfs_cv_clf[i, j]  = per_clf
                if cv_komor is not None:
                    per_clf = np.mean(cv_komor, axis=(0, 1))
                    grid_komor_cv[i, j]     = np.max(per_clf)
                    grid_komor_cv_clf[i, j] = per_clf
                if pr_abfs is not None:
                    per_clf = np.mean(pr_abfs[:, -1, :], axis=0)
                    grid_abfs_preq[i, j]      = np.max(per_clf)
                    grid_abfs_preq_clf[i, j]  = per_clf
                if pr_komor is not None:
                    per_clf = np.mean(pr_komor[:, -1, :], axis=0)
                    grid_komor_preq[i, j]     = np.max(per_clf)
                    grid_komor_preq_clf[i, j] = per_clf
 
        x_labels = [str(ni) for ni in N_INFORMATIVES]
        y_labels = [str(cs) for cs in CHUNK_SIZES]
 
        # ---- 6. GAP HEATMAPS ----
        for protocol_label, gap_grid in [
            ('CV',          grid_abfs_cv   - grid_komor_cv),
            ('Prequential', grid_abfs_preq - grid_komor_preq),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            vmax = np.nanmax(np.abs(gap_grid))
            im = ax.imshow(gap_grid, vmin=-vmax, vmax=vmax,
                           cmap='RdBu', aspect='auto')
            for i in range(len(CHUNK_SIZES)):
                for j in range(len(N_INFORMATIVES)):
                    val = gap_grid[i, j]
                    if not np.isnan(val):
                        txt_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                        ax.text(j, i, f'{val:+.3f}',
                                ha='center', va='center',
                                fontsize=10, color=txt_color)
                    else:
                        ax.text(j, i, 'N/A', ha='center', va='center',
                                fontsize=9, color='grey')
            ax.set_xticks(range(len(N_INFORMATIVES)))
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.set_yticks(range(len(CHUNK_SIZES)))
            ax.set_yticklabels(y_labels, fontsize=10)
            ax.set_xlabel('n_informative', fontsize=11)
            ax.set_ylabel('chunk_size',    fontsize=11)
            ax.set_title(
                f'Gap heatmap (ABFS - Komorniczak) - {protocol_label}\n'
                f'{drift_type} drift ({n_concepts} concepts) '
                f'[best classifier per cell]',
                fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            proto_short = protocol_label.lower().replace('quential', 'q')
            fname = os.path.join(FIGURES_DIR,
                f'gap_heatmap_{proto_short}_{drift_type}.png')
            if not os.path.exists(fname):
                fig.savefig(fname, dpi=150)
                print(f"Gap heatmap saved: {fname}")
            else:
                print(f"File already exists: {fname}")
            plt.close()
            
 
        # ---- 7. SENSITIVITY CURVES ----
        ni_baseline_idx = N_INFORMATIVES.index(10) # n_informative=10 fixed for chunk_size curves
        cs_baseline_idx = CHUNK_SIZES.index(200) # chunk_size=200 fixed for n_informative curves
        print(f"Baseline indices - n_informative: {ni_baseline_idx}, chunk_size: {cs_baseline_idx}")

        
        for protocol_label, abfs_grid_clf, komor_grid_clf, clf_names_used in [
            ('CV',          grid_abfs_cv_clf,   grid_komor_cv_clf,   clf_names_cv),
            ('Prequential', grid_abfs_preq_clf, grid_komor_preq_clf, clf_names_preq),
        ]:
            # --- BA vs chunk_size (n_informative=10 fixed) ---
            fig, ax = plt.subplots(figsize=(8, 4))
            for clf_id, name in enumerate(clf_names_used):
                abfs_vals  = abfs_grid_clf[:,  ni_baseline_idx, clf_id]
                komor_vals = komor_grid_clf[:, ni_baseline_idx, clf_id]
                color = CLF_COLORS.get(name, f'C{clf_id}')
                ax.plot(CHUNK_SIZES, abfs_vals,  color=color,
                        label=f'{name} ABFS',  linewidth=1.5,
                        marker='o', markersize=5)
                ax.plot(CHUNK_SIZES, komor_vals, color=color,
                        label=f'{name} Komor', linewidth=1.5,
                        linestyle='--', marker='s', markersize=5)
            ax.axhline(y=1/n_concepts, color='red', linestyle=':',
                       linewidth=1.0, label='random baseline')
            ax.set_xlabel('chunk_size', fontsize=11)
            ax.set_ylabel('Mean balanced accuracy', fontsize=10)
            ax.set_title(
                f'BA vs chunk_size (n_informative=10) - {protocol_label}\n'
                f'{drift_type} drift', fontsize=11)
            ax.legend(fontsize=8, ncol=2,
                      bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_xticks(CHUNK_SIZES)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            proto_short = protocol_label.lower().replace('quential', 'q')
            fname = os.path.join(FIGURES_DIR,
                f'sensitivity_chunk_{proto_short}_{drift_type}.png')
            if not os.path.exists(fname):
                fig.savefig(fname, dpi=150)
                print(f"Sensitivity (chunk) saved: {fname}")
            else:
                print(f"File already exists: {fname}")
            plt.close()
 
            # --- BA vs n_informative (chunk_size=200 fixed) ---
            fig, ax = plt.subplots(figsize=(8, 4))
            for clf_id, name in enumerate(clf_names_used):
                abfs_vals  = abfs_grid_clf[cs_baseline_idx,  :, clf_id]
                komor_vals = komor_grid_clf[cs_baseline_idx, :, clf_id]
                color = CLF_COLORS.get(name, f'C{clf_id}')
                ax.plot(N_INFORMATIVES, abfs_vals,  color=color,
                        label=f'{name} ABFS',  linewidth=1.5,
                        marker='o', markersize=5)
                ax.plot(N_INFORMATIVES, komor_vals, color=color,
                        label=f'{name} Komor', linewidth=1.5,
                        linestyle='--', marker='s', markersize=5)
            ax.axhline(y=1/n_concepts, color='red', linestyle=':',
                       linewidth=1.0, label='random baseline')
            ax.set_xlabel('n_informative', fontsize=11)
            ax.set_ylabel('Mean balanced accuracy', fontsize=10)
            ax.set_title(
                f'BA vs n_informative (chunk_size=200) - {protocol_label}\n'
                f'{drift_type} drift', fontsize=11)
            ax.legend(fontsize=8, ncol=2,
                      bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_xticks(N_INFORMATIVES)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR,
                f'sensitivity_ninf_{proto_short}_{drift_type}.png')
            if not os.path.exists(fname):
                fig.savefig(fname, dpi=150)
                print(f"Sensitivity (n_informative) saved: {fname}")
            else:
                print(f"File already exists: {fname}")
            plt.close()
 
print("\nAnalysis complete.")
 