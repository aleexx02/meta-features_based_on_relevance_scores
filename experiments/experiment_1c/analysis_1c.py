# analysis_1c.py
# ============================================================
# Analysis of Experiment 1c results (prequential evaluation).
#
# Usage:
#   python analysis_1c.py --sanity --performance --shap --metrics --stream_analysis
#
# Loads pre-computed results (.npy files) from
# results/experiment_1c/ and produces:
#
#   1. Sanity check plots (per replication x drift type):
#      - Relevance scores over time
#      - Meta-features over windows
#      - PCA projection of meta-feature vectors
#
#   2. Performance trajectory over time:
#      - Cumulative balanced accuracy per window per classifier
#      - Mean +/- std band across replications
#      - Concept boundaries marked as vertical lines
#      - One plot per meta-feature set per drift type
#
#   3. SHAP analysis:
#      - Mean absolute SHAP values per meta-feature (4 classifiers)
#      - One plot per meta-feature set per drift type
#        averaged over replications
#
#   4. Additional metrics heatmaps (F1, Kappa):
#      - Final value per classifier per meta-feature set
#      - Mean, std, median across replications
#      - Includes Komorniczak baseline
#
#   5. Stream analysis plots (per replication x drift type):
#      - Feature-mean drift intensity, ABFS relevance-score change,
#        and label entropy overlaid on one plot
#      - Class distribution of the underlying binary classification
#        target over time
#      Same diagnostics as Experiment 3's --stream_analysis, computed
#      directly from a freshly-generated stream (rather than loaded
#      from a saved npy) since this experiment's synthetic streams
#      aren't persisted to disk the way the real streams are.
#
#   6. Gap heatmap (one per drift type):
#      - gap_heatmap_preq_{drift_type}.png
#      - Gap (ABFS raw v2.0 minus best-of-9 Komorniczak measure group)
#        at the final window, mean across replications, one cell per
#        classifier. Same "raw v2.0 vs Komorniczak best" comparison
#        used in analysis_2.py / analysis_3.py; the unit here is
#        drift type rather than a chunk_size x n_informative grid cell
#        (Exp2) or a stream (Exp3), since 1c doesn't sweep those.
#

# SHAP classifiers (sklearn-compatible proxies):
#   GNB -- GaussianNB
#   KNN -- KNeighborsClassifier
#   HT  -- DecisionTreeClassifier
#   MLP -- MLPClassifier

# Inputs (from results/experiment_1c/):
#   clf_ba_*.npy, clf_f1_*.npy, clf_kappa_*.npy
#   clf_komor_concept_classif_ba_*.npy, etc.
# Each file has shape (n_replications, n_windows, n_clfs)
# Komorniczak files: (n_measures, n_replications, n_windows, n_clfs)
#
# Outputs saved to results/experiment_1c/figures/analysis/
# ============================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import clone
from scipy.stats import entropy
import shap
import os
import sys
import warnings
import csv

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures_raw, extract_metafeatures_raw_temporal,
    MF_NAMES_RAW, MF_NAMES_RAW_TEMPORAL
)
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from plot_results import print_sanity_check_summary

# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description='Analysis for Experiment 1c.')
parser.add_argument('--sanity',     action='store_true', help='Run sanity check plots')
parser.add_argument('--performance', action='store_true', help='Run performance trajectory plots')
parser.add_argument('--shap',       action='store_true', help='Run SHAP analysis')
parser.add_argument('--metrics',    action='store_true', help='Run metrics heatmaps')
parser.add_argument('--stream_analysis', action='store_true', help='Run stream analysis plots')
parser.add_argument('--gap', action='store_true', help='Run gap heatmap (ABFS raw v2.0 vs Komorniczak best)')
parser.add_argument('--summary', action='store_true')

args = parser.parse_args()

RUN_SANITY_CHECK = args.sanity
RUN_PERFORMANCE   = args.performance
RUN_SHAP         = args.shap
RUN_METRICS      = args.metrics
RUN_STREAM_ANALYSIS = args.stream_analysis
RUN_GAP          = args.gap

print(f"\nRunning analysis for Experiment 1c")
print(f"Sanity check    : {RUN_SANITY_CHECK}")
print(f"Performance     : {RUN_PERFORMANCE}")
print(f"SHAP            : {RUN_SHAP}")
print(f"Metrics         : {RUN_METRICS}")
print(f"Stream analysis : {RUN_STREAM_ANALYSIS}")
print(f"Gap             : {RUN_GAP}")

# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_1c')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_1c', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  CONFIGURATION
# ============================================================
N_CHUNKS       = 5000
CHUNK_SIZE     = 200
N_FEATURES     = 10
WARMUP_WINDOWS = 10
SCORE_INTERVAL = 100
N_REPLICATIONS = 5

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

MF_CONFIGS = [
    ('raw',          'Raw scores (v2.0)',      MF_NAMES_RAW,          5),
    ('raw_temporal', 'Raw + temporal (v2.1)',  MF_NAMES_RAW_TEMPORAL, 6),
]

DRIFT_CONFIGS = [
    ('sudden',  20, 9999, 21),
    ('gradual',  6,    5, 25),
]

ABFS_MF_CONFIGS_FULL = [
    ('aggstats',     'Aggregate stats (v1.1)'),
    ('raw',          'Raw scores (v2.0)'),
    ('raw_temporal', 'Raw + temporal (v2.1)'),
]

ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {'aggstats': 'Aggstats (v1.1)', 'raw': 'Raw scores (v2.0)',
                    'raw_temporal': 'Raw + temporal (v2.1)'}


MEASURES = ['clustering', 'complexity', 'concept', 'general',
    'info-theory', 'itemset', 'landmarking', 'model-based', 'statistical']
STATISTICAL_IDX = MEASURES.index('statistical')

clf_names = [name for name, _ in BASE_CLFS_PREQUENTIAL]

# colors per classifier for trajectory plots
CLF_COLORS = {
    'GNB': '#e6194b',
    'KNN': '#3cb44b',
    'PAC': '#4363d8',
    'HT':  '#f58231',
    'MLP':  '#911eb4',
}

palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', '#5e5151', '#08332b',
    '#000000', '#a9a9a9', '#ff69b4', '#00ced1', '#ff8c00'
]

# SHAP: sklearn-compatible proxies for all 4 classifiers
SHAP_CLFS = [
    ('GNB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('HT',  DecisionTreeClassifier(random_state=11313)),
    ('MLP', MLPClassifier(random_state=11313)),
]


# ============================================================
#  HELPERS
# ============================================================

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
    return [i for i in range(1, n_chunks) if
            concept_labels_all[i] != concept_labels_all[i-1]]


def extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing):
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
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
        accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
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

    if drift_type == 'sudden':
        concept_labels_all = np.array([
            int(np.bincount(concept_selector_saved[
                i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]).argmax())
            for i in range(N_CHUNKS)])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, CHUNK_SIZE)

    boundaries = get_concept_boundaries(concept_labels_all, N_CHUNKS)

    # pass 2: meta-features
    results = {}
    for mf_type, mf_label, mf_names, _ in MF_CONFIGS:
        abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
            accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
        meta_features  = []
        concept_labels = []
        wt_prev        = None
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


def get_stream_boundaries(drift_type, n_drifts, concept_sigmoid_spacing):
    """Get concept boundaries from one representative stream (seed=RANDOM_STATES[0])."""
    config = {
        'n_drifts': n_drifts,
        'n_chunks': N_CHUNKS,
        'chunk_size': CHUNK_SIZE,
        'n_features': N_FEATURES,
        'n_informative': N_FEATURES,
        'n_redundant': 0,
        'n_repeated': 0,
        'concept_sigmoid_spacing': concept_sigmoid_spacing,
        'random_state': RANDOM_STATES[0]
    }
    stream = StreamGenerator(**config)
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
        accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
    concept_selector_saved = stream.concept_selector.copy()

    if drift_type == 'sudden':
        concept_labels_all = np.array([
            int(np.bincount(concept_selector_saved[
                i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]).argmax())
            for i in range(N_CHUNKS)])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, CHUNK_SIZE)

    # shift by warmup
    boundaries_stream = get_concept_boundaries(concept_labels_all, N_CHUNKS)
    boundaries_meta   = [b - WARMUP_WINDOWS for b in boundaries_stream
                         if b - WARMUP_WINDOWS > 0]
    return boundaries_meta


def extract_stream_diagnostics(rs, drift_type, n_drifts, concept_sigmoid_spacing):
    """
    One pass over the stream computing PER-CHUNK diagnostics: class
    distribution (of the underlying binary classification target that
    StreamGenerator produces -- NOT the concept label), feature-mean
    drift intensity, label entropy, and ABFS relevance-score change
    (delta_relevance) -- all on the same per-chunk index, with no
    WARMUP_WINDOWS shift (unlike extract_stream_data's meta-feature
    pass), since this diagnostic covers the whole stream from chunk 0
    rather than feeding the concept classifiers.

    Mirrors generate_real_streams.py's run_stream_analysis(), adapted
    for a freshly-generated (not saved-to-disk) synthetic stream.
    Concept boundaries reuse this file's own local
    assign_labels_gradual / bincount-majority-vote logic (the same
    pattern used by extract_stream_data and get_stream_boundaries
    above), since this file doesn't share a consolidated
    concept-labelling helper with Experiment 2.
    """
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
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
        accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)

    class_distribution, drift_intensity, label_entropy_vals = [], [], []
    delta_relevance = []
    prev_mean, wt_prev = None, None
    # underlying StreamGenerator target is binary by default in this
    # project's Exp1c config (no n_classes override is set anywhere
    # above) -- not re-verified against the actual stream output here
    n_classes = 2

    stream.reset()
    for X_chunk, y_chunk in stream:
        counts = np.bincount(y_chunk, minlength=n_classes)
        probs  = counts / np.sum(counts)
        class_distribution.append(probs)

        mean = np.mean(X_chunk, axis=0)
        if prev_mean is None:
            drift_intensity.append(0.0)
        else:
            drift_intensity.append(np.linalg.norm(mean - prev_mean))
        prev_mean = mean
        label_entropy_vals.append(entropy(probs + 1e-10))

        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt = abfs.relevance_scores()
        if wt_prev is None:
            delta_relevance.append(0.0)
        else:
            delta_relevance.append(np.linalg.norm(wt - wt_prev))
        wt_prev = wt

    concept_selector_saved = stream.concept_selector.copy()
    if drift_type == 'sudden':
        concept_labels_all = np.array([
            int(np.bincount(concept_selector_saved[
                i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]).argmax())
            for i in range(N_CHUNKS)])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, CHUNK_SIZE)
    boundaries = get_concept_boundaries(concept_labels_all, N_CHUNKS)

    return (np.array(class_distribution), np.array(drift_intensity),
            np.array(label_entropy_vals), np.array(delta_relevance),
            boundaries)

def _final_per_clf(arr, has_reps):
    """Final-window BA per classifier. arr: (n_reps,n_win,n_clf) if has_reps
    else (n_win,n_clf). Returns (n_clf,) vector (mean over reps if present)."""
    if arr is None:
        return None
    return np.mean(arr[:, -1, :], axis=0) if has_reps else arr[-1, :]


def best_side(load_fn, keys, has_reps):
    """Given a list of (label, prefix) keys, return (best_label, best_clf, best_ba)
    = the single (group/version, classifier) with the highest final BA."""
    best = (None, None, -1.0)
    for label, prefix in keys:
        d = load_fn(prefix)
        v = _final_per_clf(d, has_reps)
        if v is None:
            continue
        j = int(np.nanargmax(v))
        if v[j] > best[2]:
            best = (label, clf_names[j], float(v[j]))
    return best


def write_summary_csv(path, title, header, rows):
    """Write the summary as a CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Saved: {path}")



# ============================================================
#  1. SANITY CHECK PLOTS
# ============================================================
if RUN_SANITY_CHECK:
    print("\n" + "="*60)
    print("1. SANITY CHECK PLOTS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for rep_id, rs in enumerate(RANDOM_STATES):
            print(f"\nDrift: {drift_type} | Rep {rep_id+1}/{N_REPLICATIONS} (seed={rs})")

            scores_over_time, concept_labels_all, boundaries, mf_results = \
                extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing)

            # relevance scores
            fig, ax = plt.subplots(figsize=(14, 4))
            for j in range(N_FEATURES):
                ax.plot(scores_over_time[:, j], label=f'f{j+1}')
            for b in boundaries:
                ax.axvline(x=b * CHUNK_SIZE // SCORE_INTERVAL, color='red',
                    linestyle='--', linewidth=1.0, alpha=0.7)
            ax.axvline(x=-1, color='red', linestyle='--', linewidth=1.0,
                label='concept boundary')
            ax.set_xlabel('Time (x100 instances)')
            ax.set_ylabel('Relevance score')
            ax.set_title(f'ABFS relevance scores - {drift_type} drift '
                f'(seed={rs}) - experiment [1c]')
            ax.legend(ncol=5, fontsize=8)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR,
                f'relevance_scores_{drift_type}_rep{rep_id}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"Relevance scores saved at: '{fname}'")

            for mf_type, mf_label, mf_names, n_mf_cols in MF_CONFIGS:
                X = mf_results[mf_type]['X']
                y = mf_results[mf_type]['y']
                unique_concepts = np.unique(y)

                # meta-features over windows
                fig, axes = plt.subplots(2, n_mf_cols, figsize=(4*n_mf_cols, 6))
                axes = axes.flatten()
                for k, name in enumerate(mf_names):
                    axes[k].plot(X[:, k], color='steelblue')
                    for b in boundaries:
                        drift_w = b - WARMUP_WINDOWS
                        if drift_w > 0:
                            axes[k].axvline(x=drift_w, color='red',
                                linestyle='--', linewidth=1.0)
                    axes[k].set_title(name, fontsize=9)
                    axes[k].set_xlabel('Window')
                    axes[k].set_ylabel('Value')
                fig.suptitle(f'Meta-features ({mf_type}) - {drift_type} drift '
                    f'(seed={rs}) - experiment [1c]', fontsize=11)
                fig.tight_layout()
                fname = os.path.join(FIGURES_DIR,
                    f'metafeatures_{mf_type}_{drift_type}_rep{rep_id}.png')
                fig.savefig(fname, dpi=150)
                plt.close()
                print(f"Meta-features saved at: '{fname}'")

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
                        alpha=0.6, edgecolors='none', s=30)
                ax.set_xlabel(
                    f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
                ax.set_ylabel(
                    f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
                ax.set_title(f'PCA ({mf_type}) - {drift_type} drift '
                    f'(seed={rs}) - experiment [1c]')
                ax.legend(ncol=4, fontsize=8)
                fig.tight_layout()
                fname = os.path.join(FIGURES_DIR,
                    f'pca_{mf_type}_{drift_type}_rep{rep_id}.png')
                fig.savefig(fname, dpi=150)
                plt.close()
                print(f"PCA saved at: '{fname}'")

                print_sanity_check_summary(
                    f'{drift_type} drift (seed={rs})', True,
                    mf_type, mf_names, X, y, X[:, :N_FEATURES], N_FEATURES)


# ============================================================
#  2. PERFORMANCE TRAJECTORY OVER TIME
# ============================================================
if RUN_PERFORMANCE:
    print("\n" + "="*60)
    print("2. PERFORMANCE TRAJECTORY OVER TIME")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:

        # get concept boundaries in meta-window index space
        boundaries_meta = get_stream_boundaries(
            drift_type, n_drifts, concept_sigmoid_spacing)

        # for gradual: only mark main drift points (every 4 boundaries)
        if drift_type == 'gradual':
            main_boundaries = boundaries_meta[::4]
        else:
            main_boundaries = boundaries_meta

        random_baseline = 1 / n_concepts

        for mf_type, mf_label, mf_names, _ in MF_CONFIGS:
            path = os.path.join(RESULTS_DIR,
                f'clf_ba_{mf_type}_{drift_type}.npy')
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue

            # shape: (n_replications, n_windows, n_clfs)
            traj = np.load(path)
            n_windows = traj.shape[1]
            x_axis    = np.arange(n_windows)

            fig, ax = plt.subplots(figsize=(14, 4))

            for clf_id, name in enumerate(clf_names):
                mean_traj = np.mean(traj[:, :, clf_id], axis=0)
                std_traj  = np.std(traj[:, :, clf_id],  axis=0)
                color     = CLF_COLORS.get(name, f'C{clf_id}')
                ax.plot(x_axis, mean_traj, label=name, color=color, linewidth=1.5)
                ax.fill_between(x_axis,
                    mean_traj - std_traj,
                    mean_traj + std_traj,
                    alpha=0.15, color=color)

            # concept boundaries
            for b in main_boundaries:
                ax.axvline(x=b, color='grey', linestyle='--',
                    linewidth=0.8, alpha=0.6)

            ax.axhline(y=random_baseline, color='red', linestyle='--',
                linewidth=1.0, label='random baseline')
            ax.set_xlabel('Window')
            ax.set_ylabel('Cumulative balanced accuracy')
            ax.set_title(f'Performance trajectory - {mf_label} - '
                f'{drift_type} drift ({n_concepts} concepts) - experiment [1c]')
            ax.legend(fontsize=9, ncol=3)
            ax.set_xlim(0, n_windows)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR,
                f'performance_{mf_type}_{drift_type}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"Performance plot saved at: '{fname}'")


# ============================================================
#  3. SHAP -- all 4 classifiers, all 3 ABFS versions
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("3. SHAP ANALYSIS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for mf_type, mf_label, mf_names, _ in MF_CONFIGS:
            fname = os.path.join(FIGURES_DIR,
                f'shap_all_clfs_{mf_type}_{drift_type}.png')
            if os.path.exists(fname):
                print(f"Exists: {fname}"); continue

            print(f"\nSHAP: {mf_label} - {drift_type} drift")

            all_X, all_y = [], []
            for rep_id, rs in enumerate(RANDOM_STATES):
                _, _, _, mf_results = extract_stream_data(
                    rs, drift_type, n_drifts, concept_sigmoid_spacing)
                all_X.append(mf_results[mf_type]['X'])
                all_y.append(mf_results[mf_type]['y'])

            X_all = np.vstack(all_X)
            y_all = np.concatenate(all_y)
            X_all[np.isnan(X_all)] = 1
            X_all[np.isinf(X_all)] = 1

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes_flat = axes.flatten()

            for clf_idx, (clf_name, clf_proto) in enumerate(SHAP_CLFS):
                ax  = axes_flat[clf_idx]
                clf = clone(clf_proto)
                clf.fit(X_all, y_all)

                explainer   = shap.KernelExplainer(
                    clf.predict_proba, shap.sample(X_all, 100))
                shap_values = explainer.shap_values(
                    shap.sample(X_all, 200), nsamples=100)

                shap_array    = np.array(shap_values)
                mean_abs_shap = (np.mean(np.abs(shap_array), axis=(0, 2))
                                 if shap_array.ndim == 3
                                 else np.mean(np.abs(shap_array), axis=0))

                sorted_idx = np.argsort(mean_abs_shap)[::-1]
                ax.bar(range(len(mf_names)), mean_abs_shap[sorted_idx],
                       color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(mf_names)))
                ax.set_xticklabels([mf_names[i] for i in sorted_idx],
                                   rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('Mean |SHAP|', fontsize=9)
                ax.set_title(clf_name, fontsize=11)

            fig.suptitle(f'SHAP - {mf_label} - {drift_type} drift - '
                         f'experiment [1c]\n(averaged over '
                         f'{N_REPLICATIONS} replications)', fontsize=12)
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"SHAP plot saved: {fname}")


# ============================================================
#  4. ADDITIONAL METRICS HEATMAPS (F1, KAPPA)
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60)
    print("4. ADDITIONAL METRICS HEATMAPS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for metric, metric_label in [('f1', 'F1'), ('kappa', 'Kappa')]:

            # Komorniczak baseline: statistical measure group, final window
            rc_path = os.path.join(RESULTS_DIR,
                f'clf_komor_concept_classif_{metric}_{drift_type}.npy')
            if os.path.exists(rc_path):
                rc_raw = np.load(rc_path)
                # shape: (n_measures, n_replications, n_windows, n_clfs)
                # take final window value per replication
                rc_final  = rc_raw[STATISTICAL_IDX, :, -1, :]  # (n_rep, n_clfs)
                rc_mean   = np.mean(rc_final,   axis=0)
                rc_std    = np.std(rc_final,    axis=0)
                rc_median = np.median(rc_final, axis=0)
            else:
                rc_mean = None
                print(f"Warning: {rc_path} not found")

            all_rows = []
            for mf_type, mf_display_label in ABFS_MF_CONFIGS_FULL:
                path = os.path.join(RESULTS_DIR,
                    f'clf_{metric}_{mf_type}_{drift_type}.npy')
                if not os.path.exists(path):
                    print(f"Warning: {path} not found, skipping.")
                    continue
                raw = np.load(path)
                # shape: (n_replications, n_windows, n_clfs)
                # take final window value per replication
                final       = raw[:, -1, :]  # (n_replications, n_clfs)
                mean_vals   = np.mean(final,   axis=0)
                std_vals    = np.std(final,    axis=0)
                median_vals = np.median(final, axis=0)
                all_rows.append((mf_display_label, mean_vals, std_vals, median_vals))

            if rc_mean is not None:
                all_rows.append(
                    ('Komorniczak (statistical)', rc_mean, rc_std, rc_median))

            if not all_rows:
                continue

            matrix        = np.array([r[1] for r in all_rows])
            matrix_std    = np.array([r[2] for r in all_rows])
            matrix_median = np.array([r[3] for r in all_rows])
            row_labels    = [r[0] for r in all_rows]

            # print summary
            print(f"\n{metric_label} - {drift_type} drift - experiment [1c]")
            print(f"{'Meta-features':<25s}", end='')
            for name in clf_names:
                print(f"{name:>10s}", end='')
            print()
            print(f"{'-' * (25 + 10 * len(clf_names))}")
            for label, mean_vals, _, __ in all_rows:
                print(f"{label:<25s}", end='')
                for v in mean_vals:
                    print(f"{v:>10.3f}", end='')
                print()

            fig, ax = plt.subplots(
                figsize=(10, max(3, len(all_rows) * 0.9)))
            im = ax.imshow(matrix, vmin=0.0, vmax=1.0,
                cmap='Blues', aspect='auto')
            for i in range(len(all_rows)):
                for j in range(len(clf_names)):
                    val    = matrix[i, j]
                    std    = matrix_std[i, j]
                    median = matrix_median[i, j]
                    txt_color = 'white' if val > 0.6 else 'black'
                    ax.text(j, i,
                        f'{val:.3f}\n(+/-{std:.3f})\nmed:{median:.3f}',
                        ha='center', va='center', fontsize=7,
                        color=txt_color, linespacing=1.4)
            ax.set_xticks(range(len(clf_names)))
            ax.set_xticklabels(clf_names, fontsize=9)
            ax.set_yticks(range(len(all_rows)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_title(f'{metric_label} - {drift_type} drift '
                f'({n_concepts} concepts) - experiment [1c]', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR,
                f'heatmap_{metric}_{drift_type}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"Heatmap saved at: '{fname}'")


# ============================================================
#  5. STREAM ANALYSIS PLOTS
# ============================================================
if RUN_STREAM_ANALYSIS:
    print("\n" + "="*60)
    print("5. STREAM ANALYSIS PLOTS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for rep_id, rs in enumerate(RANDOM_STATES):
            print(f"\nDrift: {drift_type} | Rep {rep_id+1}/{N_REPLICATIONS} (seed={rs})")

            class_dist, drift_intensity, entropy_vals, delta_relevance, boundaries = \
                extract_stream_diagnostics(rs, drift_type, n_drifts, concept_sigmoid_spacing)

            drift_intensity_n = drift_intensity / (np.max(drift_intensity) + 1e-10)
            delta_relevance_n = delta_relevance / (np.max(delta_relevance) + 1e-10)

            # ---- drift intensity vs ABFS relevance change vs entropy ----
            fname = os.path.join(FIGURES_DIR,
                f'stream_drift_entropy_{drift_type}_rep{rep_id}.png')
            if not os.path.exists(fname):
                fig, ax1 = plt.subplots(figsize=(14, 4))

                ax1.plot(drift_intensity_n, color='steelblue',
                        label='Drift intensity', linewidth=1.5)
                ax1.plot(delta_relevance_n, color='purple',
                        label='ABFS relevance change', linewidth=1.2, alpha=0.7)
                ax1.set_ylabel('Normalized value')

                ax2 = ax1.twinx()
                ax2.plot(entropy_vals, color='darkorange',
                        label='Label entropy', alpha=0.7)
                ax2.set_ylabel('Entropy', color='darkorange')

                for b in boundaries:
                    ax1.axvline(x=b, color='red', linestyle='--',
                                linewidth=0.8, alpha=0.7)

                ax1.set_xlabel('Chunk')

                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

                ax1.set_title(f'Drift vs ABFS dynamics - {drift_type} drift '
                    f'(seed={rs}) - experiment [1c]')

                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Stream drift/entropy plot saved at: '{fname}'")
            else:
                print(f"Exists: {fname}")

            # ---- class distribution over time ----
            fname = os.path.join(FIGURES_DIR,
                f'class_distribution_{drift_type}_rep{rep_id}.png')
            if not os.path.exists(fname):
                fig, ax = plt.subplots(figsize=(14, 4))
                for c in range(class_dist.shape[1]):
                    ax.plot(class_dist[:, c], label=f'class {c}', linewidth=1.2)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--',
                               linewidth=0.7, alpha=0.6)
                ax.set_xlabel('Chunk')
                ax.set_ylabel('Proportion')
                ax.set_title(f'Class distribution over time - {drift_type} drift '
                    f'(seed={rs}) - experiment [1c]')
                ax.legend(ncol=4, fontsize=8)
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Class distribution plot saved at: '{fname}'")
            else:
                print(f"Exists: {fname}")


# ============================================================
#  6. GAP HEATMAP -- ABFS raw v2.0 vs Komorniczak best-of-9,
#  one file per drift type (no chunk_size/n_informative grid here
#  the way Exp2 has, so drift type is the unit instead).
# ============================================================
if RUN_GAP:
    print("\n" + "="*60)
    print("6. GAP HEATMAP")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        komor_path = os.path.join(RESULTS_DIR,
            f'clf_komor_concept_classif_ba_{drift_type}.npy')
        if not os.path.exists(komor_path):
            print(f"Warning: {komor_path} not found, skipping {drift_type}."); continue
        komor_final        = np.load(komor_path)[:, :, -1, :]
        komor_mean_per_clf = np.mean(komor_final, axis=1)
        komor_best         = np.max(komor_mean_per_clf, axis=0)   # (n_clfs,)

        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR, f'gap_heatmap_preq_{version}_{drift_type}.png')
            if os.path.exists(fname):
                print(f"Exists: {fname}"); continue
            abfs_path = os.path.join(RESULTS_DIR, f'clf_ba_{version}_{drift_type}.npy')
            if not os.path.exists(abfs_path):
                print(f"Warning: {abfs_path} not found, skipping {version}/{drift_type}."); continue
            abfs_mean = np.mean(np.load(abfs_path)[:, -1, :], axis=0)  # (n_clfs,)
            gap_row   = abfs_mean - komor_best
            vmax      = np.max(np.abs(gap_row)) if np.any(~np.isnan(gap_row)) else 1.0

            fig, ax = plt.subplots(figsize=(max(6, len(clf_names) * 1.8), 2.8))
            im = ax.imshow(gap_row.reshape(1, -1), vmin=-vmax, vmax=vmax, cmap='RdBu', aspect='auto')
            for j in range(len(clf_names)):
                val = gap_row[j]
                ax.text(j, 0, f'{val:+.3f}', ha='center', va='center', fontsize=12,
                        color='white' if abs(val) > vmax * 0.6 else 'black')
            ax.set_xticks(range(len(clf_names))); ax.set_xticklabels(clf_names, fontsize=11)
            ax.set_yticks([0]); ax.set_yticklabels([f'{drift_type} drift'], fontsize=10)
            ax.set_xlabel('Classifier', fontsize=11)
            ax.set_title(f'Gap ({ABFS_LABELS[version]} minus Komorniczak best) -- '
                         f'{drift_type} drift -- experiment [1c]', fontsize=11)
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=6)
            cbar.set_label('Gap (BA)', fontsize=9)
            fig.subplots_adjust(bottom=0.35)
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"Gap heatmap saved at: '{fname}'")
            print(f"Gap heatmap saved at: '{fname}'")



if args.summary:
    print("\n" + "="*60); print("SUMMARY TABLE"); print("="*60)
    rows = []
    for drift_type, n_drifts, css, n_concepts in DRIFT_CONFIGS:
        # Komorniczak best-of-9 (mean over reps, final window)
        kpath = os.path.join(RESULTS_DIR, f'clf_komor_concept_classif_ba_{drift_type}.npy')
        if not os.path.exists(kpath):
            continue
        kfin = np.mean(np.load(kpath)[:, :, -1, :], axis=1)   # (n_measures, n_clfs)
        ki, kj = np.unravel_index(np.nanargmax(kfin), kfin.shape)
        k_label, k_clf, k_ba = MEASURES[ki], clf_names[kj], float(kfin[ki, kj])
        # best ABFS over available versions (mean over reps, final window)
        best = (None, None, -1.0)
        for version, vlabel in ABFS_MF_CONFIGS_FULL:
            apath = os.path.join(RESULTS_DIR, f'clf_ba_{version}_{drift_type}.npy')
            if not os.path.exists(apath):
                continue
            af = np.mean(np.load(apath)[:, -1, :], axis=0)    # (n_clfs,)
            j = int(np.nanargmax(af))
            if af[j] > best[2]:
                best = (vlabel, clf_names[j], float(af[j]))
        if best[0] is None:
            continue
        rb = 1.0 / n_concepts
        rows.append([drift_type, N_FEATURES, n_concepts, f'{rb:.3f}',
                     f'{k_label} / {k_clf}', f'{k_ba:.3f}',
                     f'{best[0]} / {best[1]}', f'{best[2]:.3f}',
                     f'{best[2]-k_ba:+.3f}'])
    header = ['drift', 'n_feat', 'n_conc', 'baseline',
              'best Komor (grp/clf)', 'Komor BA',
              'best ABFS (ver/clf)', 'ABFS BA', 'gap']
    write_summary_csv(os.path.join(RESULTS_DIR, 'summary_exp1c.csv'),
                      'Experiment 1c summary', header, rows)