# analysis_2.py
# ============================================================
# Analysis of Experiment 2 results (stream configuration sensitivity).
#
# Usage:
#   python experiments/experiment_2/analysis_2.py [--sanity] [--performance] [--shap] [--metrics] [--grid] [--stream_analysis]
#
# Flags:
#   --sanity      : relevance scores, meta-features per version, PCA per version (rep 0 only)
#   --performance : cumulative BA trajectory over windows (prequential)
#   --shap        : SHAP -- all 4 classifiers, all 3 ABFS versions, per cell
#   --metrics     : F1 and Kappa heatmaps per cell (prequential only)
#   --grid        : gap heatmap + sensitivity curves across 4x4 grid
#   --stream_analysis : feature-mean drift intensity, ABFS relevance-score
#                   change, label entropy, and class distribution over
#                   time (rep 0 only) -- same diagnostics as Experiment
#                   3's --stream_analysis, computed directly from a
#                   freshly-generated stream rather than loaded from a
#                   saved npy, since these synthetic streams aren't
#                   persisted to disk the way the real streams are.
#
# SHAP classifiers (sklearn-compatible proxies):
#   GNB -- GaussianNB
#   KNN -- KNeighborsClassifier
#   HT  -- DecisionTreeClassifier
#   MLP -- MLPClassifier

# Inputs (from results/experiment_2/):
#   preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy  shape: (n_reps, n_windows, n_clfs)
#   preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy shape: (n_reps, n_windows, n_clfs)
#   (same pattern for f1, kappa)
#
# Outputs saved to results/experiment_2/figures/analysis/
# ============================================================


import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone as skclone
from scipy.stats import entropy
import shap
import os
import sys
import warnings
import csv

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
    MF_NAMES_AGGSTATS,
)
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
from streams.generate_synthetic_streams import (
    assign_labels_gradual,
    get_exp2_concept_labels,
    EXP2_N_CHUNKS       as N_CHUNKS,
    EXP2_N_FEATURES     as N_FEATURES,
    EXP2_CHUNK_SIZES    as CHUNK_SIZES,
    EXP2_N_INFORMATIVES as N_INFORMATIVES,
    EXP2_DRIFT_CONFIGS  as DRIFT_CONFIGS,
)


# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description='Analysis for Experiment 2.')
parser.add_argument('--sanity',      action='store_true')
parser.add_argument('--performance', action='store_true')
parser.add_argument('--shap',        action='store_true')
parser.add_argument('--metrics',     action='store_true')
parser.add_argument('--grid',        action='store_true')
parser.add_argument('--stream_analysis', action='store_true')
parser.add_argument('--vanilla', action='store_true')
parser.add_argument('--summary', action='store_true')
parser.add_argument('--concept_dist_features', action='store_true')
args = parser.parse_args()

RUN_SANITY      = args.sanity
RUN_PERFORMANCE = args.performance
RUN_SHAP        = args.shap
RUN_METRICS     = args.metrics
RUN_GRID        = args.grid
RUN_STREAM_ANALYSIS = args.stream_analysis

print(f"\nRunning analysis for Experiment 2 (synthetic stream configuration sensitivity):")
print(f"Sanity:          {RUN_SANITY}")
print(f"Performance:     {RUN_PERFORMANCE}")
print(f"SHAP:            {RUN_SHAP}")
print(f"Metrics:         {RUN_METRICS}")
print(f"Grid:            {RUN_GRID}")
print(f"Stream analysis: {RUN_STREAM_ANALYSIS}")
print(f"Summary:         {args.summary}")


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================
WARMUP_WINDOWS        = 10
SCORE_INTERVAL        = 100
N_REPLICATIONS        = 5
CHUNK_SIZE_DEFAULT    = 200
N_INFORMATIVE_DEFAULT = 10

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

MEASURES = [
    'clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical',
]

ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {
    'aggstats':     'Aggstats',
    'raw':          'Raw scores',
    'raw_temporal': 'Raw + temporal',
}

clf_names_preq = [name for name, _ in BASE_CLFS_PREQUENTIAL]
N_CLFS         = len(clf_names_preq)

CLF_COLORS = {
    'GNB': '#e6194b', 'KNN': '#3cb44b',
    'HT':  '#f032e6', 'MLP': '#911eb4',
}

PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', '#5e5151', '#08332b',
    '#000000', '#a9a9a9', '#ff69b4', '#00ced1', '#ff8c00',
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

def make_tag(chunk_size, n_informative, drift_type):
    return f"chunk{chunk_size}_ninf{n_informative}_{drift_type}"


def load(prefix, tag, optional=False):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{tag}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)


def load_komor_best(tag):
    """Return mean final-window BA for the best Komorniczak measure group per clf."""
    best = None
    for measure in MEASURES:
        arr = load(f'preq_komor_{measure}_ba', tag, optional=True)
        if arr is None:
            continue
        per_clf = np.mean(arr[:, -1, :], axis=0)
        if best is None or np.max(per_clf) > np.max(best):
            best = per_clf
    return best


def feat_names_for(version, n_features):
    """Return feature names for a given ABFS version."""
    if version == 'aggstats':
        return MF_NAMES_AGGSTATS
    elif version == 'raw':
        return [f'r_f{j+1}' for j in range(n_features)]
    else:  # raw_temporal
        return [f'r_f{j+1}' for j in range(n_features)] + ['delta_mean', 'cosine_sim']


def get_concept_boundaries(concept_labels_all, n_chunks):
    return [i for i in range(1, n_chunks)
            if concept_labels_all[i] != concept_labels_all[i-1]]


def extract_stream_data_all_versions(rs, drift_type, n_drifts,
                                     concept_sigmoid_spacing,
                                     chunk_size, n_informative):
    """
    Single-pass extraction of all 3 ABFS versions + relevance scores.

    Returns
    -------
    scores_over_time : np.ndarray (n_score_samples, N_FEATURES)
    concept_labels_all : np.ndarray (N_CHUNKS,)
    boundaries : list of int
    X_by_version : dict {version: np.ndarray (n_windows, dim)}
    y : np.ndarray (n_windows,)
    """
    config = dict(
        n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=chunk_size,
        n_features=N_FEATURES, n_informative=n_informative,
        n_redundant=0, n_repeated=0,
        concept_sigmoid_spacing=concept_sigmoid_spacing,
        random_state=rs,
    )
    stream = StreamGenerator(**config)

    # pass 1: relevance scores over time + concept labels
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size,
                      class_window_size=chunk_size)
    scores_over_time, instance_counter = [], 0
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
            if instance_counter % SCORE_INTERVAL == 0:
                scores_over_time.append(abfs.relevance_scores())
            instance_counter += 1
    scores_over_time = np.array(scores_over_time)
    concept_labels_all = get_exp2_concept_labels(
        stream, drift_type, N_CHUNKS, chunk_size)

    boundaries = get_concept_boundaries(concept_labels_all, N_CHUNKS)

    # pass 2: all 3 ABFS versions
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size,
                      class_window_size=chunk_size)
    mf = {'aggstats': [], 'raw': [], 'raw_temporal': []}
    concept_labels = []
    wt_prev, window_counter = None, 0

    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()
        t_since     = abfs.time_since_drift

        if window_counter >= WARMUP_WINDOWS:
            mf['aggstats'].append(
                extract_metafeatures(wt, wt_prev, drift_count, t_since))
            mf['raw'].append(extract_metafeatures_raw(wt))
            mf['raw_temporal'].append(
                extract_metafeatures_raw_temporal(wt, wt_prev))
            concept_labels.append(concept_labels_all[window_counter])

        wt_prev = wt
        window_counter += 1

    def clean(lst):
        a = np.array(lst, dtype=float)
        a[np.isnan(a)] = 1; a[np.isinf(a)] = 1
        return a

    X_by_version = {v: clean(mf[v]) for v in ABFS_VERSIONS}
    y = np.array(concept_labels)
    return scores_over_time, concept_labels_all, boundaries, X_by_version, y


def get_stream_boundaries_meta(drift_type, n_drifts, concept_sigmoid_spacing,
                               chunk_size, n_informative):
    _, concept_labels_all, boundaries, _, _ = extract_stream_data_all_versions(
        RANDOM_STATES[0], drift_type, n_drifts, concept_sigmoid_spacing,
        chunk_size, n_informative)
    return [b - WARMUP_WINDOWS for b in boundaries if b - WARMUP_WINDOWS > 0]


def extract_stream_diagnostics(rs, drift_type, n_drifts, concept_sigmoid_spacing,
                               chunk_size, n_informative):
    """
    One pass over the stream computing PER-CHUNK diagnostics: class
    distribution (of the underlying binary classification target that
    StreamGenerator produces -- NOT the concept label), feature-mean
    drift intensity, label entropy, and ABFS relevance-score change
    (delta_relevance) -- all on the same per-chunk index, with no
    WARMUP_WINDOWS shift (unlike extract_stream_data_all_versions's
    pass 2), since this diagnostic covers the whole stream from chunk
    0 rather than feeding the meta-feature classifiers.

    Mirrors generate_real_streams.py's run_stream_analysis(), adapted
    for a freshly-generated (not saved-to-disk) synthetic stream.
    Concept boundaries are obtained via get_exp2_concept_labels, the
    same single source of truth used everywhere else in this file, so
    boundary positions stay consistent with --sanity/--grid for the
    same tag.
    """
    config = dict(
        n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=chunk_size,
        n_features=N_FEATURES, n_informative=n_informative,
        n_redundant=0, n_repeated=0,
        concept_sigmoid_spacing=concept_sigmoid_spacing,
        random_state=rs,
    )
    stream = StreamGenerator(**config)
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size,
                      class_window_size=chunk_size)

    class_distribution, drift_intensity, label_entropy_vals = [], [], []
    delta_relevance = []
    prev_mean, wt_prev = None, None
    # underlying StreamGenerator target is binary by default in this
    # project's Exp2 config (no n_classes override is set anywhere
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
        # relevance_scores() returns a plain list, not an ndarray --
        # convert before subtracting, or list - list raises TypeError
        wt = np.array(abfs.relevance_scores())
        if wt_prev is None:
            delta_relevance.append(0.0)
        else:
            delta_relevance.append(np.linalg.norm(wt - wt_prev))
        wt_prev = wt

    concept_labels_all = get_exp2_concept_labels(
        stream, drift_type, N_CHUNKS, chunk_size)
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
            best = (label, clf_names_preq[j], float(v[j]))
    return best


def write_summary_csv(path, header, rows):
    """Write CSV in spreadsheet-friendly European format:
    - semicolon as separator
    - comma as decimal separator
    """
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}".replace(".", ",")
        return v

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow([fmt(v) for v in row])

    print(f"  Saved: {path}")



def write_dict_csv(path, rows):
    """Write dict rows as spreadsheet-friendly CSV:
    semicolon separator and comma decimal separator.
    """
    if not rows:
        return

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}".replace(".", ",")
        return v

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        for r in rows:
            writer.writerow([fmt(r.get(k, "")) for k in fields])

    print(f"  Saved: {path}")


# ============================================================
#  1. SANITY CHECK PLOTS -- all 3 ABFS versions
# ============================================================
if RUN_SANITY:
    print("\n" + "="*60)
    print("1. SANITY CHECK PLOTS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
                print(f"\n  {tag}")

                rs = RANDOM_STATES[0]
                scores_over_time, concept_labels_all, boundaries, \
                    X_by_version, y = extract_stream_data_all_versions(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)
                unique_concepts = np.unique(y)

                # relevance scores -- one per cell
                fname = os.path.join(FIGURES_DIR,
                                     f'relevance_scores_{tag}_rep0.png')
                if not os.path.exists(fname):
                    fig, ax = plt.subplots(figsize=(14, 4))
                    for j in range(N_FEATURES):
                        ax.plot(scores_over_time[:, j],
                                label=f'f{j+1}', linewidth=0.8)
                    for b in boundaries:
                        ax.axvline(x=b * chunk_size // SCORE_INTERVAL,
                                   color='red', linestyle='--',
                                   linewidth=0.8, alpha=0.6)
                    ax.axvline(x=-1, color='red', linestyle='--',
                               linewidth=0.8, label='concept boundary')
                    ax.set_xlabel('Time (x100 instances)')
                    ax.set_ylabel('Relevance score')
                    ax.set_title(f'Relevance scores -- {tag} -- rep0')
                    ax.legend(ncol=5, fontsize=7)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    plt.close(); print(f"  Saved: {fname}")
                else:
                    print(f"  Exists: {fname}")

                # meta-features and PCA -- one per version
                for version in ABFS_VERSIONS:
                    X     = X_by_version[version]
                    names = feat_names_for(version, N_FEATURES)
                    n_f   = len(names)

                    fname = os.path.join(FIGURES_DIR,
                                         f'metafeatures_{version}_{tag}_rep0.png')
                    if not os.path.exists(fname):
                        n_cols = 5
                        n_rows = (n_f + n_cols - 1) // n_cols
                        fig, axes = plt.subplots(n_rows, n_cols,
                                                 figsize=(4*n_cols, 3*n_rows))
                        axes_flat = axes.flatten()
                        for k in range(n_f):
                            axes_flat[k].plot(X[:, k], color='steelblue',
                                              linewidth=0.8)
                            for b in boundaries:
                                drift_w = b - WARMUP_WINDOWS
                                if drift_w > 0:
                                    axes_flat[k].axvline(
                                        x=drift_w, color='red',
                                        linestyle='--', linewidth=0.8)
                            axes_flat[k].set_title(names[k], fontsize=8)
                            axes_flat[k].set_xlabel('Window', fontsize=7)
                        for k in range(n_f, len(axes_flat)):
                            axes_flat[k].set_visible(False)
                        fig.suptitle(
                            f'Meta-features ({ABFS_LABELS[version]}) -- '
                            f'{tag} -- rep0', fontsize=10)
                        fig.tight_layout()
                        fig.savefig(fname, dpi=150, bbox_inches='tight')
                        plt.close(); print(f"  Saved: {fname}")
                    else:
                        print(f"  Exists: {fname}")

                    fname = os.path.join(FIGURES_DIR,
                                         f'pca_{version}_{tag}_rep0.png')
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
                        ax.set_title(
                            f'PCA -- {ABFS_LABELS[version]} -- {tag} -- rep0')
                        ax.legend(ncol=4, fontsize=7)
                        fig.tight_layout()
                        fig.savefig(fname, dpi=150, bbox_inches='tight')
                        plt.close(); print(f"  Saved: {fname}")
                    else:
                        print(f"  Exists: {fname}")


# ============================================================
#  2. PERFORMANCE TRAJECTORY (PREQUENTIAL)
# ============================================================
if RUN_PERFORMANCE:
    print("\n" + "="*60)
    print("2. PERFORMANCE TRAJECTORY (PREQUENTIAL)")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)

                boundaries_meta = get_stream_boundaries_meta(
                    drift_type, n_drifts, concept_sigmoid_spacing,
                    chunk_size, n_informative)
                main_boundaries = (boundaries_meta[::4]
                                   if drift_type == 'gradual'
                                   else boundaries_meta)
                random_baseline = 1 / n_concepts

                # all 3 ABFS versions + Komorniczak statistical representative
                sources = [(f'preq_abfs_{v}_ba', ABFS_LABELS[v])
                           for v in ABFS_VERSIONS]
                sources += [('preq_komor_statistical_ba',
                             'Komorniczak (statistical)')]

                for prefix, label in sources:
                    short = prefix.replace('preq_', '').replace('_ba', '')
                    fname = os.path.join(FIGURES_DIR,
                                         f'trajectory_{short}_{tag}.png')
                    if os.path.exists(fname):
                        print(f"  Exists: {fname}"); continue

                    data = load(prefix, tag, optional=True)
                    if data is None:
                        continue
                    n_windows = data.shape[1]
                    x_axis    = np.arange(n_windows)

                    fig, ax = plt.subplots(figsize=(14, 4))
                    for clf_id, name in enumerate(clf_names_preq):
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
                    ax.axhline(y=random_baseline, color='red', linestyle='--',
                               linewidth=1.0, label='random baseline')
                    ax.set_xlabel('Window')
                    ax.set_ylabel('Cumulative balanced accuracy')
                    ax.set_title(f'Performance trajectory -- {label} -- {tag}')
                    ax.legend(fontsize=9, ncol=3)
                    ax.set_xlim(0, n_windows)
                    ax.set_ylim(0, 1)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  3. SHAP -- all 4 classifiers, all 3 ABFS versions
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("3. SHAP ANALYSIS - all 4 classifiers, all 3 ABFS versions")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)

                # check if all 3 version figures already exist
                all_done = all(
                    os.path.exists(os.path.join(
                        FIGURES_DIR, f'shap_all_clfs_{v}_{tag}.png'))
                    for v in ABFS_VERSIONS
                )
                if all_done:
                    print(f"  All SHAP figures exist for {tag} -- skipping.")
                    continue

                print(f"\n  SHAP: {tag}")

                # collect X_by_version across all replications
                all_X_by_version = {v: [] for v in ABFS_VERSIONS}
                all_y = []
                for rs in RANDOM_STATES:
                    _, _, _, X_by_version, y = extract_stream_data_all_versions(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)
                    for v in ABFS_VERSIONS:
                        all_X_by_version[v].append(X_by_version[v])
                    all_y.append(y)

                y_all = np.concatenate(all_y)

                for version in ABFS_VERSIONS:
                    fname = os.path.join(FIGURES_DIR,
                                         f'shap_all_clfs_{version}_{tag}.png')
                    if os.path.exists(fname):
                        print(f"  Exists: {fname}"); continue

                    X_all = np.vstack(all_X_by_version[version])
                    X_all[np.isnan(X_all)] = 1; X_all[np.isinf(X_all)] = 1
                    feat_names = feat_names_for(version, N_FEATURES)

                    print(f"  [{version}] X={X_all.shape}  y={y_all.shape}")

                    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                    axes_flat = axes.flatten()

                    for clf_idx, (clf_name, clf_proto) in enumerate(SHAP_CLFS):
                        ax  = axes_flat[clf_idx]
                        clf = skclone(clf_proto)
                        clf.fit(X_all, y_all)

                        explainer   = shap.KernelExplainer(
                            clf.predict_proba, shap.sample(X_all, 50))
                        shap_values = explainer.shap_values(
                            shap.sample(X_all, 100), nsamples=50)

                        shap_array    = np.array(shap_values)
                        mean_abs_shap = (
                            np.mean(np.abs(shap_array), axis=(0, 2))
                            if shap_array.ndim == 3
                            else np.mean(np.abs(shap_array), axis=0))

                        sorted_idx = np.argsort(mean_abs_shap)[::-1]
                        ax.bar(range(len(feat_names)),
                               mean_abs_shap[sorted_idx],
                               color='steelblue', alpha=0.8)
                        ax.set_xticks(range(len(feat_names)))
                        ax.set_xticklabels(
                            [feat_names[i] for i in sorted_idx],
                            rotation=45, ha='right', fontsize=7)
                        ax.set_ylabel('Mean |SHAP|', fontsize=9)
                        ax.set_title(clf_name, fontsize=11)

                    fig.suptitle(
                        f'SHAP -- {ABFS_LABELS[version]} -- {tag}\n'
                        f'({N_REPLICATIONS} replications combined)',
                        fontsize=12)
                    plt.tight_layout()
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  4. METRICS HEATMAPS (F1, KAPPA)
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60)
    print("4. METRICS HEATMAPS (F1, KAPPA)")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)

                for metric, metric_label in [('f1', 'F1'), ('kappa', 'Kappa')]:
                    fname = os.path.join(FIGURES_DIR,
                                         f'heatmap_{metric}_preq_{tag}.png')
                    if os.path.exists(fname):
                        print(f"  Exists: {fname}"); continue

                    # Komorniczak: all 9 measures
                    komor_rows = []
                    for measure in MEASURES:
                        data = load(f'preq_komor_{measure}_{metric}',
                                    tag, optional=True)
                        if data is not None:
                            final = data[:, -1, :]
                            komor_rows.append((measure,
                                               np.mean(final, axis=0),
                                               np.std(final, axis=0)))

                    # ABFS: all 3 versions
                    abfs_rows = []
                    for version in ABFS_VERSIONS:
                        data = load(f'preq_abfs_{version}_{metric}',
                                    tag, optional=True)
                        if data is not None:
                            final = data[:, -1, :]
                            abfs_rows.append((ABFS_LABELS[version],
                                              np.mean(final, axis=0),
                                              np.std(final, axis=0)))

                    if not komor_rows or not abfs_rows:
                        continue

                    komor_m = np.array([r[1] for r in komor_rows])
                    komor_s = np.array([r[2] for r in komor_rows])
                    abfs_m  = np.array([r[1] for r in abfs_rows])
                    abfs_s  = np.array([r[2] for r in abfs_rows])

                    fig, axes = plt.subplots(
                        1, 2, figsize=(22, max(5, len(komor_rows) * 0.75)),
                        gridspec_kw={'width_ratios': [3, 1.5]})

                    for ax, matrix, std_mat, row_labels, title in [
                        (axes[0], komor_m, komor_s,
                         [r[0] for r in komor_rows],
                         f'Komorniczak -- {metric_label}'),
                        (axes[1], abfs_m, abfs_s,
                         [r[0] for r in abfs_rows],
                         f'ABFS -- {metric_label}'),
                    ]:
                        im = ax.imshow(matrix, vmin=0.0, vmax=1.0,
                                       cmap='Blues', aspect='auto')
                        for i in range(len(row_labels)):
                            for j in range(N_CLFS):
                                val = matrix[i, j]; std = std_mat[i, j]
                                txt_color = 'white' if val > 0.6 else 'black'
                                ax.text(j, i,
                                        f'{val:.3f}\n(+/-{std:.3f})',
                                        ha='center', va='center',
                                        fontsize=9, color=txt_color,
                                        linespacing=1.4)
                        ax.set_xticks(range(N_CLFS))
                        ax.set_xticklabels(clf_names_preq, fontsize=10)
                        ax.set_yticks(range(len(row_labels)))
                        ax.set_yticklabels(row_labels, fontsize=9)
                        ax.set_title(title, fontsize=12)

                    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    fig.suptitle(
                        f'{metric_label} -- prequential -- {tag}', fontsize=12)
                    plt.tight_layout()
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  5. STREAM ANALYSIS PLOTS (rep 0 only, same convention as --sanity)
# ============================================================
if RUN_STREAM_ANALYSIS:
    print("\n" + "="*60)
    print("5. STREAM ANALYSIS PLOTS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
                print(f"\n  {tag}")

                rs = RANDOM_STATES[0]
                class_dist, drift_intensity, entropy_vals, delta_relevance, \
                    boundaries = extract_stream_diagnostics(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)

                drift_intensity_n = drift_intensity / (np.max(drift_intensity) + 1e-10)
                delta_relevance_n = delta_relevance / (np.max(delta_relevance) + 1e-10)

                # ---- drift intensity vs ABFS relevance change vs entropy ----
                fname = os.path.join(FIGURES_DIR,
                                     f'stream_drift_entropy_{tag}_rep0.png')
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

                    ax1.set_title(f'Drift vs ABFS dynamics -- {tag} -- rep0')

                    fig.tight_layout()
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    plt.close(); print(f"  Saved: {fname}")
                else:
                    print(f"  Exists: {fname}")

                # ---- class distribution over time ----
                fname = os.path.join(FIGURES_DIR,
                                     f'class_distribution_{tag}_rep0.png')
                if not os.path.exists(fname):
                    fig, ax = plt.subplots(figsize=(14, 4))
                    for c in range(class_dist.shape[1]):
                        ax.plot(class_dist[:, c], label=f'class {c}', linewidth=1.2)
                    for b in boundaries:
                        ax.axvline(x=b, color='grey', linestyle='--',
                                   linewidth=0.7, alpha=0.6)
                    ax.set_xlabel('Chunk')
                    ax.set_ylabel('Proportion')
                    ax.set_title(f'Class distribution over time -- {tag} -- rep0')
                    ax.legend(ncol=4, fontsize=8)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150, bbox_inches='tight')
                    plt.close(); print(f"  Saved: {fname}")
                else:
                    print(f"  Exists: {fname}")


# ============================================================
#  6+7. GRID: GAP HEATMAP + SENSITIVITY CURVES
# ============================================================
if RUN_GRID:
    print("\n" + "="*60)
    print("6+7. GAP HEATMAP + SENSITIVITY CURVES")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:

        grid_abfs_preq      = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)),
                                      np.nan)
        grid_komor_preq     = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)),
                                      np.nan)
        grid_abfs_preq_clf  = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES),
                                       N_CLFS), np.nan)
        grid_komor_preq_clf = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES),
                                       N_CLFS), np.nan)

        for i, chunk_size in enumerate(CHUNK_SIZES):
            for j, n_informative in enumerate(N_INFORMATIVES):
                tag = make_tag(chunk_size, n_informative, drift_type)

                # EMF raw vector
                pr_abfs = load('preq_abfs_raw_ba', tag, optional=True)
                if pr_abfs is not None:
                    per_clf = np.mean(pr_abfs[:, -1, :], axis=0)
                    grid_abfs_preq[i, j]     = np.max(per_clf)
                    grid_abfs_preq_clf[i, j] = per_clf

                # Komorniczak best of 9
                komor_best = load_komor_best(tag)
                if komor_best is not None:
                    grid_komor_preq[i, j]     = np.max(komor_best)
                    grid_komor_preq_clf[i, j] = komor_best

        x_labels = [str(ni) for ni in N_INFORMATIVES]
        y_labels  = [str(cs) for cs in CHUNK_SIZES]

        # ---- gap heatmaps, one per EMF version (shared color scale) ----
        version_grids = {}
        for version in EMF_VERSIONS:
            g = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
            for i, chunk_size in enumerate(CHUNK_SIZES):
                for j, n_informative in enumerate(N_INFORMATIVES):
                    tag = make_tag(chunk_size, n_informative, drift_type)
                    pr = load(f'preq_abfs_{version}_ba', tag, optional=True)
                    kb = load_komor_best(tag)
                    if pr is not None and kb is not None:
                        a = np.max(np.mean(pr[:, -1, :], axis=0))
                        g[i, j] = a - np.max(kb)
            version_grids[version] = g

        finite = np.concatenate([g[np.isfinite(g)] for g in version_grids.values()]) \
                 if any(np.any(np.isfinite(g)) for g in version_grids.values()) else np.array([])
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0

        for version in EMF_VERSIONS:
            gap_grid = version_grids[version]
            if not np.any(np.isfinite(gap_grid)):
                continue
            fname = os.path.join(FIGURES_DIR,
                                 f'gap_heatmap_preq_{version}_{drift_type}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(gap_grid, vmin=-vmax, vmax=vmax, cmap='RdBu', aspect='auto')
            for i in range(len(CHUNK_SIZES)):
                for j in range(len(N_INFORMATIVES)):
                    val = gap_grid[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                                fontsize=10, color='white' if abs(val) > vmax*0.6 else 'black')
                    else:
                        ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='grey')
            ax.set_xticks(range(len(N_INFORMATIVES))); ax.set_xticklabels(x_labels, fontsize=10)
            ax.set_yticks(range(len(CHUNK_SIZES))); ax.set_yticklabels(y_labels, fontsize=10)
            ax.set_xlabel('n_informative', fontsize=11); ax.set_ylabel('chunk_size', fontsize=11)
            ax.set_title(f'Gap (best EMF {EMF_LABELS[version]} minus Komorniczak best)\n'
                         f'{drift_type} drift ({n_concepts} concepts)', fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Gap heatmap saved: {fname}")

        # sensitivity curves
        ni_idx = N_INFORMATIVES.index(N_INFORMATIVE_DEFAULT)
        cs_idx = CHUNK_SIZES.index(CHUNK_SIZE_DEFAULT)

        # BA vs chunk_size (n_informative=10 fixed)
        fname = os.path.join(FIGURES_DIR,
            f'sensitivity_chunk_preq_ninf{N_INFORMATIVE_DEFAULT}_{drift_type}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(8, 4))
            for clf_id, name in enumerate(clf_names_preq):
                color = CLF_COLORS.get(name, f'C{clf_id}')
                ax.plot(CHUNK_SIZES, grid_abfs_preq_clf[:, ni_idx, clf_id],
                        color=color, label=f'{name} EMF',
                        linewidth=1.5, marker='o', markersize=5)
                ax.plot(CHUNK_SIZES, grid_komor_preq_clf[:, ni_idx, clf_id],
                        color=color, label=f'{name} Komor',
                        linewidth=1.5, linestyle='--', marker='s', markersize=5)
            ax.axhline(y=1/n_concepts, color='red', linestyle=':',
                       linewidth=1.0, label='random baseline')
            ax.set_xlabel('chunk_size', fontsize=11)
            ax.set_ylabel('Mean balanced accuracy', fontsize=10)
            ax.set_title(
                f'BA vs chunk_size (n_informative={N_INFORMATIVE_DEFAULT})\n'
                f'{drift_type} drift', fontsize=11)
            ax.legend(fontsize=8, ncol=2,
                      bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_xticks(CHUNK_SIZES)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Sensitivity (chunk) saved: {fname}")

        # BA vs n_informative (chunk_size=200 fixed)
        fname = os.path.join(FIGURES_DIR,
            f'sensitivity_ninf_preq_chunk{CHUNK_SIZE_DEFAULT}_{drift_type}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(8, 4))
            for clf_id, name in enumerate(clf_names_preq):
                color = CLF_COLORS.get(name, f'C{clf_id}')
                ax.plot(N_INFORMATIVES, grid_abfs_preq_clf[cs_idx, :, clf_id],
                        color=color, label=f'{name} ABFS',
                        linewidth=1.5, marker='o', markersize=5)
                ax.plot(N_INFORMATIVES, grid_komor_preq_clf[cs_idx, :, clf_id],
                        color=color, label=f'{name} Komor',
                        linewidth=1.5, linestyle='--', marker='s', markersize=5)
            ax.axhline(y=1/n_concepts, color='red', linestyle=':',
                       linewidth=1.0, label='random baseline')
            ax.set_xlabel('n_informative', fontsize=11)
            ax.set_ylabel('Mean balanced accuracy', fontsize=10)
            ax.set_title(
                f'BA vs n_informative (chunk_size={CHUNK_SIZE_DEFAULT})\n'
                f'{drift_type} drift', fontsize=11)
            ax.legend(fontsize=8, ncol=2,
                      bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_xticks(N_INFORMATIVES)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Sensitivity (ninf) saved: {fname}")



if args.vanilla:
    print("\n" + "="*60)
    print("VANILLA BASELINE COMPARISON")
    print("="*60)

    def best_final_ba(prefix, cell):
        """Best classifier's final-window BA, mean over reps. None if missing."""
        d = load(prefix, cell, optional=True)
        if d is None:
            return None
        return float(np.max(np.mean(d[:, -1, :], axis=0)))

    def vanilla_row(cell):
        v = best_final_ba('preq_vanilla_ba', cell)
        a = max([b for b in (best_final_ba(f'preq_abfs_{ver}_ba', cell)
                             for ver in ABFS_VERSIONS) if b is not None], default=None)
        k = max([b for b in (best_final_ba(f'preq_komor_{m}_ba', cell)
                             for m in MEASURES) if b is not None], default=None)
        return v, a, k

    # Exp 2 cells: built from the config lists, same as the evaluate script
    CELLS = [f'chunk{cs}_ninf{ni}_{drift}'
             for drift, _, _, _ in DRIFT_CONFIGS
             for cs in CHUNK_SIZES
             for ni in N_INFORMATIVES]

    rows = []
    for cell in CELLS:
        v, a, k = vanilla_row(cell)
        if v is None:
            print(f"  {cell}: no vanilla results -- skipping"); continue
        rows.append((cell, v, a, k))
        print(f"  {cell:35s}  vanilla={v:.3f}  abfs={a:.3f}  komor={k:.3f}")

    import csv
    rows_csv = []
    for cell, v, a, k in rows:
        rows_csv.append(dict(
            cell=cell,
            vanilla_ba=v,
            abfs_best_ba=a,
            komor_best_ba=k
        ))

    out = os.path.join(RESULTS_DIR, 'vanilla_comparison_exp2.csv')
    write_dict_csv(out, rows_csv)
    print(f"\n  Saved: {out}")


if args.summary:
    print("\n" + "="*60); print("SUMMARY TABLE"); print("="*60)
    rows = []
    for drift_type, n_drifts, css, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)
                loadf = lambda prefix, t=tag: load(prefix, t, optional=True)
                kb = best_side(loadf, [(m, f'preq_komor_{m}_ba') for m in MEASURES], has_reps=True)
                ab = best_side(loadf, [(ABFS_LABELS[v], f'preq_abfs_{v}_ba') for v in ABFS_VERSIONS], has_reps=True)
                if kb[0] is None or ab[0] is None:
                    continue
                rb = 1.0 / n_concepts
                rows.append([
                    drift_type,
                    chunk_size,
                    n_informative,
                    N_FEATURES,
                    n_concepts,
                    rb,
                    f'{kb[0]} / {kb[1]}',
                    kb[2],
                    f'{ab[0]} / {ab[1]}',
                    ab[2],
                    ab[2] - kb[2]
                ])
    header = ['drift', 'chunk', 'n_inform', 'n_feat', 'n_conc', 'baseline',
              'best Komor (grp/clf)', 'Komor BA',
              'best ABFS (ver/clf)', 'ABFS BA', 'gap']
    write_summary_csv(os.path.join(RESULTS_DIR, 'summary_exp2.csv'), header, rows)




if args.concept_dist_features:
    print("\n" + "=" * 60)
    print("CONCEPT SEPARATION IN FEATURE SPACE")
    print("=" * 60)

    # Concept separation is measured at one chunk size: chunk_size only
    # regroups the same instances into windows, so the per-concept mean is
    # invariant to it apart from boundary-straddling windows.
    CS = CHUNK_SIZE_DEFAULT          # 200, the Exp 2 baseline

    rows_means, rows_dist, rows_summary = [], [], []
    print(f"  chunk_size={CS}   seeds: {list(RANDOM_STATES)}")

    for drift_type, n_drifts, spacing, n_concepts in DRIFT_CONFIGS:
        for n_informative in N_INFORMATIVES:
            cell = f'ninf{n_informative}_{drift_type}'

            per_rep_pairs = {}   # (a, b) -> [l2 per rep]
            cm_rep0, uniq_rep0 = None, None

            for rep, seed in enumerate(RANDOM_STATES):
                cfg = dict(
                    n_drifts=n_drifts,
                    n_chunks=N_CHUNKS,
                    chunk_size=CS,
                    n_features=N_FEATURES,
                    n_informative=n_informative,
                    n_redundant=0,
                    n_repeated=0,
                    concept_sigmoid_spacing=spacing,
                    random_state=int(seed)
                )
                stream = StreamGenerator(**cfg)

                # StreamGenerator is lazy -- iterate FIRST so concept bookkeeping exists
                stream.reset()
                means = np.array([Xc.mean(axis=0) for Xc, yc in stream])

                # same label source as --sanity / --grid / --stream_analysis
                concept_labels_all = get_exp2_concept_labels(
                    stream, drift_type, N_CHUNKS, CS
                )

                concs = np.asarray(concept_labels_all)
                uniq = np.unique(concs)

                # mean feature vector for each concept in this replication
                cm = np.array([means[concs == c].mean(axis=0) for c in uniq])

                if rep == 0:
                    cm_rep0, uniq_rep0 = cm, uniq

                # collect pairwise distances for this replication
                d = []
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        l2 = float(np.linalg.norm(cm[i] - cm[j]))
                        d.append(l2)
                        per_rep_pairs.setdefault(
                            (int(uniq[i]), int(uniq[j])),
                            []
                        ).append(l2)

                if d:
                    print(f"    {cell} rep{rep} (seed={seed}): "
                          f"n_conc={len(uniq)}  L2 mean={np.mean(d):.4f}")

            if not per_rep_pairs:
                print(f"  {cell}: no usable reps -- skipped")
                continue

            # ---------------------------------------------------------
            # Summary statistics computed from the SAME pairwise values
            # written to concept_distances_exp2.csv
            # ---------------------------------------------------------
            pair_means = np.array(
                [np.mean(vals) for vals in per_rep_pairs.values()],
                dtype=float
            )

            l2_min = float(np.min(pair_means))
            l2_max = float(np.max(pair_means))
            l2_mean = float(np.mean(pair_means))
            l2_mean_std = float(np.std(pair_means))

            print(f"  {cell:18s} n_feat={cm_rep0.shape[1]} "
                  f"n_conc={len(uniq_rep0):2d}  L2 mean={l2_mean:.4f} "
                  f"+/-{l2_mean_std:.4f}  (min={l2_min:.4f} max={l2_max:.4f}, "
                  f"{len(RANDOM_STATES)} reps)\n")

            rows_summary.append(dict(
                cell=cell,
                n_informative=n_informative,
                n_concepts=len(uniq_rep0),
                n_reps=len(RANDOM_STATES),
                l2_min=round(l2_min, 4),
                l2_max=round(l2_max, 4),
                l2_mean=round(l2_mean, 4),
                l2_mean_std=round(l2_mean_std, 4)
            ))

            # detailed pairwise distances
            for (a, b), vals in sorted(per_rep_pairs.items()):
                rows_dist.append(dict(
                    cell=cell,
                    n_informative=n_informative,
                    concept_a=a,
                    concept_b=b,
                    l2=round(float(np.mean(vals)), 4),
                    l2_std=round(float(np.std(vals)), 4),
                    n_reps=len(vals)
                ))

            # rep 0 only -- see header note
            for c, m in zip(uniq_rep0, cm_rep0):
                rows_means.append(dict(
                    cell=cell,
                    n_informative=n_informative,
                    concept=int(c),
                    **{f'f{k}': round(float(v), 4) for k, v in enumerate(m)}
                ))

    for rows, name in [
    (rows_means, 'concept_feature_means'),
    (rows_dist, 'concept_distances'),
    (rows_summary, 'concept_distance_summary')
    ]:
        out = os.path.join(RESULTS_DIR, f'{name}_exp2.csv')
        write_dict_csv(out, rows)
        print(f"  Saved: {out}")


print("\nAnalysis 2 complete.")