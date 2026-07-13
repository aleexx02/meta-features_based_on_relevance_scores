# analysis_5.py
# ============================================================
# Analysis of Experiment 5 results.
#
# Streams analysed here: INSECTS-abrupt_balanced,
# INSECTS-abrupt_imbalanced, INSECTS-incgradual_balanced,
# INSECTS-incgradual_imbalanced (genuinely annotated, Table 2, Souza
# et al. 2020), and SPAM (Katakis, Tsoumakas, Vlahavas, 2010 -- see
# generate_real_streams.py for the approximate-drift caveats on this
# one). REAL_STREAMS / N_FEATURES are imported from
# streams.generate_real_streams, so this list updates automatically
# if streams are added/removed there.
#
# electricity and covtype (semi-synthetic, injected drift) are
# analysed separately in analysis_4.py.
#
# I load the pre-computed prequential results from
# results/experiment_5/ and generate the following figures
# per stream:
#
# --sanity
#   relevance_scores_{stream}.png
#     ABFS relevance scores over time, with concept drift boundaries
#     marked in red. For streams with more than
#     N_FEATURES_CAP_THRESHOLD raw features (currently just SPAM,
#     500 features), only the top N_TOP_FEATURES_CAPPED (by variance
#     over time) are plotted -- plotting all 500 would be both
#     unreadable and slow. See select_top_relevance_features().
#
#   metafeatures_{version}_{stream}.png  (one per ABFS version)
#     Each meta-feature dimension plotted over windows, with drift
#     boundaries marked. Same feature-capping as above applies to
#     the 'raw' and 'raw_temporal' versions (their dimensionality
#     scales with n_features); 'aggstats' is always small (8 dims)
#     and is never capped.
#
#   pca_{version}_{stream}.png  (one per ABFS version)
#     2D PCA projection of the meta-feature space, coloured by
#     concept label. Deliberately NOT capped -- PCA is cheap
#     regardless of input dimensionality and benefits from seeing
#     all features, unlike the per-feature trajectory plots above.
#
# --performance
#   trajectory_abfs_{stream}.png
#     Cumulative balanced accuracy over windows for all
#     3 ABFS versions stacked vertically, with drift
#     boundaries and random baseline marked.
#
#   trajectory_komor_{stream}.png
#     Same but for all 9 Komorniczak measure groups
#     arranged in a 3x3 grid.
#
# --shap
#   shap_all_clfs_{version}_{stream}.png  (one per ABFS version)
#     Mean |SHAP| feature importance for all 4 classifiers
#     arranged in a 2x2 subplot. Same feature-capping as --sanity
#     applies here, and for the same reason it matters more: running
#     shap.KernelExplainer on 500 raw dimensions would be slow, not
#     just hard to read, so the cap is applied to the input matrix
#     before fitting/explaining, not just at display time.
#
# --metrics
#   heatmap_f1_{stream}.png
#   heatmap_kappa_{stream}.png
#     Side-by-side heatmaps of final F1 / Kappa values
#     for Komorniczak (left) and ABFS (right).
#
# --stream_analysis
#   stream_drift_entropy_{stream}.png
#     Feature-mean drift intensity, ABFS relevance-score change, and
#     label entropy overlaid on one plot, with drift boundaries
#     marked. Loads pre-computed diagnostics from
#     data/real/annotated_streams_analysis/ (saved by
#     generate_real_streams.py) and re-extracts ABFS relevance scores
#     for the delta_relevance overlay.
#
#   class_distribution_{stream}.png
#     Per-chunk class distribution of the real classification target
#     (species label for INSECTS, spam/not-spam for SPAM) over time.
#
# --gap
#   gap_heatmap_preq_exp5_{stream}.png  (one PER STREAM, not combined)
#     Gap (ABFS raw v2.0 minus best-of-9 Komorniczak measure group)
#     at the final window, one cell per classifier, for that stream
#     alone. Filename includes the stream name so multiple streams'
#     gap heatmaps don't collide or get confused with each other.
#
# Usage:
#   python experiments/experiment_5/analysis_5.py --sanity --performance --shap --metrics --stream_analysis --gap
#
# Inputs (from results/experiment_5/):
#   concept_labels_{stream}.npy                  concept label per window
#   preq_abfs_{version}_ba_{stream}.npy  cumulative BA trajectory
#   preq_komor_{measure}_ba_{stream}.npy same for each Komorniczak group
#   (same pattern for f1, kappa)
# ============================================================

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
import csv

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
from streams.generate_real_streams import REAL_STREAMS, N_FEATURES, CHUNK_SIZE
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone as skclone
from sklearn.decomposition import PCA
# import shap


# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sanity',      action='store_true')
parser.add_argument('--performance', action='store_true')
parser.add_argument('--shap',        action='store_true')
parser.add_argument('--metrics',     action='store_true')
parser.add_argument('--gap',         action='store_true')
parser.add_argument('--stream_analysis', action='store_true')
parser.add_argument('--bars', action='store_true')
parser.add_argument('--concept_dist', action='store_true')
parser.add_argument('--vanilla', action='store_true')
parser.add_argument('--summary', action='store_true')
parser.add_argument('--sparsity', action='store_true',
                    help='SPAM relevance-spread check (effective dimensionality)')

args = parser.parse_args()

RUN_SANITY      = args.sanity
RUN_PERFORMANCE = args.performance
RUN_SHAP        = args.shap
RUN_METRICS     = args.metrics
RUN_GAP         = args.gap
RUN_STREAM_ANALYSIS = args.stream_analysis
RUN_BARS        = args.bars
RUN_CONCEPT_DIST = args.concept_dist
RUN_SUMMARY     = args.summary

print(f"\nRunning analysis for Experiment 5 (annotated real streams):")
print(f"Sanity          : {RUN_SANITY}")
print(f"Performance     : {RUN_PERFORMANCE}")
print(f"SHAP            : {RUN_SHAP}")
print(f"Metrics         : {RUN_METRICS}")
print(f"Gap             : {RUN_GAP}")
print(f"Stream analysis : {RUN_STREAM_ANALYSIS}")
print(f"Bars            : {RUN_BARS}")
print(f"Summary         : {RUN_SUMMARY}")


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

REAL_STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')
REAL_GT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_gt')
REAL_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_analysis')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_5')
FIGURES_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_5', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================

# Streams with more raw ABFS features than this get truncated to the
# top N_TOP_FEATURES_CAPPED (by relevance-score variance) in the
# per-feature plots (relevance_scores, metafeatures_raw*, SHAP). PCA
# is never capped -- see module docstring. Threshold is set above
# INSECTS' max (33 features) so INSECTS is unaffected; SPAM (500) is
# the stream this currently applies to.
N_FEATURES_CAP_THRESHOLD = 40
N_TOP_FEATURES_CAPPED    = 20

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

# one colour per concept -- enough for up to 32 concepts
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
    """Load a result .npy file from results/experiment_5/."""
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)


def load_gt(stream_name):
    """Load the drift chunk indices for this stream."""
    return np.load(os.path.join(REAL_GT_DIR, f'{stream_name}.npy'))


def feat_names_for(version, n_features):
    """Return human-readable feature names for a given ABFS version."""
    if version == 'aggstats':
        return MF_NAMES_AGGSTATS
    elif version == 'raw':
        return [f'r_f{j+1}' for j in range(n_features)]
    else:
        return [f'r_f{j+1}' for j in range(n_features)] + ['delta_mean', 'cosine_sim']


def select_top_relevance_features(X_raw, n_features):
    """
    For streams with more raw features than N_FEATURES_CAP_THRESHOLD,
    select the N_TOP_FEATURES_CAPPED features with the highest
    variance over time in their relevance score -- the most
    dynamic/diagnostically interesting subset, rather than an
    arbitrary first-N truncation. X_raw must be the 'raw' version's
    meta-feature matrix (shape n_windows x n_features), i.e. the raw
    per-feature relevance scores with no aggregation.

    Returns sorted indices (ascending, so plots stay readable in
    original feature order) of the selected features, or all indices
    unchanged if n_features is already at or under the threshold.
    """
    if n_features <= N_FEATURES_CAP_THRESHOLD:
        return np.arange(n_features)
    variances = np.var(X_raw, axis=0)
    top_idx   = np.argsort(variances)[::-1][:N_TOP_FEATURES_CAPPED]
    return np.sort(top_idx)


def re_extract_stream(stream_name, drift_chunks):
    """
    Re-run ABFS on the stream to get relevance scores and all 3 meta-feature
    versions for sanity and SHAP plots. This avoids having to store large
    intermediate arrays to disk -- only the prequential results are saved by
    evaluate_concept_classification_5.py.
    """
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
            best = (label, CLF_NAMES[j], float(v[j]))
    return best


def write_summary_csv(path, title, header, rows):
    """Write the summary as a CSV file."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Saved: {path}")



def plot_concept_distribution(concept_labels, title, out_path, n_concepts=None):
    """Histogram of window counts per concept label."""
    import numpy as np, matplotlib.pyplot as plt, os
    if os.path.exists(out_path):
        print(f"  Exists: {out_path}"); return
    labels = np.asarray(concept_labels)
    if n_concepts is None:
        n_concepts = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_concepts)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n_concepts), counts, color='steelblue', alpha=0.85)
    ax.axhline(len(labels) / n_concepts, color='red', linestyle='--',
               linewidth=1.0, label='uniform (balanced) level')
    ax.set_xlabel('Concept label'); ax.set_ylabel('Number of windows')
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(); print(f"  Saved: {out_path}")



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

    for stream_name in REAL_STREAMS:
        n_features      = N_FEATURES[stream_name]
        n_concepts      = N_CONCEPTS[stream_name]
        random_baseline = 1.0 / n_concepts
        drift_chunks    = load_gt(stream_name)
        boundaries      = list(drift_chunks)
        print(f"\n  {stream_name}")

        scores_over_time, X_by_version, y = re_extract_stream(
            stream_name, drift_chunks)
        unique_concepts = np.unique(y)

        # Feature indices to use for the per-feature plots below
        # (relevance_scores, metafeatures_raw, metafeatures_raw_temporal).
        # PCA further down deliberately uses the FULL feature set
        # regardless of this -- see select_top_relevance_features().
        feature_idx = select_top_relevance_features(
            X_by_version['raw'], n_features)
        capped = len(feature_idx) < n_features
        if capped:
            print(f"  {stream_name}: {n_features} raw features -> showing "
                  f"top {len(feature_idx)} by relevance-score variance in "
                  f"per-feature plots (PCA still uses all {n_features})")

        # ---- relevance scores over time ----
        fname = os.path.join(FIGURES_DIR, f'relevance_scores_{stream_name}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for j in feature_idx:
                ax.plot(scores_over_time[:, j], linewidth=0.7, alpha=0.8,
                        label=f'f{j+1}')
            for b in boundaries:
                ax.axvline(x=b, color='red', linestyle='--',
                           linewidth=0.8, alpha=0.7)
            ax.axvline(x=-1, color='red', linestyle='--',
                       linewidth=0.8, label='concept boundary')
            ax.set_xlabel('Window')
            ax.set_ylabel('Relevance score')
            cap_note = f' (top {len(feature_idx)} by variance)' if capped else ''
            ax.set_title(f'Relevance scores{cap_note} -- {stream_name}\n'
                         f'({n_concepts} concepts, '
                         f'random baseline={random_baseline:.3f})')
            ax.legend(ncol=6, fontsize=6, loc='upper right')
            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        # ---- meta-features over time + PCA -- one per version ----
        for version in ABFS_VERSIONS:
            X     = X_by_version[version]
            names = feat_names_for(version, n_features)
            n_f   = len(names)

            # Which columns of X to actually plot in the per-feature
            # trajectory grid below. aggstats is always small (8 dims)
            # and never capped. raw_temporal's last 2 columns
            # (delta_mean, cosine_sim) are global stream-level scalars,
            # not per-feature, so they're always kept regardless of
            # the per-feature cap.
            if version == 'aggstats':
                plot_idx = np.arange(n_f)
            elif version == 'raw':
                plot_idx = feature_idx
            else:  # raw_temporal
                plot_idx = np.concatenate([feature_idx, [n_features, n_features + 1]])
            n_f_plot = len(plot_idx)

            # meta-feature trajectories
            fname = os.path.join(FIGURES_DIR,
                                 f'metafeatures_{version}_{stream_name}.png')
            if not os.path.exists(fname):
                n_cols = 5
                n_rows = (n_f_plot + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(4*n_cols, 3*n_rows))
                axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
                for plot_k, k in enumerate(plot_idx):
                    axes_flat[plot_k].plot(X[:, k], color='steelblue', linewidth=0.8)
                    for b in boundaries:
                        axes_flat[plot_k].axvline(x=b, color='red',
                                                  linestyle='--', linewidth=0.8, alpha=0.7)
                    axes_flat[plot_k].set_title(names[k], fontsize=8)
                    axes_flat[plot_k].set_xlabel('Window', fontsize=7)
                for plot_k in range(n_f_plot, len(axes_flat)):
                    axes_flat[plot_k].set_visible(False)
                cap_note = f' (top {n_f_plot} by variance)' if n_f_plot < n_f else ''
                fig.suptitle(f'Meta-features ({ABFS_LABELS[version]}){cap_note} -- '
                             f'{stream_name}', fontsize=10)
                fig.tight_layout()
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")
            else:
                print(f"  Exists: {fname}")

            # PCA projection coloured by concept -- ALWAYS uses the
            # full X, not plot_idx, regardless of feature capping above
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
                ax.set_title(f'PCA -- {ABFS_LABELS[version]}\n{stream_name}')
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

    for stream_name in REAL_STREAMS:
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
                    ax.set_title(f'{ABFS_LABELS[version]} -- no data'); continue
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
            fig.suptitle(f'ABFS trajectories -- {stream_name}\n'
                         f'({n_concepts} concepts | '
                         f'random baseline={random_baseline:.3f})',
                         fontsize=12)
            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")

        # all 9 Komorniczak groups in a 3x3 grid
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
                    ax.set_title(f'{measure} -- no data'); continue
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
            fig.suptitle(f'Komorniczak trajectories -- {stream_name}',
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
    import shap
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)

    for stream_name in REAL_STREAMS:
        n_features = N_FEATURES[stream_name]
        print(f"\n  {stream_name}")

        # concept_labels_{stream}.npy contains the ground truth concept label
        # per window, derived from the known drift boundaries (not from ABFS).
        # It was saved by evaluate_concept_classification_5.py.
        y = load('concept_labels', stream_name)
        if y is None:
            print(f"  concept labels not found -- skipping."); continue

        all_done = all(
            os.path.exists(os.path.join(
                FIGURES_DIR, f'shap_all_clfs_{v}_{stream_name}.png'))
            for v in ABFS_VERSIONS)
        if all_done:
            print(f"  All SHAP figures exist -- skipping."); continue

        drift_chunks = load_gt(stream_name)
        _, X_by_version, _ = re_extract_stream(stream_name, drift_chunks)

        # Same feature cap as --sanity, applied here BEFORE fitting and
        # running shap.KernelExplainer -- for SPAM-scale feature counts
        # this isn't just a display fix, it materially cuts SHAP's
        # runtime, which scales with input dimensionality.
        feature_idx = select_top_relevance_features(
            X_by_version['raw'], n_features)
        capped = len(feature_idx) < n_features

        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR,
                                 f'shap_all_clfs_{version}_{stream_name}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue

            X          = X_by_version[version]
            feat_names = feat_names_for(version, n_features)

            if version == 'raw':
                sel = feature_idx
            elif version == 'raw_temporal':
                sel = np.concatenate([feature_idx, [n_features, n_features + 1]])
            else:  # aggstats -- never capped, n_f always small (8)
                sel = np.arange(len(feat_names))

            X          = X[:, sel]
            feat_names = [feat_names[i] for i in sel]

            cap_note = f' (top {len(sel)} by variance)' if capped and version != 'aggstats' else ''
            print(f"  SHAP [{version}]: X={X.shape}{cap_note}")

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

            fig.suptitle(f'SHAP -- {ABFS_LABELS[version]}{cap_note}\n{stream_name}',
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
            ax.set_title(f'Komorniczak -- {metric_label}', fontsize=12)

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
            ax.set_title(f'ABFS -- {metric_label}', fontsize=12)

            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle(
                f'{metric_label} -- {stream_name}\n'
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

    for stream_name in REAL_STREAMS:
        print(f"\n  {stream_name}")

        drift_chunks = load_gt(stream_name)
        boundaries   = list(drift_chunks)

        # ---- load analysis data ----
        try:
            drift_intensity = np.load(os.path.join(
                REAL_ANALYSIS_DIR, f'{stream_name}_drift_intensity.npy'))

            class_dist = np.load(os.path.join(
                REAL_ANALYSIS_DIR, f'{stream_name}_class_distribution.npy'))

            entropy_vals = np.load(os.path.join(
                REAL_ANALYSIS_DIR, f'{stream_name}_label_entropy.npy'))

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
            print(f"  Missing analysis files for {stream_name} -- skipping.")
            continue

        n_chunks = len(drift_intensity)

        # ========================================================
        # Drift + entropy
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

            # ---- entropy ----
            ax2 = ax1.twinx()
            ax2.plot(entropy_vals, color='darkorange',
                    label='Label entropy', alpha=0.7)
            ax2.set_ylabel('Entropy', color='darkorange')

            # ---- boundaries ----
            for b in boundaries:
                ax1.axvline(x=b, color='red', linestyle='--',
                            linewidth=0.8, alpha=0.7)

            ax1.set_xlabel('Window')

            # ---- legend ----
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

            ax1.set_title(f'Drift vs ABFS dynamics -- {stream_name}')

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
            ax.set_title(f'Class distribution over time -- {stream_name}')
            ax.legend(ncol=4, fontsize=8)

            fig.tight_layout()
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {fname}")
        else:
            print(f"  Exists: {fname}")


#  ============================================================
#  GAP HEATMAP - one file PER STREAM, 3 rows (one per ABFS version).
#  Each cell (version, clf) is that version's final-window BA minus
#  the best-of-9 Komorniczak group for that same classifier. best_komor
#  is computed per classifier (max over the 9 groups), held fixed across
#  the 3 rows so the rows are directly comparable -- a bluer row means a
#  genuinely stronger version, not a weaker Komorniczak reference. Keeping
#  the versions separate (rather than max-ing over them) makes the variant
#  flip visible: aggstats positive where raw is negative on balanced INSECTS.
# ============================================================
if RUN_GAP:
    print("\n" + "="*60)
    print("GAP HEATMAP - per ABFS version vs best Komorniczak group (per classifier)")
    print("="*60)

    for stream_name in REAL_STREAMS:
        fname = os.path.join(FIGURES_DIR,
                             f'gap_heatmap_preq_exp5_{stream_name}.png')
        if os.path.exists(fname):
            print(f"  Exists: {fname}"); continue

        # best Komorniczak per classifier: element-wise max over the 9
        # groups at the final window
        komor_finals = []
        for measure in MEASURES:
            data = load(f'preq_komor_{measure}_ba', stream_name, optional=True)
            if data is not None:
                komor_finals.append(data[-1, :])

        if not komor_finals:
            print(f"  {stream_name}: missing Komorniczak data -- skipping.")
            continue
        best_komor = np.nanmax(np.vstack(komor_finals), axis=0)   # (N_CLFS,)

        # per-version gap rows: version final BA minus best_komor
        gap_matrix = np.full((len(ABFS_VERSIONS), N_CLFS), np.nan)
        for v_id, version in enumerate(ABFS_VERSIONS):
            data = load(f'preq_abfs_{version}_ba', stream_name, optional=True)
            if data is not None:
                gap_matrix[v_id, :] = data[-1, :] - best_komor

        if np.all(np.isnan(gap_matrix)):
            print(f"  {stream_name}: missing ABFS data -- skipping.")
            continue

        # provenance print: per version, per classifier
        for v_id, version in enumerate(ABFS_VERSIONS):
            for j, clf in enumerate(CLF_NAMES):
                val = gap_matrix[v_id, j]
                if not np.isnan(val):
                    print(f"    {stream_name:32s} {version:12s} {clf:4s} "
                          f"Komor {best_komor[j]:.3f}  gap {val:+.3f}")

        vmax = (np.nanmax(np.abs(gap_matrix))
                if np.any(~np.isnan(gap_matrix)) else 1.0)

        fig, ax = plt.subplots(
            figsize=(max(6, N_CLFS * 1.8), 1.1 * len(ABFS_VERSIONS) + 1.6))
        im = ax.imshow(gap_matrix, vmin=-vmax, vmax=vmax,
                       cmap='RdBu', aspect='auto')
        for i in range(len(ABFS_VERSIONS)):
            for j in range(N_CLFS):
                val = gap_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                            fontsize=11,
                            color='white' if abs(val) > vmax * 0.6 else 'black')
        ax.set_xticks(range(N_CLFS))
        ax.set_xticklabels(CLF_NAMES, fontsize=11)
        ax.set_yticks(range(len(ABFS_VERSIONS)))
        ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
        ax.set_xlabel('Classifier', fontsize=11, labelpad=8)
        ax.set_title(f'Gap per ABFS version (version minus best Komorniczak) '
                     f'-- {stream_name}', fontsize=11, pad=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Gap (BA)', fontsize=9)
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(); print(f"  Saved: {fname}")

        

# ============================================================
#  BARS - BA per ABFS version (best clf) + Komorniczak, per stream
#  (Exp 5 has no swept parameter, so this replaces the sensitivity
#   curves used in Exp 2/3/4: grouped bars, one group per stream.)
# ============================================================
if RUN_BARS:
    print("\n" + "="*60)
    print("BA per version, per stream")
    print("="*60)

    VERSION_COLORS = {'aggstats': '#911eb4', 'raw': '#4363d8', 'raw_temporal': '#f58231'}

    streams, abfs_ba, komor_ba = [], {v: [] for v in ABFS_VERSIONS}, []
    for stream_name in REAL_STREAMS:
        # best clf per ABFS version at the final window
        row_ok = False
        per_version = {}
        for version in ABFS_VERSIONS:
            d = load(f'preq_abfs_{version}_ba', stream_name, optional=True)
            per_version[version] = float(np.max(d[-1, :])) if d is not None else np.nan
            row_ok |= d is not None
        # best Komorniczak group (best clf)
        kb = np.nan
        for measure in MEASURES:
            d = load(f'preq_komor_{measure}_ba', stream_name, optional=True)
            if d is not None:
                v = float(np.max(d[-1, :]))
                kb = v if np.isnan(kb) else max(kb, v)
        if not row_ok and np.isnan(kb):
            continue
        streams.append(stream_name)
        for version in ABFS_VERSIONS:
            abfs_ba[version].append(per_version[version])
        komor_ba.append(kb)

    if not streams:
        print("  no data -- skipping.")
    else:
        fname = os.path.join(FIGURES_DIR, 'ba_per_version.png')
        n_groups = len(streams)
        n_bars   = len(ABFS_VERSIONS) + 1            # 3 versions + Komorniczak
        width    = 0.8 / n_bars
        x        = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.8), 5))
        for bi, version in enumerate(ABFS_VERSIONS):
            ax.bar(x + bi * width, abfs_ba[version], width,
                   color=VERSION_COLORS[version],
                   label=f'ABFS {ABFS_LABELS[version]}')
        ax.bar(x + len(ABFS_VERSIONS) * width, komor_ba, width,
               color='#3cb44b', label='Komorniczak best-of-9')

        # random-baseline marker per stream
        for gi, s in enumerate(streams):
            rb = 1.0 / N_CONCEPTS[s]
            ax.hlines(rb, x[gi] - width/2, x[gi] + n_bars*width - width/2,
                      color='red', linestyle='--', linewidth=1.0,
                      label='random baseline' if gi == 0 else None)

        ax.set_xticks(x + (n_bars - 1) * width / 2)
        ax.set_xticklabels(streams, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Final balanced accuracy (best clf)')
        ax.set_ylim(0, 1)
        ax.set_title('Exp 5: BA per ABFS version vs Komorniczak (best classifier)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(); print(f"  Saved: {fname}")



if RUN_CONCEPT_DIST:
    print("\n" + "="*60); print("CONCEPT DISTRIBUTION"); print("="*60)
    for stream_name in REAL_STREAMS:
        cl = load('concept_labels', stream_name, optional=True)
        if cl is None:
            print(f"  {stream_name}: no concept_labels — skipping"); continue
        plot_concept_distribution(
            cl,
            f'Concept distribution - {stream_name} ({N_CONCEPTS[stream_name]} concepts)',
            os.path.join(FIGURES_DIR, f'concept_distribution_{stream_name}.png'),
            n_concepts=N_CONCEPTS[stream_name])
        


if args.vanilla:
    print("\n" + "="*60)
    print("VANILLA BASELINE COMPARISON")
    print("="*60)

    def best_final_ba(prefix, stream):
        """Best classifier's final-window BA. Real streams have NO rep axis:
        arrays are (n_windows, n_clfs), not (n_reps, n_windows, n_clfs)."""
        d = load(prefix, stream, optional=True)
        if d is None:
            return None
        return float(np.max(d[-1, :]))          # <-- no mean over reps

    def vanilla_row(stream):
        v = best_final_ba('preq_vanilla_ba', stream)
        a = max([b for b in (best_final_ba(f'preq_abfs_{ver}_ba', stream)
                             for ver in ABFS_VERSIONS) if b is not None], default=None)
        k = max([b for b in (best_final_ba(f'preq_komor_{m}_ba', stream)
                             for m in MEASURES) if b is not None], default=None)
        return v, a, k

    rows = []
    for stream in REAL_STREAMS:
        v, a, k = vanilla_row(stream)
        if v is None:
            print(f"  {stream}: no vanilla results -- skipping"); continue
        rows.append((stream, v, a, k))
        print(f"  {stream:25s}  vanilla={v:.3f}  abfs={a:.3f}  komor={k:.3f}  "
              f"(baseline {1/N_CONCEPTS[stream]:.3f})")

    import csv
    out = os.path.join(RESULTS_DIR, 'vanilla_comparison_exp5.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stream', 'vanilla_ba', 'abfs_best_ba', 'komor_best_ba', 'random_baseline'])
        for s, v, a, k in rows:
            w.writerow([s, round(v, 3), round(a, 3), round(k, 3), round(1/N_CONCEPTS[s], 3)])
    print(f"\n  Saved: {out}")


if args.summary:
    print("\n" + "="*60); print("SUMMARY TABLE"); print("="*60)
    rows = []
    for s in REAL_STREAMS:
        loadf = lambda prefix, st=s: load(prefix, st, optional=True)
        kb = best_side(loadf, [(m, f'preq_komor_{m}_ba') for m in MEASURES], has_reps=False)
        ab = best_side(loadf, [(ABFS_LABELS[v], f'preq_abfs_{v}_ba') for v in ABFS_VERSIONS], has_reps=False)
        if kb[0] is None or ab[0] is None:
            continue
        rb = 1.0 / N_CONCEPTS[s]
        n_drifts_real = len(load_gt(s))
        cl = load('concept_labels', s, optional=True)
        n_win = len(cl) if cl is not None else '-'
        rows.append([s, N_FEATURES[s], N_CONCEPTS[s], n_drifts_real, n_win, f'{rb:.3f}',
                     f'{kb[0]} / {kb[1]}', f'{kb[2]:.3f}',
                     f'{ab[0]} / {ab[1]}', f'{ab[2]:.3f}',
                     f'{ab[2]-kb[2]:+.3f}'])
    header = ['stream', 'n_feat', 'n_conc', 'n_drifts', 'n_win', 'baseline',
              'best Komor (grp/clf)', 'Komor BA',
              'best ABFS (ver/clf)', 'ABFS BA', 'gap']
    write_summary_csv(os.path.join(RESULTS_DIR, 'summary_exp5.csv'),
                      'Experiment 5 summary (real streams)', header, rows)
    



if args.sparsity:
    print("\n" + "="*60)
    print("RELEVANCE SPREAD / EFFECTIVE DIMENSIONALITY")
    print("="*60)

    import strlearn as sl
    from abfs.abfs_implementation import ABFS_match

    for stream_name in REAL_STREAMS:
        nf = N_FEATURES[stream_name]
        stream_path = os.path.join(REAL_STREAM_DIR, f'{stream_name}.npy')
        stream = sl.streams.NPYParser(stream_path, chunk_size=CHUNK_SIZE, n_chunks=100000)
        abfs = ABFS_match(n_features=nf, categorical_features=[],
                          accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)

        all_scores = []
        for _ in range(100000):
            try:
                X, y = stream.get_chunk()
            except Exception:
                break
            if len(X) == 0:
                break
            for i in range(len(X)):
                abfs.update(X[i], y[i])
            all_scores.append(abfs.relevance_scores().copy())

        R = np.array(all_scores)                 # (n_windows, n_features)
        mean_rel = R.mean(axis=0)
        srt = np.sort(mean_rel)[::-1]
        std_rel = R.std(axis=0)                 # how much each feature's relevance VARIES
        srt_v = np.sort(std_rel)[::-1]
        cum_v = np.cumsum(srt_v) / srt_v.sum()
        total = srt.sum()

        print(f"\n  {stream_name}  (n_features = {nf})")
        if total <= 0:
            print("    all relevance scores are zero -- skipping")
            continue
        cum = np.cumsum(srt) / total
        # how many features to reach 50/80/90/95% of total relevance
        for share in [0.5, 0.8, 0.9, 0.95]:
            k = int(np.searchsorted(cum, share) + 1)
            print(f"    top {k:4d} features ({k/nf:5.1%} of them) carry {share:.0%} of total relevance")
        # how many are non-negligible at all
        for thr in [0.01, 0.05, 0.1]:
            print(f"    features with mean relevance > {thr}: {(mean_rel > thr).sum()} / {nf}")


        for share in [0.8, 0.9]:
            k = int(np.searchsorted(cum_v, share) + 1)
            print(f"    top {k:4d} features carry {share:.0%} of relevance VARIATION")

print("\nAnalysis 5 complete.")

