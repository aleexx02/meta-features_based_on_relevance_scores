# analysis_2.py
# ============================================================
# Analysis of Experiment 2 results (stream configuration sensitivity).
#
# Usage:
#   python experiments/experiment_2/analysis_2.py [--sanity] [--performance] [--shap] [--metrics] [--grid]
#
# Flags:
#   --sanity      : relevance scores, meta-features per window, PCA (rep 0 only per cell)
#   --performance : cumulative BA trajectory over windows (prequential)
#   --shap        : SHAP feature importance per cell (MLP, raw v2.0)
#   --metrics     : F1 and Kappa heatmaps per cell (prequential only)
#   --grid        : gap heatmap + sensitivity curves across 4x4 grid (prequential only)
#
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
from sklearn.neural_network import MLPClassifier
import shap
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL


# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description='Analysis for Experiment 2.')
parser.add_argument('--sanity',      action='store_true', help='Sanity check plots per cell')
parser.add_argument('--performance', action='store_true', help='Trajectory plots (prequential)')
parser.add_argument('--shap',        action='store_true', help='SHAP analysis per cell')
parser.add_argument('--metrics',     action='store_true', help='F1 and Kappa heatmaps per cell')
parser.add_argument('--grid',        action='store_true', help='Gap heatmap + sensitivity curves')
args = parser.parse_args()

RUN_SANITY      = args.sanity
RUN_PERFORMANCE = args.performance
RUN_SHAP        = args.shap
RUN_METRICS     = args.metrics
RUN_GRID        = args.grid

print(f"\nRunning analysis for Experiment 2")
print(f"Sanity:      {RUN_SANITY}")
print(f"Performance: {RUN_PERFORMANCE}")
print(f"SHAP:        {RUN_SHAP}")
print(f"Metrics:     {RUN_METRICS}")
print(f"Grid:        {RUN_GRID}")


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
N_CHUNKS              = 5000
N_FEATURES            = 20
WARMUP_WINDOWS        = 10
SCORE_INTERVAL        = 100
N_REPLICATIONS        = 5
CHUNK_SIZE_DEFAULT    = 200
N_INFORMATIVE_DEFAULT = 10

CHUNK_SIZES    = [100, 200, 500, 1000]
N_INFORMATIVES = [3, 5, 10, 15]

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

DRIFT_CONFIGS = [
    ('sudden',  20, 9999, 21),
    ('gradual',  6,    5, 25),
]

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

MF_NAMES = [f'r_f{j+1}' for j in range(N_FEATURES)]

clf_names_preq = [name for name, _ in BASE_CLFS_PREQUENTIAL]
N_CLFS         = len(clf_names_preq)

CLF_COLORS = {
    'GNB': '#e6194b', 'KNN': '#3cb44b',
    'HT':  '#f032e6', 'MLP': '#911eb4',
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
    path = os.path.join(RESULTS_DIR, f'{prefix}_{tag}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)


def load_abfs(version, tag, optional=False):
    return load(f'preq_abfs_{version}_ba', tag, optional=optional)


def load_komor_best(tag):
    """Return mean final-window BA for the best Komorniczak measure group per clf."""
    best = None
    for measure in MEASURES:
        arr = load(f'preq_komor_{measure}_ba', tag, optional=True)
        if arr is None:
            continue
        per_clf = np.mean(arr[:, -1, :], axis=0)  # (n_clfs,)
        if best is None or np.max(per_clf) > np.max(best):
            best = per_clf
    return best  # (n_clfs,) or None


def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing,
                        stream.n_drifts)[1][::chunk_size]
    concept, decreasing, labels = 0, True, []
    for chunk in range(n_chunks):
        if decreasing:
            if concept % 4 == 0 and e[chunk] < 0.9:  concept += 1
            if concept % 4 == 1 and e[chunk] < 0.75: concept += 1
            if concept % 4 == 2 and e[chunk] < 0.25: concept += 1
            if concept % 4 == 3 and e[chunk] < 0.1:  concept += 1; decreasing = False
        else:
            if concept % 4 == 0 and e[chunk] > 0.1:  concept += 1
            if concept % 4 == 1 and e[chunk] > 0.25: concept += 1
            if concept % 4 == 2 and e[chunk] > 0.75: concept += 1
            if concept % 4 == 3 and e[chunk] > 0.9:  concept += 1; decreasing = True
        labels.append(concept)
    return np.array(labels)


def get_concept_boundaries(concept_labels_all, n_chunks):
    return [i for i in range(1, n_chunks)
            if concept_labels_all[i] != concept_labels_all[i-1]]


def extract_stream_data(rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative):
    config = dict(
        n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=chunk_size,
        n_features=N_FEATURES, n_informative=n_informative,
        n_redundant=0, n_repeated=0,
        concept_sigmoid_spacing=concept_sigmoid_spacing,
        random_state=rs,
    )
    stream = StreamGenerator(**config)

    # pass 1: relevance scores over time
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
    concept_selector_saved = stream.concept_selector.copy()
    scores_over_time = np.array(scores_over_time)

    if drift_type == 'sudden':
        concept_labels_all = np.array([
            int(np.bincount(concept_selector_saved[i*chunk_size:(i+1)*chunk_size]).argmax())
            for i in range(N_CHUNKS)
        ])
    else:
        concept_labels_all = assign_labels_gradual(stream, N_CHUNKS, chunk_size)

    boundaries = get_concept_boundaries(concept_labels_all, N_CHUNKS)

    # pass 2: raw score meta-features (v2.0)
    abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                      accuracy_window_size=chunk_size,
                      class_window_size=chunk_size)
    meta_features, concept_labels = [], []
    wt_prev, window_counter = None, 0

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
    X[np.isnan(X)] = 1; X[np.isinf(X)] = 1
    return scores_over_time, concept_labels_all, boundaries, X, y


def get_stream_boundaries_meta(drift_type, n_drifts, concept_sigmoid_spacing,
                               chunk_size, n_informative):
    _, concept_labels_all, boundaries, _, _ = extract_stream_data(
        RANDOM_STATES[0], drift_type, n_drifts, concept_sigmoid_spacing,
        chunk_size, n_informative)
    return [b - WARMUP_WINDOWS for b in boundaries if b - WARMUP_WINDOWS > 0]


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
                print(f"\n  {tag}")

                rs = RANDOM_STATES[0]
                scores_over_time, concept_labels_all, boundaries, X, y = \
                    extract_stream_data(rs, drift_type, n_drifts,
                                        concept_sigmoid_spacing,
                                        chunk_size, n_informative)
                unique_concepts = np.unique(y)

                # relevance scores over time
                fname = os.path.join(FIGURES_DIR, f'relevance_scores_{tag}_rep0.png')
                if not os.path.exists(fname):
                    fig, ax = plt.subplots(figsize=(14, 4))
                    for j in range(N_FEATURES):
                        ax.plot(scores_over_time[:, j], label=f'f{j+1}', linewidth=0.8)
                    for b in boundaries:
                        ax.axvline(x=b * chunk_size // SCORE_INTERVAL,
                                   color='red', linestyle='--', linewidth=0.8, alpha=0.6)
                    ax.axvline(x=-1, color='red', linestyle='--',
                               linewidth=0.8, label='concept boundary')
                    ax.set_xlabel('Time (x100 instances)')
                    ax.set_ylabel('Relevance score')
                    ax.set_title(f'Relevance scores — {tag} — rep0')
                    ax.legend(ncol=5, fontsize=7)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150); plt.close()
                    print(f"  Saved: {fname}")

                # meta-features per window
                fname = os.path.join(FIGURES_DIR, f'metafeatures_{tag}_rep0.png')
                if not os.path.exists(fname):
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
                    fig.suptitle(f'Meta-features (raw v2.0) — {tag} — rep0', fontsize=10)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150); plt.close()
                    print(f"  Saved: {fname}")

                # PCA
                fname = os.path.join(FIGURES_DIR, f'pca_{tag}_rep0.png')
                if not os.path.exists(fname):
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
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                    ax.set_title(f'PCA — {tag} — rep0')
                    ax.legend(ncol=4, fontsize=7)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150); plt.close()
                    print(f"  Saved: {fname}")


# ============================================================
#  2. PERFORMANCE TRAJECTORY OVER TIME (PREQUENTIAL)
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
                main_boundaries = (boundaries_meta[::4] if drift_type == 'gradual'
                                   else boundaries_meta)
                random_baseline = 1 / n_concepts

                # one figure per ABFS version + one for Komorniczak best group
                sources = [(f'preq_abfs_{v}_ba', ABFS_LABELS[v]) for v in ABFS_VERSIONS]
                sources += [(f'preq_komor_statistical_ba', 'Komorniczak (statistical)')]

                for prefix, label in sources:
                    short = prefix.replace('preq_', '').replace('_ba', '')
                    fname = os.path.join(FIGURES_DIR,
                                         f'trajectory_{short}_{tag}.png')
                    if os.path.exists(fname):
                        print(f"  Exists: {fname}")
                        continue

                    data = load(prefix, tag, optional=True)
                    if data is None:
                        continue
                    # shape: (n_reps, n_windows, n_clfs)
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
                    ax.set_title(f'Performance trajectory — {label} — {tag}')
                    ax.legend(fontsize=9, ncol=3)
                    ax.set_xlim(0, n_windows)
                    ax.set_ylim(0, 1)
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150); plt.close()
                    print(f"  Saved: {fname}")


# ============================================================
#  3. SHAP ANALYSIS  (raw v2.0, MLP, all replications combined)
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("3. SHAP ANALYSIS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag   = make_tag(chunk_size, n_informative, drift_type)
                fname = os.path.join(FIGURES_DIR, f'shap_raw_{tag}.png')
                if os.path.exists(fname):
                    print(f"  Exists: {fname}")
                    continue
                print(f"\n  SHAP: {tag}")

                all_X, all_y = [], []
                for rs in RANDOM_STATES:
                    _, _, _, X, y = extract_stream_data(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)
                    all_X.append(X); all_y.append(y)

                X_all = np.vstack(all_X)
                y_all = np.concatenate(all_y)
                X_all[np.isnan(X_all)] = 1; X_all[np.isinf(X_all)] = 1

                mlp = MLPClassifier(random_state=11313)
                mlp.fit(X_all, y_all)

                explainer   = shap.KernelExplainer(
                    mlp.predict_proba, shap.sample(X_all, 100))
                shap_values = explainer.shap_values(
                    shap.sample(X_all, 200), nsamples=100)

                shap_array = np.array(shap_values)
                mean_abs_shap = (np.mean(np.abs(shap_array), axis=(0, 2))
                                 if shap_array.ndim == 3
                                 else np.mean(np.abs(shap_array), axis=0))

                sorted_idx = np.argsort(mean_abs_shap)[::-1]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(MF_NAMES)), mean_abs_shap[sorted_idx],
                       color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(MF_NAMES)))
                ax.set_xticklabels([MF_NAMES[i] for i in sorted_idx],
                                   rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Mean absolute SHAP value')
                ax.set_title(f'SHAP — raw scores v2.0 — {tag}\n'
                             f'(MLP, {N_REPLICATIONS} replications combined)')
                fig.tight_layout()
                fig.savefig(fname, dpi=150); plt.close()
                print(f"  Saved: {fname}")


# ============================================================
#  4. ADDITIONAL METRICS HEATMAPS (F1, KAPPA) — prequential only
# ============================================================
if RUN_METRICS:
    print("\n" + "="*60)
    print("4. METRICS HEATMAPS (F1, KAPPA) — prequential")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for chunk_size in CHUNK_SIZES:
            for n_informative in N_INFORMATIVES:
                tag = make_tag(chunk_size, n_informative, drift_type)

                for metric, metric_label in [('f1', 'F1'), ('kappa', 'Kappa')]:
                    fname = os.path.join(FIGURES_DIR,
                                         f'heatmap_{metric}_preq_{tag}.png')
                    if os.path.exists(fname):
                        print(f"  Exists: {fname}")
                        continue

                    # ABFS: all 3 versions
                    abfs_rows = []
                    for version in ABFS_VERSIONS:
                        data = load(f'preq_abfs_{version}_{metric}', tag, optional=True)
                        if data is not None:
                            final = data[:, -1, :]
                            abfs_rows.append((ABFS_LABELS[version],
                                              np.mean(final, axis=0),
                                              np.std(final, axis=0)))

                    # Komorniczak: statistical as representative
                    komor_rows = []
                    for measure in MEASURES:
                        data = load(f'preq_komor_{measure}_{metric}', tag, optional=True)
                        if data is not None:
                            final = data[:, -1, :]
                            komor_rows.append((measure,
                                               np.mean(final, axis=0),
                                               np.std(final, axis=0)))

                    if not abfs_rows and not komor_rows:
                        continue

                    n_rows_k = len(komor_rows)
                    n_rows_a = len(abfs_rows)
                    if n_rows_k == 0 or n_rows_a == 0:
                        continue

                    komor_m = np.array([r[1] for r in komor_rows])
                    komor_s = np.array([r[2] for r in komor_rows])
                    abfs_m  = np.array([r[1] for r in abfs_rows])
                    abfs_s  = np.array([r[2] for r in abfs_rows])

                    fig, axes = plt.subplots(
                        1, 2, figsize=(22, max(5, n_rows_k * 0.75)),
                        gridspec_kw={'width_ratios': [3, 1.5]})

                    for ax, matrix, std_mat, row_labels, title in [
                        (axes[0], komor_m, komor_s,
                         [r[0] for r in komor_rows],
                         f'Komorniczak — {metric_label}'),
                        (axes[1], abfs_m, abfs_s,
                         [r[0] for r in abfs_rows],
                         f'ABFS — {metric_label}'),
                    ]:
                        im = ax.imshow(matrix, vmin=0.0, vmax=1.0,
                                       cmap='Blues', aspect='auto')
                        for i in range(len(row_labels)):
                            for j in range(N_CLFS):
                                val = matrix[i, j]; std = std_mat[i, j]
                                txt_color = 'white' if val > 0.6 else 'black'
                                ax.text(j, i, f'{val:.3f}\n(±{std:.3f})',
                                        ha='center', va='center', fontsize=9,
                                        color=txt_color, linespacing=1.4)
                        ax.set_xticks(range(N_CLFS))
                        ax.set_xticklabels(clf_names_preq, fontsize=10)
                        ax.set_yticks(range(len(row_labels)))
                        ax.set_yticklabels(row_labels, fontsize=9)
                        ax.set_title(title, fontsize=12)

                    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    fig.suptitle(f'{metric_label} — prequential — {tag}', fontsize=12)
                    plt.tight_layout()
                    fig.savefig(fname, dpi=150); plt.close()
                    print(f"  Saved: {fname}")


# ============================================================
#  5+6. GRID ANALYSES: GAP HEATMAP + SENSITIVITY CURVES (prequential only)
# ============================================================
if RUN_GRID:
    print("\n" + "="*60)
    print("5+6. GAP HEATMAP + SENSITIVITY CURVES")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:

        # For gap and sensitivity: use ABFS raw (v2.0) and best-of-9 Komorniczak
        grid_abfs_preq      = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
        grid_komor_preq     = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES)), np.nan)
        grid_abfs_preq_clf  = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES), N_CLFS), np.nan)
        grid_komor_preq_clf = np.full((len(CHUNK_SIZES), len(N_INFORMATIVES), N_CLFS), np.nan)

        for i, chunk_size in enumerate(CHUNK_SIZES):
            for j, n_informative in enumerate(N_INFORMATIVES):
                tag = make_tag(chunk_size, n_informative, drift_type)

                pr_abfs = load('preq_abfs_raw_ba', tag, optional=True)
                if pr_abfs is not None:
                    per_clf = np.mean(pr_abfs[:, -1, :], axis=0)
                    grid_abfs_preq[i, j]     = np.max(per_clf)
                    grid_abfs_preq_clf[i, j] = per_clf

                komor_best = load_komor_best(tag)
                if komor_best is not None:
                    grid_komor_preq[i, j]     = np.max(komor_best)
                    grid_komor_preq_clf[i, j] = komor_best

        x_labels = [str(ni) for ni in N_INFORMATIVES]
        y_labels = [str(cs) for cs in CHUNK_SIZES]

        # ---- gap heatmap ----
        gap_grid = grid_abfs_preq - grid_komor_preq
        fname = os.path.join(FIGURES_DIR,
                             f'gap_heatmap_preq_{drift_type}.png')
        if not os.path.exists(fname):
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
            ax.set_ylabel('chunk_size', fontsize=11)
            ax.set_title(
                f'Gap heatmap (ABFS raw v2.0 − Komorniczak best) — Prequential\n'
                f'{drift_type} drift ({n_concepts} concepts) [best clf per cell]',
                fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(fname, dpi=150); plt.close()
            print(f"  Gap heatmap saved: {fname}")

        # ---- sensitivity curves ----
        ni_idx = N_INFORMATIVES.index(N_INFORMATIVE_DEFAULT)
        cs_idx = CHUNK_SIZES.index(CHUNK_SIZE_DEFAULT)

        # BA vs chunk_size (n_informative=10 fixed)
        fname = os.path.join(FIGURES_DIR,
            f'sensitivity_chunk_preq_ninf_{N_INFORMATIVE_DEFAULT}_{drift_type}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(8, 4))
            for clf_id, name in enumerate(clf_names_preq):
                color = CLF_COLORS.get(name, f'C{clf_id}')
                ax.plot(CHUNK_SIZES, grid_abfs_preq_clf[:, ni_idx, clf_id],
                        color=color, label=f'{name} ABFS',
                        linewidth=1.5, marker='o', markersize=5)
                ax.plot(CHUNK_SIZES, grid_komor_preq_clf[:, ni_idx, clf_id],
                        color=color, label=f'{name} Komor',
                        linewidth=1.5, linestyle='--', marker='s', markersize=5)
            ax.axhline(y=1/n_concepts, color='red', linestyle=':',
                       linewidth=1.0, label='random baseline')
            ax.set_xlabel('chunk_size', fontsize=11)
            ax.set_ylabel('Mean balanced accuracy', fontsize=10)
            ax.set_title(
                f'BA vs chunk_size (n_informative={N_INFORMATIVE_DEFAULT}) — Prequential\n'
                f'{drift_type} drift', fontsize=11)
            ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_xticks(CHUNK_SIZES)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(fname, dpi=150); plt.close()
            print(f"  Sensitivity (chunk) saved: {fname}")

        # BA vs n_informative (chunk_size=200 fixed)
        fname = os.path.join(FIGURES_DIR,
            f'sensitivity_ninf_preq_chunk_{CHUNK_SIZE_DEFAULT}_{drift_type}.png')
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
                f'BA vs n_informative (chunk_size={CHUNK_SIZE_DEFAULT}) — Prequential\n'
                f'{drift_type} drift', fontsize=11)
            ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_xticks(N_INFORMATIVES)
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(fname, dpi=150); plt.close()
            print(f"  Sensitivity (ninf) saved: {fname}")

print("\nAnalysis 2 complete.")