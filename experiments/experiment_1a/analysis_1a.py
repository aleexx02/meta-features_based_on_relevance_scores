# analysis_1a.py
# ============================================================
# Analysis of Experiment 1a results 
#

# Loads pre-computed results (.npy files) from
# results/experiment_1a/ and produces:
#
#   1. Sanity check plots (per replication x drift type):
#      - Relevance scores over time
#      - Meta-features over windows
#      - PCA projection of meta-feature vectors
#
#   2. Performance variance across replications:
#      - Mean balanced accuracy per replication per classifier
#      - One plot per meta-feature set per drift type
#
#   3. SHAP analysis:
#      - Mean absolute SHAP values per meta-feature (across 4 different classifiers)
#      - One plot per meta-feature set per drift type
#        averaged over replications
#
#   4. Additional metrics heatmaps (F1, Kappa):
#      - Same format as balanced accuracy heatmaps
#      - Includes Komorniczak (statistical) as baseline
#
# Inputs (from results/experiment_1a/):
#            clf_replication_ba_*.npy, clf_replication_f1_*.npy,
#            clf_replication_kappa_*.npy (from replication_check_1a.py)
#
# Outputs saved to results/experiment_1a/figures/analysis/
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # experiments/ for classifier_sweep_komor.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))  # project root for plot_results.py, ABFS implementation and meta-features extraction

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures_raw, extract_metafeatures_raw_temporal,
    MF_NAMES_RAW, MF_NAMES_RAW_TEMPORAL
)
from classifier_sweep_komor import BASE_CLFS
from plot_results import print_sanity_check_summary

# ============================================================
#  ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description='Analysis for Experiment 1a.')
parser.add_argument('--sanity', action='store_true', help='Run sanity check plots')
parser.add_argument('--variance', action='store_true', help='Run variance plots')
parser.add_argument('--shap', action='store_true', help='Run SHAP analysis')
parser.add_argument('--metrics', action='store_true', help='Run metrics heatmaps')
parser.add_argument('--bars', action='store_true')

args = parser.parse_args()

RUN_SANITY_CHECK = args.sanity
RUN_VARIANCE = args.variance
RUN_SHAP = args.shap
RUN_METRICS = args.metrics
RUN_BARS = args.bars

print(f"\nRunning analysis for Experiment 1a")
print(f"Sanity check: {RUN_SANITY_CHECK}")
print(f"Variance: {RUN_VARIANCE}")
print(f"SHAP: {RUN_SHAP}")
print(f"Metrics: {RUN_METRICS}")

# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1a')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'experiment_1a', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  CONFIGURATION
# ============================================================
N_CHUNKS = 5000
CHUNK_SIZE = 200
N_FEATURES = 10
WARMUP_WINDOWS = 10
SCORE_INTERVAL = 100
N_REPLICATIONS = 5

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

MF_CONFIGS = [('raw', 'Raw scores (v2.0)', MF_NAMES_RAW, 5),
    ('raw_temporal', 'Raw + temporal (v2.1)', MF_NAMES_RAW_TEMPORAL, 6)]

DRIFT_CONFIGS = [('sudden', 20, 9999, 21), ('gradual', 6, 5, 25)]

ABFS_MF_CONFIGS_FULL = [('aggstats', 'Aggregate stats (v1.1)'),
    ('raw', 'Raw scores (v2.0)'), ('raw_temporal', 'Raw + temporal (v2.1)')]

MEASURES = ['clustering', 'complexity', 'concept', 'general',
    'info-theory', 'itemset', 'landmarking', 'model-based', 'statistical']
STATISTICAL_IDX = MEASURES.index('statistical')

clf_names = [name for name, _ in BASE_CLFS]

ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS = {
    'aggstats':     'Aggstats (v1.1)',
    'raw':          'Raw scores (v2.0)',
    'raw_temporal': 'Raw + temporal (v2.1)',
}

palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', '#5e5151', '#08332b',
    '#000000', '#a9a9a9', '#ff69b4', '#00ced1', '#ff8c00'
]

SHAP_CLFS = [(name, clf) for name, clf in BASE_CLFS if name != 'SVM']

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
    return [i for i in range(1, n_chunks) if concept_labels_all[i] != concept_labels_all[i-1]]


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
        abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
            accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
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

            fig, ax = plt.subplots(figsize=(14, 4))
            for j in range(N_FEATURES):
                ax.plot(scores_over_time[:, j], label=f'f{j+1}')
            for b in boundaries:
                ax.axvline(x=b * CHUNK_SIZE // SCORE_INTERVAL, color='red',
                    linestyle='--', linewidth=1.0, alpha=0.7)
            ax.axvline(x=-1, color='red', linestyle='--', linewidth=1.0, label='concept boundary')
            ax.set_xlabel('Time (x100 instances)')
            ax.set_ylabel('Relevance score')
            ax.set_title(f'ABFS relevance scores - {drift_type} drift (seed={rs}) - experiment 1a')
            ax.legend(ncol=5, fontsize=8)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR, f'relevance_scores_{drift_type}_rep{rep_id}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"Relevance scores saved at: '{fname}'")

            for mf_type, mf_label, mf_names, n_mf_cols in MF_CONFIGS:
                X = mf_results[mf_type]['X']
                y = mf_results[mf_type]['y']
                unique_concepts = np.unique(y)

                fig, axes = plt.subplots(2, n_mf_cols, figsize=(4 * n_mf_cols, 6))
                axes = axes.flatten()
                for k, name in enumerate(mf_names):
                    axes[k].plot(X[:, k], color='steelblue')
                    for b in boundaries:
                        drift_w = b - WARMUP_WINDOWS
                        if drift_w > 0:
                            axes[k].axvline(x=drift_w, color='red', linestyle='--', linewidth=1.0)
                    axes[k].set_title(name, fontsize=9)
                    axes[k].set_xlabel('Window')
                    axes[k].set_ylabel('Value')
                fig.suptitle(f'Meta-features ({mf_type}) - {drift_type} drift (seed={rs}) - experiment 1a', fontsize=11)
                fig.tight_layout()
                fname = os.path.join(FIGURES_DIR, f'metafeatures_{mf_type}_{drift_type}_rep{rep_id}.png')
                fig.savefig(fname, dpi=150)
                plt.close()
                print(f"Meta-features per window saved at: '{fname}'")

                colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_concepts)}
                pca = PCA(n_components=2)
                projected = pca.fit_transform(X)
                fig, ax = plt.subplots(figsize=(8, 5))
                for c in unique_concepts:
                    mask = y == c
                    ax.scatter(projected[mask, 0], projected[mask, 1], color=colors[c],
                        label=f'concept {c}', alpha=0.6, edgecolors='none', s=30)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
                ax.set_title(f'PCA ({mf_type}) - {drift_type} drift (seed={rs}) - experiment 1a')
                ax.legend(ncol=4, fontsize=8)
                fig.tight_layout()
                fname = os.path.join(FIGURES_DIR, f'pca_{mf_type}_{drift_type}_rep{rep_id}.png')
                fig.savefig(fname, dpi=150)
                plt.close()
                print(f"PCA plot saved at: '{fname}'")

                print_sanity_check_summary(f'{drift_type} drift (seed={rs})', True,
                    mf_type, mf_names, X, y, X[:, :N_FEATURES], N_FEATURES)


# ============================================================
#  2. PERFORMANCE VARIANCE ACROSS REPLICATIONS
# ============================================================
if RUN_VARIANCE:
    print("\n" + "="*60)
    print("2. PERFORMANCE VARIANCE ACROSS REPLICATIONS")
    print("="*60)

    for drift_type, n_drifts, concept_sigmoid_spacing, n_concepts in DRIFT_CONFIGS:
        for mf_type, mf_label, mf_names, _ in MF_CONFIGS:

            path = os.path.join(RESULTS_DIR, f'clf_ba_{mf_type}_{drift_type}.npy')
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue

            clf_res = np.load(path) # (n_replications, n_folds, n_clfs)
            rep_means = np.mean(clf_res, axis=1) # (n_replications, n_clfs)
            rep_medians = np.median(clf_res, axis=1) # (n_replications, n_clfs)
            rep_stds = np.std(clf_res, axis=1) # (n_replications, n_clfs)

            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(N_REPLICATIONS)
            width = 0.15
            for clf_id, (name, _) in enumerate(BASE_CLFS):
                ax.bar(x + clf_id * width, rep_means[:, clf_id], width=width, label=name, alpha=0.6)
                ax.errorbar(x + clf_id * width, rep_means[:, clf_id], yerr=rep_stds[:, clf_id], fmt='none', color='black', capsize=3, linewidth=1)
                ax.scatter(x + clf_id * width, rep_medians[:, clf_id], marker='_', color='black', s=100, zorder=5)
            ax.axhline(y=1/n_concepts, color='red', linestyle='--', linewidth=1.0, label='random baseline')
            ax.set_xlabel('Replication')
            ax.set_ylabel('Mean balanced accuracy')
            ax.set_title(f'Performance variance - {mf_label} - {drift_type} drift - experiment 1a')
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels([f'Rep {i+1}\n(seed={RANDOM_STATES[i]})' for i in range(N_REPLICATIONS)], fontsize=8)
            ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR, f'variance_{mf_type}_{drift_type}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"Performance variance plot saved at: '{fname}'")



# ============================================================
#  3. SHAP ANALYSIS
# ============================================================
if RUN_SHAP:
    print("\n" + "="*60)
    print("3. SHAP ANALYSIS")
    print("="*60)

    # SHAP for the same batch classifiers as the 1a sweep, excluding SVM
    # (SVC has no predict_proba by default and is too slow under KernelExplainer).
    # 1a is fully batch, so SHAP explains the exact models used for the numbers.
    SHAP_CLFS = [(name, clf) for name, clf in BASE_CLFS if name != 'SVM']

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

            # shared background + evaluation samples across classifiers
            X_bg = shap.sample(X_all, 100)
            X_ev = shap.sample(X_all, 200)

            for clf_name, clf_proto in SHAP_CLFS:
                fname = os.path.join(FIGURES_DIR, f'shap_{mf_type}_{clf_name}_{drift_type}.png')
                if os.path.exists(fname):
                    print(f"  {clf_name}: exists, skipping"); continue
                print(f"  Classifier: {clf_name}")

                clf = clone(clf_proto)
                clf.fit(X_all, y_all)

                explainer = shap.KernelExplainer(clf.predict_proba, X_bg)
                shap_values = explainer.shap_values(X_ev, nsamples=100)

                shap_array = np.array(shap_values)
                if shap_array.ndim == 3:          # (n_samples, n_features, n_classes)
                    mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
                else:
                    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

                sorted_idx = np.argsort(mean_abs_shap)[::-1]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(mf_names)), mean_abs_shap[sorted_idx], color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(mf_names)))
                ax.set_xticklabels([mf_names[i] for i in sorted_idx], rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Mean absolute SHAP value')
                ax.set_title(f'SHAP feature importance - {mf_label} - {drift_type} drift - experiment 1a\n'
                             f'({clf_name}, pooled over {N_REPLICATIONS} replications)')
                fig.tight_layout()
                fig.savefig(fname, dpi=150)
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

            # Komorniczak baseline: for 1a use replication files (shuffled CV),
            rc_path = os.path.join(RESULTS_DIR, f'clf_replication_{metric}_{drift_type}.npy')

            if os.path.exists(rc_path):
                rc_raw  = np.load(rc_path)  # (n_measures, n_replications, n_folds, n_clfs)
                rc_mean = np.mean(rc_raw[STATISTICAL_IDX], axis=(0, 1))
                rc_std  = np.std(rc_raw[STATISTICAL_IDX],  axis=(0, 1))
            else:
                rc_mean = None
                print(f"Warning: {rc_path} not found")

            all_rows = []
            for mf_type, mf_display_label in ABFS_MF_CONFIGS_FULL:
                path = os.path.join(RESULTS_DIR, f'clf_{metric}_{mf_type}_{drift_type}.npy')
                if not os.path.exists(path):
                    print(f"Warning: {path} not found, skipping.")
                    continue
                raw = np.load(path)
                mean_vals = np.mean(raw,   axis=(0, 1))
                std_vals = np.std(raw,    axis=(0, 1))
                median_vals = np.median(raw, axis=(0, 1))
                all_rows.append((mf_display_label, mean_vals, std_vals, median_vals))

            if rc_mean is not None:
                rc_median = np.median(rc_raw[STATISTICAL_IDX], axis=(0, 1))
                all_rows.append(('Komorniczak (statistical)', rc_mean, rc_std, rc_median))
            if not all_rows:
                continue

            matrix = np.array([r[1] for r in all_rows])
            matrix_std = np.array([r[2] for r in all_rows])
            matrix_median = np.array([r[3] for r in all_rows])
            row_labels = [r[0] for r in all_rows]

            print(f"\n{metric_label} - {drift_type} drift - experiment 1a")
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

            fig, ax = plt.subplots(figsize=(10, max(3, len(all_rows) * 0.9)))
            im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
            
            for i in range(len(all_rows)):
                for j in range(len(BASE_CLFS)):
                    val = matrix[i, j]
                    std = matrix_std[i, j]
                    median = matrix_median[i, j]
                    txt_color = 'white' if val > 0.6 else 'black'
                    ax.text(j, i, f'{val:.3f}\n(±{std:.3f})\nmed:{median:.3f}', ha='center', va='center', fontsize=7, color=txt_color, linespacing=1.4)
            
            ax.set_xticks(range(len(BASE_CLFS)))
            ax.set_xticklabels(clf_names, fontsize=9)
            ax.set_yticks(range(len(all_rows)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_title(f'{metric_label} - {drift_type} drift ({n_concepts} concepts) - experiment 1a', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            fig.tight_layout()
            fname = os.path.join(FIGURES_DIR, f'heatmap_{metric}_{drift_type}.png')
            fig.savefig(fname, dpi=150)
            plt.close()
            print(f"Heatmap for metric {metric_label} saved at: '{fname}'")


# ============================================================
#  5. BARS - BA per ABFS version (best clf) + Komorniczak, per drift type
#  1a is batch CV. Shapes:
#    ABFS  clf_ba_{version}_{drift}.npy  -> (n_reps, n_folds, n_clfs)
#    Komor clf_replication_ba_{drift}.npy  -> (n_measures, n_reps, n_folds, n_clfs)
#  Score = mean over reps AND folds, then best classifier (and best measure for Komor).
# ============================================================
if RUN_BARS:
    print("ABFS_VERSIONS =", ABFS_VERSIONS)                # <-- ¿está vacío?
    print("DRIFT_CONFIGS =", DRIFT_CONFIGS)                # <-- ¿entra al bucle?

    print("\n" + "="*60)
    print("BA per version, per drift type (batch CV)")
    print("="*60)

    VERSION_COLORS = {'aggstats': '#911eb4', 'raw': '#4363d8', 'raw_temporal': '#f58231'}

    drifts, abfs_ba, komor_ba, baselines = [], {v: [] for v in ABFS_VERSIONS}, [], []
    
    print("RESULTS_DIR =", RESULTS_DIR)
    import glob
    print("npy encontrados:", glob.glob(os.path.join(RESULTS_DIR, 'clf_ba_*.npy')))

    for drift_type, n_drifts, css, n_concepts in DRIFT_CONFIGS:
        # ABFS: (n_reps, n_folds, n_clfs) -> mean over reps+folds -> (n_clfs,) -> best clf
        print("drift_type =", drift_type)
        per_version = {}
        any_ok = False
        for version in ABFS_VERSIONS:
            apath = os.path.join(RESULTS_DIR, f'clf_ba_{version}_{drift_type}.npy')
            if os.path.exists(apath):
                arr = np.load(apath)                      # (n_reps, n_folds, n_clfs)
                per_clf = np.mean(arr, axis=(0, 1))       # (n_clfs,)
                per_version[version] = float(np.nanmax(per_clf))
                any_ok = True
            else:
                per_version[version] = np.nan

        # Komorniczak: (n_measures, n_reps, n_folds, n_clfs) -> mean over reps+folds
        #              -> (n_measures, n_clfs) -> best measure & clf
        kpath = os.path.join(RESULTS_DIR, f'clf_replication_ba_{drift_type}.npy')
        kb = np.nan
        if os.path.exists(kpath):
            karr = np.load(kpath)                         # (n_measures, n_reps, n_folds, n_clfs)
            kmean = np.mean(karr, axis=(1, 2))            # (n_measures, n_clfs)
            kb = float(np.nanmax(kmean))

        if not any_ok and np.isnan(kb):
            continue
        drifts.append(drift_type)
        for version in ABFS_VERSIONS:
            abfs_ba[version].append(per_version[version])
        komor_ba.append(kb)
        baselines.append(1.0 / n_concepts)

    if not drifts:
        print("  no data -- skipping.")
    else:
        fname = os.path.join(FIGURES_DIR, 'ba_per_version.png')
        n_groups = len(drifts); n_bars = len(ABFS_VERSIONS) + 1
        width = 0.8 / n_bars; x = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(max(7, n_groups * 2.2), 5))
        for bi, version in enumerate(ABFS_VERSIONS):
            ax.bar(x + bi * width, abfs_ba[version], width,
                   color=VERSION_COLORS[version], label=f'ABFS {ABFS_LABELS[version]}')
        ax.bar(x + len(ABFS_VERSIONS) * width, komor_ba, width,
               color='#3cb44b', label='Komorniczak best-of-9')
        for gi in range(n_groups):
            ax.hlines(baselines[gi], x[gi] - width/2, x[gi] + n_bars*width - width/2,
                      color='red', linestyle='--', linewidth=1.0,
                      label='random baseline' if gi == 0 else None)
        ax.set_xticks(x + (n_bars - 1) * width / 2)
        ax.set_xticklabels([f'{d} drift' for d in drifts], fontsize=10)
        ax.set_ylabel('Balanced accuracy (mean over reps & folds, best clf)')
        ax.set_ylim(0, 1)
        ax.set_title('Exp 1a: BA per ABFS version vs Komorniczak (best classifier)')
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, axis='y')
        fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(); print(f"  Saved: {fname}")