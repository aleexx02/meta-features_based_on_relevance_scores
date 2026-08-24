# analysis_3.py
# ============================================================
# Analysis of Experiment 3 results (SEA / STAGGER / LED, chunk_size sweep).
#
# The evaluate script produces results per grid cell
# {gen}_chunk{cs}_{drift}, shape (n_reps, n_windows, n_clfs). This
# script reads those and produces figures.
#
# Per-cell plots (sanity / PCA / SHAP / stream_analysis) need ONE
# realization of a stream. Since Exp 3 streams are not saved to disk
# (regenerated per cell), this script regenerates the rep-0
# (SEED) realization of each cell via the builder in
# exp3_specs(). Performance / metrics / gap figures average over the
# rep axis.
#
# To keep the per-cell plot set manageable (24 cells), the sanity /
# SHAP / stream_analysis / performance / metrics / gap figures are
# produced only for a REFERENCE chunk_size (REF_CHUNK_SIZE) per
# (generator, drift) -- the chunk_size=200 cell, matching the rest of
# the project's default. The --grid flag is what spans the full
# chunk_size axis: it draws BA-vs-chunk_size sensitivity curves (ReMF
# raw vs Komorniczak best-of-9) using every cell.
#
# Concept label = generative concept id (exact).
#
# Flags:
#   --sanity           relevance_scores / metafeatures_{version} / pca_{version}
#   --performance      trajectory_ReMF / trajectory_komor   (mean over reps)
#   --shap             shap_all_clfs_{version}
#   --metrics          heatmap_f1 / heatmap_kappa            (mean over reps)
#   --stream_analysis  stream_drift_entropy / class_distribution (inline)
#   --gap              gap_heatmap_preq_exp3_{cell}          (mean over reps; fixed layout)
#   --grid             ba_vs_chunksize_{gen}_{drift}         (sensitivity curves)
#
# Usage:
#   python experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics --stream_analysis --gap --grid
# ============================================================

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
import csv

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
from streams.generate_synthetic_streams import exp3_specs, SEED, CHUNK_SIZES_EXP3 as CHUNK_SIZES
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
from plot_results import _plot_repr_comparison
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone as skclone
from sklearn.decomposition import PCA
import shap


# ============================================================
#  ARGS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sanity',          action='store_true')
parser.add_argument('--performance',     action='store_true')
parser.add_argument('--shap',            action='store_true')
parser.add_argument('--metrics',         action='store_true')
parser.add_argument('--gap',             action='store_true')
parser.add_argument('--stream_analysis', action='store_true')
parser.add_argument('--grid',            action='store_true')
parser.add_argument('--vanilla', action='store_true')
parser.add_argument('--summary',         action='store_true')
parser.add_argument('--concept_dist_features', action='store_true')
parser.add_argument('--concept_dist_metafeatures', action='store_true')
args = parser.parse_args()

EXP_TAG = 'exp3'
print(f"\nExperiment 3 analysis (SEA/STAGGER/LED, chunk_size sweep)")
print(f"sanity={args.sanity} performance={args.performance} shap={args.shap} "
      f"metrics={args.metrics} gap={args.gap} stream_analysis={args.stream_analysis} "
      f"grid={args.grid} summary={args.summary}")


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_3', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  CONFIG
# ============================================================
SPECS        = exp3_specs()
SPEC_BY_NAME = {s['name']: s for s in SPECS}

# Per-cell plots (sanity/SHAP/stream_analysis/performance/metrics) are
# generated for ALL chunk sizes, as in Experiment 2, so any cell can be
# chosen for the body later without re-running. The filenames already
# carry the chunk size, so nothing collides.
GEN_DRIFT_PAIRS = sorted({(s['gen_name'], s['transition']) for s in SPECS})
REF_CELLS = [s['name'] for s in SPECS]   # every cell, all chunk sizes

MEASURES = [
    'clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical',
]
ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {'aggstats': 'Aggstats', 'raw': 'Raw scores',
                 'raw_temporal': 'Raw + temporal'}
CLF_NAMES = [n for n, _ in BASE_CLFS_PREQUENTIAL]
N_CLFS    = len(CLF_NAMES)
CLF_COLORS = {'GNB': '#e6194b', 'KNN': '#3cb44b', 'HT': '#f032e6', 'MLP': '#911eb4'}
PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
           '#f032e6', '#808000', '#469990', '#9a6324', '#000075', '#800000']
SHAP_CLFS = [('GNB', GaussianNB()), ('KNN', KNeighborsClassifier()),
             ('HT', DecisionTreeClassifier(random_state=11313)),
             ('MLP', MLPClassifier(random_state=11313))]


# ============================================================
#  HELPERS
# ============================================================

def load(prefix, cell, optional=False):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{cell}.npy')
    if not os.path.exists(path):
        if not optional:
            print(f"  Warning: {path} not found.")
        return None
    return np.load(path)


def boundaries_from_cpc(cpc):
    return [int(i) for i in np.where(np.diff(cpc) != 0)[0] + 1]


def feat_names_for(version, n_features):
    if version == 'aggstats':
        return MF_NAMES_AGGSTATS
    if version == 'raw':
        return [f'r_f{j+1}' for j in range(n_features)]
    return [f'r_f{j+1}' for j in range(n_features)] + ['delta_mean', 'cosine_sim']


def regenerate_cell(cell):
    """Rebuild the rep-0 (SEED) realization of a cell."""
    spec = SPEC_BY_NAME[cell]
    data, cpc = spec['builder'](SEED)
    return data, cpc, spec


def re_extract(cell):
    """Re-run ReMF on the rep-0 realization -> scores, {version: X}, concept labels."""
    data, cpc, spec = regenerate_cell(cell)
    n_features = spec['n_features']; chunk_size = spec['chunk_size']
    X_full = data[:, :-1]; y_full = data[:, -1]
    abfs = ABFS_match(n_features=n_features, categorical_features=[],
                      accuracy_window_size=chunk_size, class_window_size=chunk_size)
    scores, mf, labels = [], {'aggstats': [], 'raw': [], 'raw_temporal': []}, []
    wt_prev = None
    for ci in range(len(cpc)):
        s = ci * chunk_size; e = s + chunk_size
        Xc, yc = X_full[s:e], y_full[s:e]
        for i in range(len(Xc)):
            abfs.update(Xc[i], yc[i])
        wt = abfs.relevance_scores(); dc = abfs.pop_drift_count(); ts = abfs.time_since_drift
        scores.append(wt)
        mf['aggstats'].append(extract_metafeatures(wt, wt_prev, dc, ts))
        mf['raw'].append(extract_metafeatures_raw(wt))
        mf['raw_temporal'].append(extract_metafeatures_raw_temporal(wt, wt_prev))
        labels.append(int(cpc[ci])); wt_prev = wt
    def clean(a):
        a = np.array(a, dtype=float); a[np.isnan(a)] = 0; a[np.isinf(a)] = 0; return a
    return np.array(scores), {v: clean(mf[v]) for v in ABFS_VERSIONS}, np.array(labels), cpc, spec


def stream_diagnostics(cell):
    data, cpc, spec = regenerate_cell(cell)
    chunk_size = spec['chunk_size']
    X_full = data[:, :-1]; y_full = data[:, -1].astype(int)
    n_classes = int(np.max(y_full)) + 1
    di, cd, le, prev = [], [], [], None
    for ci in range(len(cpc)):
        s = ci * chunk_size; e = s + chunk_size
        Xc, yc = X_full[s:e], y_full[s:e]
        mean = np.mean(Xc, axis=0)
        di.append(0.0 if prev is None else float(np.linalg.norm(mean - prev))); prev = mean
        counts = np.bincount(yc, minlength=n_classes); probs = counts / max(1, counts.sum())
        cd.append(probs); le.append(float(scipy_entropy(probs + 1e-10)))
    return np.array(di), np.array(cd), np.array(le), cpc, spec


def gap_heatmap(cell, gap_row, vmax, title):
    """Fixed gap heatmap: vertical colorbar on the right, taller strip,
    reserved bottom margin so the classifier labels are never clipped."""
    fname = os.path.join(FIGURES_DIR, f'gap_heatmap_preq_{EXP_TAG}_{cell}.png')
    if os.path.exists(fname):
        print(f"  Exists: {fname}"); return
    fig, ax = plt.subplots(figsize=(max(6, N_CLFS * 1.8), 2.8))
    im = ax.imshow(gap_row.reshape(1, -1), vmin=-vmax, vmax=vmax, cmap='RdBu', aspect='auto')
    for j in range(N_CLFS):
        val = gap_row[j]
        ax.text(j, 0, f'{val:+.3f}', ha='center', va='center', fontsize=12,
                color='white' if abs(val) > vmax * 0.6 else 'black')
    ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=11)
    ax.set_yticks([0]); ax.set_yticklabels([cell], fontsize=9)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.set_xlabel('Classifier', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=6)
    cbar.set_label('Gap (BA)', fontsize=9)
    fig.subplots_adjust(bottom=0.35)
    fig.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")

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
#  SANITY  (reference cells only)
# ============================================================
if args.sanity:
    print("\n" + "="*60); print("SANITY"); print("="*60)
    for cell in REF_CELLS:
        print(f"\n  {cell}")
        scores, X_by_version, y, cpc, spec = re_extract(cell)
        n_features = spec['n_features']
        n_concepts = len(np.unique(y)); rb = 1.0 / n_concepts
        boundaries = boundaries_from_cpc(cpc); unique_concepts = np.unique(y)

        fname = os.path.join(FIGURES_DIR, f'relevance_scores_{cell}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for j in range(n_features):
                ax.plot(scores[:, j], linewidth=0.7, alpha=0.8, label=f'f{j+1}')
            for b in boundaries:
                ax.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_xlabel('Window'); ax.set_ylabel('Relevance score')
            ax.set_title(f'Relevance scores -- {cell}\n({n_concepts} concepts, baseline={rb:.3f})')
            ax.legend(ncol=6, fontsize=6, loc='upper right')
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")

        for version in ABFS_VERSIONS:
            X = X_by_version[version]; names = feat_names_for(version, n_features); n_f = len(names)
            fname = os.path.join(FIGURES_DIR, f'metafeatures_{version}_{cell}.png')
            if not os.path.exists(fname):
                n_cols = 5; n_rows = (n_f + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
                axes_flat = np.array(axes).flatten()
                for k in range(n_f):
                    axes_flat[k].plot(X[:, k], color='steelblue', linewidth=0.8)
                    for b in boundaries:
                        axes_flat[k].axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
                    axes_flat[k].set_title(names[k], fontsize=8); axes_flat[k].set_xlabel('Window', fontsize=7)
                for k in range(n_f, len(axes_flat)):
                    axes_flat[k].set_visible(False)
                fig.suptitle(f'Meta-features ({ABFS_LABELS[version]}) -- {cell}', fontsize=10)
                fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")

            fname = os.path.join(FIGURES_DIR, f'pca_{version}_{cell}.png')
            if not os.path.exists(fname):
                colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique_concepts)}
                pca = PCA(n_components=2); proj = pca.fit_transform(X)
                fig, ax = plt.subplots(figsize=(8, 5))
                for c in unique_concepts:
                    m = y == c
                    ax.scatter(proj[m, 0], proj[m, 1], color=colors[c], label=f'concept {c}',
                               alpha=0.6, edgecolors='none', s=20)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
                ax.set_title(f'PCA -- {ABFS_LABELS[version]}\n{cell}'); ax.legend(ncol=4, fontsize=7)
                fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  PERFORMANCE  (reference cells, mean over reps)
# ============================================================
if args.performance:
    print("\n" + "="*60); print("PERFORMANCE"); print("="*60)
    for cell in REF_CELLS:
        spec = SPEC_BY_NAME[cell]
        _, cpc, _ = regenerate_cell(cell)
        n_concepts = spec['n_concepts']; rb = 1.0 / n_concepts
        boundaries = boundaries_from_cpc(cpc)
        print(f"\n  {cell}")

        fname = os.path.join(FIGURES_DIR, f'trajectory_ReMF_{cell}.png')
        if not os.path.exists(fname):
            fig, axes = plt.subplots(len(ABFS_VERSIONS), 1, figsize=(14, 4*len(ABFS_VERSIONS)), sharex=True)
            for ax, version in zip(axes, ABFS_VERSIONS):
                data = load(f'preq_abfs_{version}_ba', cell)
                if data is None:
                    ax.set_title(f'{ABFS_LABELS[version]} -- no data'); continue
                mt = np.mean(data, axis=0); x = np.arange(mt.shape[0])
                for cid, name in enumerate(CLF_NAMES):
                    ax.plot(x, mt[:, cid], label=name, color=CLF_COLORS.get(name, f'C{cid}'), linewidth=1.5)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.axhline(y=rb, color='red', linestyle='--', linewidth=1.0, label='baseline')
                ax.set_ylabel('Cumulative BA'); ax.set_title(ABFS_LABELS[version])
                ax.legend(fontsize=9, ncol=4); ax.set_ylim(0, 1)
            axes[-1].set_xlabel('Window')
            fig.suptitle(f'ReMF trajectories (mean over reps) -- {cell}\n({n_concepts} concepts, baseline={rb:.3f})', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")

        fname = os.path.join(FIGURES_DIR, f'trajectory_komor_{cell}.png')
        if not os.path.exists(fname):
            n_cols = 3; n_rows = (len(MEASURES) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3.5*n_rows), sharex=True, sharey=True)
            axes_flat = axes.flatten()
            for aid, measure in enumerate(MEASURES):
                ax = axes_flat[aid]; data = load(f'preq_komor_{measure}_ba', cell)
                if data is None:
                    ax.set_title(f'{measure} -- no data'); continue
                mt = np.mean(data, axis=0); x = np.arange(mt.shape[0])
                for cid, name in enumerate(CLF_NAMES):
                    ax.plot(x, mt[:, cid], label=name, color=CLF_COLORS.get(name, f'C{cid}'), linewidth=1.2)
                for b in boundaries:
                    ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.6, alpha=0.6)
                ax.axhline(y=rb, color='red', linestyle='--', linewidth=0.8)
                ax.set_title(measure, fontsize=10); ax.set_ylim(0, 1)
                if aid == 0:
                    ax.legend(fontsize=7, ncol=2)
            for aid in range(len(MEASURES), len(axes_flat)):
                axes_flat[aid].set_visible(False)
            fig.suptitle(f'Komorniczak trajectories (mean over reps) -- {cell}', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  SHAP  (reference cells)
# ============================================================
if args.shap:
    print("\n" + "="*60); print("SHAP"); print("="*60)
    for cell in REF_CELLS:
        print(f"\n  {cell}")
        y = load('concept_labels', cell)
        if y is None:
            print("  no concept labels -- skipping."); continue
        if all(os.path.exists(os.path.join(FIGURES_DIR, f'shap_all_clfs_{v}_{cell}.png')) for v in ABFS_VERSIONS):
            print("  all SHAP exist -- skipping."); continue
        _, X_by_version, _, _, spec = re_extract(cell)
        n_features = spec['n_features']
        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR, f'shap_all_clfs_{version}_{cell}.png')
            if os.path.exists(fname):
                continue
            X = X_by_version[version]; feat_names = feat_names_for(version, n_features)
            fig, axes = plt.subplots(2, 2, figsize=(16, 10)); axes_flat = axes.flatten()
            for cidx, (clf_name, proto) in enumerate(SHAP_CLFS):
                ax = axes_flat[cidx]; clf = skclone(proto); clf.fit(X, y)
                expl = shap.KernelExplainer(clf.predict_proba, shap.sample(X, 50))
                sv = expl.shap_values(shap.sample(X, 100), nsamples=50)
                arr = np.array(sv)
                mab = (np.mean(np.abs(arr), axis=(0, 2)) if arr.ndim == 3 else np.mean(np.abs(arr), axis=0))
                order = np.argsort(mab)[::-1]
                ax.bar(range(len(feat_names)), mab[order], color='steelblue', alpha=0.8)
                ax.set_xticks(range(len(feat_names)))
                ax.set_xticklabels([feat_names[i] for i in order], rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('Mean |SHAP|', fontsize=9); ax.set_title(clf_name, fontsize=11)
            fig.suptitle(f'SHAP -- {ABFS_LABELS[version]}\n{cell}', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  METRICS  (reference cells, mean over reps)
# ============================================================
if args.metrics:
    print("\n" + "="*60); print("METRICS"); print("="*60)
    for cell in REF_CELLS:
        spec = SPEC_BY_NAME[cell]; n_concepts = spec['n_concepts']
        print(f"\n  {cell}")
        for metric in ['f1', 'kappa']:
            fname = os.path.join(FIGURES_DIR, f'heatmap_{metric}_{cell}.png')
            if os.path.exists(fname):
                continue
            km = np.full((len(MEASURES), N_CLFS), np.nan)
            for mid, measure in enumerate(MEASURES):
                d = load(f'preq_komor_{measure}_{metric}', cell, optional=True)
                if d is not None:
                    km[mid, :] = np.mean(d[:, -1, :], axis=0)
            am = np.full((len(ABFS_VERSIONS), N_CLFS), np.nan)
            for vid, version in enumerate(ABFS_VERSIONS):
                d = load(f'preq_abfs_{version}_{metric}', cell, optional=True)
                if d is not None:
                    am[vid, :] = np.mean(d[:, -1, :], axis=0)
            ml = metric.upper() if metric == 'f1' else "Cohen's Kappa"
            fig, axes = plt.subplots(1, 2, figsize=(22, max(5, len(MEASURES) * 0.75)),
                                     gridspec_kw={'width_ratios': [3, 1.5]})
            ax = axes[0]; ax.imshow(km, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for i in range(len(MEASURES)):
                for j in range(N_CLFS):
                    v = km[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10,
                                color='white' if v > 0.6 else 'black')
            ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES, fontsize=10)
            ax.set_title(f'Komorniczak -- {ml}', fontsize=12)
            ax = axes[1]; im = ax.imshow(am, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for i in range(len(ABFS_VERSIONS)):
                for j in range(N_CLFS):
                    v = am[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10,
                                color='white' if v > 0.6 else 'black')
            ax.set_xticks(range(N_CLFS)); ax.set_xticklabels(CLF_NAMES, fontsize=10)
            ax.set_yticks(range(len(ABFS_VERSIONS))); ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
            ax.set_title(f'ReMF -- {ml}', fontsize=12)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle(f'{ml} (mean over reps) -- {cell}\nfinal window | baseline={1/n_concepts:.3f}', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


if args.stream_analysis:
    print("\n" + "="*60); print("STREAM ANALYSIS"); print("="*60)
    for cell in REF_CELLS:
        f_drift = os.path.join(FIGURES_DIR, f'stream_drift_entropy_{cell}.png')
        f_class = os.path.join(FIGURES_DIR, f'class_distribution_{cell}.png')
        if os.path.exists(f_drift) and os.path.exists(f_class):
            print(f"  {cell}: figures exist -- skipping"); continue

        print(f"\n  {cell}")
        di, cd, le, cpc, spec = stream_diagnostics(cell)
        boundaries = boundaries_from_cpc(cpc)

        # ReMF re-extraction is the expensive step -- only do it if the drift
        # figure (the one that needs it) is missing.
        if not os.path.exists(f_drift):
            scores, _, _, _, _ = re_extract(cell)
            dr = np.linalg.norm(scores[1:] - scores[:-1], axis=1)
            dr = np.concatenate([[0], dr]); dr = dr / (np.max(dr) + 1e-10)
            di_n = di / (np.max(di) + 1e-10)
            fig, ax1 = plt.subplots(figsize=(14, 4))
            ax1.plot(di_n, color='steelblue', label='Drift intensity', linewidth=1.5)
            ax1.plot(dr, color='purple', label='ReMF relevance change', linewidth=1.2, alpha=0.7)
            ax1.set_ylabel('Normalized value')
            ax2 = ax1.twinx(); ax2.plot(le, color='darkorange', label='Label entropy', alpha=0.7)
            ax2.set_ylabel('Entropy', color='darkorange')
            for b in boundaries:
                ax1.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax1.set_xlabel('Window')
            l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lab1 + lab2, loc='upper right')
            ax1.set_title(f'Drift vs ReMF dynamics -- {cell}')
            fig.tight_layout(); fig.savefig(f_drift, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {f_drift}")

        if not os.path.exists(f_class):
            fig, ax = plt.subplots(figsize=(14, 4))
            for c in range(cd.shape[1]):
                ax.plot(cd[:, c], label=f'class {c}', linewidth=1.2)
            for b in boundaries:
                ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.set_xlabel('Window'); ax.set_ylabel('Proportion')
            ax.set_title(f'Class distribution over time -- {cell}'); ax.legend(ncol=4, fontsize=8)
            fig.tight_layout(); fig.savefig(f_class, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {f_class}")

            
# ============================================================
#  GAP HEATMAP  (reference cells, fixed layout, mean over reps)
# ============================================================
if args.gap:
    print("\n" + "="*60); print("GRID GAP HEATMAP (per ReMF version)"); print("="*60)
    GENERATORS = sorted({g for (g, d) in GEN_DRIFT_PAIRS})
    DRIFTS     = sorted({d for (g, d) in GEN_DRIFT_PAIRS})

    def best_abfs_v(cell, version):
        pr = load(f'preq_abfs_{version}_ba', cell, optional=True)
        return None if pr is None else float(np.max(np.mean(pr[:, -1, :], axis=0)))

    def best_komor(cell):
        kb = None
        for measure in MEASURES:
            d = load(f'preq_komor_{measure}_ba', cell, optional=True)
            if d is None:
                continue
            v = float(np.max(np.mean(d[:, -1, :], axis=0)))
            kb = v if kb is None else max(kb, v)
        return kb

    for drift in DRIFTS:
        grids = {}
        for version in ABFS_VERSIONS:
            g = np.full((len(GENERATORS), len(CHUNK_SIZES)), np.nan)
            for ri, gen in enumerate(GENERATORS):
                for ci, cs in enumerate(CHUNK_SIZES):
                    cell = f'{gen}_chunk{cs}_{drift}'
                    a = best_abfs_v(cell, version); k = best_komor(cell)
                    if a is not None and k is not None:
                        g[ri, ci] = a - k
            grids[version] = g

        finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()]) \
                 if any(np.any(np.isfinite(g)) for g in grids.values()) else np.array([])
        if finite.size == 0:
            print(f"  {drift}: no data -- skipping."); continue
        vmax = float(np.max(np.abs(finite))) or 1.0

        for version in ABFS_VERSIONS:
            g = grids[version]
            if not np.any(np.isfinite(g)):
                continue
            fname = os.path.join(FIGURES_DIR, f'gap_grid_{version}_{drift}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            im = ax.imshow(g, vmin=-vmax, vmax=vmax, cmap='RdBu', aspect='auto')
            for ri in range(len(GENERATORS)):
                for ci in range(len(CHUNK_SIZES)):
                    v = g[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f'{v:+.3f}', ha='center', va='center', fontsize=10,
                                color='white' if abs(v) > vmax * 0.6 else 'black')
            ax.set_xticks(range(len(CHUNK_SIZES))); ax.set_xticklabels(CHUNK_SIZES)
            ax.set_yticks(range(len(GENERATORS))); ax.set_yticklabels(GENERATORS)
            ax.set_xlabel('chunk_size'); ax.set_ylabel('generator')
            ax.set_title(f'Gap (best ReMF {ABFS_LABELS[version]} minus best Komorniczak)\n{drift}')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Gap (BA)')
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  GRID  - BA vs chunk_size sensitivity curves (per gen, drift)
#  ReMF: one curve per version (best clf) + Komorniczak best-of-9
# ============================================================
if args.grid:
    print("\n" + "="*60); print("GRID: BA vs chunk_size (one figure per ReMF version)"); print("="*60)

    def komor_best_per_clf(cell):
        """Best Komorniczak group per classifier (mean over reps, final window)."""
        best = None
        for measure in MEASURES:
            d = load(f'preq_komor_{measure}_ba', cell, optional=True)
            if d is None:
                continue
            per_clf = np.mean(d[:, -1, :], axis=0)
            best = per_clf if best is None else np.nanmax(np.vstack([best, per_clf]), axis=0)
        return best

    for (gen, drift) in GEN_DRIFT_PAIRS:
        for version in ABFS_VERSIONS:
            fname = os.path.join(FIGURES_DIR,
                                 f'ba_vs_chunksize_{version}_{gen}_{drift}.png')
            if os.path.exists(fname):
                print(f"  Exists: {fname}"); continue

            xs = []
            abfs_clf  = {name: [] for name in CLF_NAMES}
            komor_clf = {name: [] for name in CLF_NAMES}

            for cs in CHUNK_SIZES:
                cell = f'{gen}_chunk{cs}_{drift}'
                pr = load(f'preq_abfs_{version}_ba', cell, optional=True)
                kb = komor_best_per_clf(cell)
                if pr is None or kb is None:
                    continue
                abfs_per_clf = np.mean(pr[:, -1, :], axis=0)
                xs.append(cs)
                for ci, name in enumerate(CLF_NAMES):
                    abfs_clf[name].append(abfs_per_clf[ci])
                    komor_clf[name].append(kb[ci])

            if not xs:
                print(f"  {gen}_{drift} [{version}]: no data -- skipping."); continue

            fig, ax = plt.subplots(figsize=(8, 5))
            for ci, name in enumerate(CLF_NAMES):
                color = CLF_COLORS.get(name, f'C{ci}')
                ax.plot(xs, abfs_clf[name], 'o-', color=color,
                        label=f'{name} ReMF', linewidth=1.5, markersize=5)
                ax.plot(xs, komor_clf[name], 's--', color=color,
                        label=f'{name} Komor', linewidth=1.5, markersize=5)
            ax.set_xscale('log'); ax.set_xticks(CHUNK_SIZES); ax.set_xticklabels(CHUNK_SIZES)
            ax.set_xlabel('chunk_size'); ax.set_ylabel('Final balanced accuracy (mean over reps)')
            ax.set_title(f'BA vs chunk_size -- ReMF {ABFS_LABELS[version]} vs Komorniczak '
                         f'-- {gen}, {drift}')
            ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


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

    rows = []
    for cell in REF_CELLS:              # or a chosen subset for the report table
        v, a, k = vanilla_row(cell)
        if v is None:
            print(f"  {cell}: no vanilla results -- skipping"); continue
        rows.append((cell, v, a, k))
        print(f"  {cell:35s}  vanilla={v:.3f}  ReMF={a:.3f}  komor={k:.3f}")

    # write a CSV so the report table can be built from it
    import csv
    rows_csv = []
    for cell, v, a, k in rows:
        rows_csv.append(dict(
            drift_type=cell,
            Vanilla_BA=v,
            ReMF_best_BA=a,
            Komorniczak_best_BA=k
        ))

    out = os.path.join(RESULTS_DIR, 'vanilla_comparison_exp3.csv')
    write_dict_csv(out, rows_csv)
    print(f"\n  Saved: {out}")



# ---- combined summary helpers (ReMF headers, per-classifier BA + cost) ----
def _per_clf_best_over(load_fn, keys, has_reps):
    """For each classifier, best final-window BA over (label, prefix) keys, and
    which label achieved it. Returns (ba_vec, lab_vec); (None,'') where absent."""
    best_ba = np.full(N_CLFS, -1.0)
    best_lab = [None] * N_CLFS
    for label, prefix in keys:
        v = _final_per_clf(load_fn(prefix), has_reps=has_reps)
        if v is None:
            continue
        for j in range(N_CLFS):
            if np.isfinite(v[j]) and v[j] > best_ba[j]:
                best_ba[j] = float(v[j]); best_lab[j] = label
    out_ba = [b if lab is not None else None for b, lab in zip(best_ba, best_lab)]
    out_lab = [lab if lab is not None else '' for lab in best_lab]
    return out_ba, out_lab


def _read_cost_lookup(path):
    """extraction_cost_expN.csv -> {(method, n_features): {time_s, ms_per_window,
    peak_mb}}. European format (';' + comma decimal). Empty if file absent."""
    lut = {}
    if not path or not os.path.exists(path):
        print(f"  [cost] {path} not found -- cost columns left blank.")
        return lut
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            try:
                method = (row.get('method') or '').strip()
                nf = int(float((row.get('n_features') or '').replace(',', '.')))
            except (TypeError, ValueError):
                continue

            def g(k):
                v = (row.get(k) or '').strip().replace(',', '.')
                try:
                    return float(v)
                except ValueError:
                    return None
            lut[(method, nf)] = dict(time_s=g('time_s'),
                                     ms_per_window=g('ms_per_window'),
                                     peak_mb=g('peak_mb'))
    return lut


def write_combined_summary(specs, results_dir, out_name, meta_of,
                           has_reps=True, cost_name=None):
    """Combined per-cell summary CSV. meta_of(spec) -> ordered [(col, val), ...].
    cost_name: extraction-cost CSV filename in results_dir, or None for blank."""
    ReMF_KEYS   = [(ABFS_LABELS[v], f'preq_abfs_{v}_ba') for v in ABFS_VERSIONS]
    KOMOR_KEYS = [(m, f'preq_komor_{m}_ba') for m in MEASURES]

    cost_path = os.path.join(results_dir, cost_name) if cost_name else None
    cost = _read_cost_lookup(cost_path)

    global_cols = [
        'ReMF best (repr / clf)',         'ReMF best BA',
        'Komorniczak best (group / clf)', 'Komorniczak best BA',
        'vanilla best (clf)',             'vanilla best BA',
        'gap ReMF - Komorniczak (best vs best)',
    ]
    cost_cols = [
    'ReMF time (ms/win)',
    'ReMF peak MB (rel.)',

    'Komorniczak time (ms/win)',
    'Komorniczak peak MB (rel.)',

    'Vanilla time (ms/win)',
    'Vanilla peak MB (rel.)',
    ]
    perclf_cols = []
    for clf in CLF_NAMES:
        perclf_cols += [
            f'ReMF {clf} BA',        f'ReMF {clf} representation',
            f'Komorniczak {clf} BA', f'Komorniczak {clf} group',
            f'vanilla {clf} BA',
        ]

    header, rows = None, []
    for spec in specs:
        cell = spec['name']
        loadf = lambda prefix, c=cell: load(prefix, c, optional=True)

        ab = best_side(loadf, ReMF_KEYS,   has_reps)
        kb = best_side(loadf, KOMOR_KEYS, has_reps)
        if ab[0] is None or kb[0] is None:
            print(f"  {cell}: no ReMF/Komorniczak results -- skipping")
            continue
        vb_vec = _final_per_clf(loadf('preq_vanilla_ba'), has_reps)
        if vb_vec is not None:
            vj = int(np.nanargmax(vb_vec)); vb_clf, vb_ba = CLF_NAMES[vj], float(vb_vec[vj])
        else:
            vb_clf, vb_ba = '', None

        remf_ba, remf_rep = _per_clf_best_over(loadf, ReMF_KEYS,   has_reps)
        kom_ba, kom_grp = _per_clf_best_over(loadf, KOMOR_KEYS, has_reps)
        van_ba = vb_vec if vb_vec is not None else [None] * N_CLFS

        nf = spec.get('n_features')
        c_remf = cost.get(('ReMF', nf), {})
        c_kom = cost.get(('komorniczak', nf), {})
        c_van = cost.get(('Vanilla', nf), {})

        meta = meta_of(spec)
        if header is None:
            header = [c for c, _ in meta] + global_cols + cost_cols + perclf_cols

        row = [v for _, v in meta] + [
            f'{ab[0]} / {ab[1]}', ab[2],
            f'{kb[0]} / {kb[1]}', kb[2],
            vb_clf, vb_ba,
            ab[2] - kb[2],
            c_remf.get('ms_per_window'), c_remf.get('peak_mb'),
            c_kom.get('ms_per_window'), c_kom.get('peak_mb'),
            c_van.get('ms_per_window'), c_van.get('peak_mb')
        ]
        for j in range(N_CLFS):
            row += [remf_ba[j], remf_rep[j], kom_ba[j], kom_grp[j],
                    (float(van_ba[j]) if van_ba[j] is not None else None)]
        rows.append(row)
        print(f"  {cell:34s} ReMF {ab[2]:.3f} ({ab[0]}/{ab[1]})  "
              f"Komor {kb[2]:.3f} ({kb[0]}/{kb[1]})  "
              f"ReMF cost {c_remf.get('ms_per_window','NA')} ms/win")

    if header is None:
        print("  No cells with results -- nothing written."); return
    write_summary_csv(os.path.join(results_dir, out_name), header, rows)


if args.summary:
    print("\n" + "="*60); print("SUMMARY TABLE (combined + cost)"); print("="*60)
    from streams.generate_synthetic_streams import TOTAL_INSTANCES

    def meta_of(spec):
        n_seg = len(spec['order'])
        return [
            ('stream', spec['gen_name']),
            ('n_features', spec['n_features']),
            ('drift_type', spec['transition']),
            ('chunk_size', spec['chunk_size']),
            ('n_segments', n_seg),
            ('n_concepts', spec['n_concepts']),
            ('random_baseline', 1.0 / spec['n_concepts']),
            ('n_instances', TOTAL_INSTANCES),
            ('instances_per_segment', TOTAL_INSTANCES // n_seg),
        ]

    write_combined_summary(SPECS, RESULTS_DIR, 'summary_exp3.csv', meta_of,
                           has_reps=True, cost_name='extraction_cost_exp3.csv')
    


if args.concept_dist_metafeatures:
    import itertools
    print("\n" + "=" * 60)
    print("CONCEPT SEPARATION IN META-FEATURE SPACE (rep-0; vanilla/komor/ReMF)")
    print("=" * 60)

    KEEP_CHUNK_SIZE = 100
    REP_SEED = SEED                      # single rep = rep-0
    BEST_KOMOR_FAMILY = {
        "led_chunk100_gradual": "general",
        "led_chunk100_sudden": "general",

        "sea_chunk100_gradual": "statistical",
        "sea_chunk100_sudden": "statistical",

        "stagger_chunk100_gradual": "complexity",
        "stagger_chunk100_sudden": "complexity",
    }

    BEST_REMF_VARIANT = {
        "led_chunk100_gradual": "aggstats",
        "led_chunk100_sudden": "aggstats",

        "sea_chunk100_gradual": "aggstats",
        "sea_chunk100_sudden": "aggstats",

        "stagger_chunk100_gradual": "aggstats",
        "stagger_chunk100_sudden": "aggstats",
    }

    STREAM_DIR = os.path.join(FIGURES_DIR, '..', 'streams')
    os.makedirs(STREAM_DIR, exist_ok=True)

    # Komorniczak per-(cell,measure,seed) pymfe cache, written by
    # evaluate_concept_classification_3.py. Same path construction as there.
    KOMOR_CACHE_DIR = os.path.join(PROJECT_ROOT, 'external', 'komorniczak',
                                   'results', 'synthetic_sea_stagger_led')

    def _load_komor_cache(cell, measure, seed):
        """Return the cached per-window pymfe matrix for one family, last
        column dropped (it is the concept id), or (None, None) if absent."""
        path = os.path.join(KOMOR_CACHE_DIR,
                            f'komor_{cell}_{measure}_seed{seed}.npy')
        if not os.path.exists(path):
            return None, None
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None, None
        X = arr[:, :-1].astype(float)
        y = arr[:, -1].astype(int)
        X[np.isnan(X)] = 0; X[np.isinf(X)] = 0
        return X, y

    # ---- local L2 + fixed distance-matrix plotter (masked diagonal) ---------
    def _pairwise_l2(cm):
        n = len(cm); D = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            d = float(np.linalg.norm(cm[i] - cm[j]))
            D[i, j] = D[j, i] = d
        return D

    # ---- rep-0 fingerprint sources ------------------------------------------
    def _remf_and_raw(cell, seed):
        """ReMF matrices {version: X}, per-window concept labels, raw windows."""
        spec = SPEC_BY_NAME[cell]
        data, cpc = spec['builder'](seed)
        n_features = spec['n_features']; chunk_size = spec['chunk_size']
        X_full = data[:, :-1]; y_full = data[:, -1]
        abfs = ABFS_match(n_features=n_features, categorical_features=[],
                          accuracy_window_size=chunk_size, class_window_size=chunk_size)
        mf = {'aggstats': [], 'raw': [], 'raw_temporal': []}
        raw_windows, labels = [], []
        wt_prev = None
        for ci in range(len(cpc)):
            s = ci * chunk_size; e = s + chunk_size
            Xc, yc = X_full[s:e], y_full[s:e]
            for i in range(len(Xc)):
                abfs.update(Xc[i], yc[i])
            wt = abfs.relevance_scores(); dc = abfs.pop_drift_count(); ts = abfs.time_since_drift
            mf['aggstats'].append(extract_metafeatures(wt, wt_prev, dc, ts))
            mf['raw'].append(extract_metafeatures_raw(wt))
            mf['raw_temporal'].append(extract_metafeatures_raw_temporal(wt, wt_prev))
            raw_windows.append(Xc); labels.append(int(cpc[ci])); wt_prev = wt

        def clean(a):
            a = np.array(a, dtype=float); a[np.isnan(a)] = 0; a[np.isinf(a)] = 0; return a
        return {v: clean(mf[v]) for v in ABFS_VERSIONS}, np.array(labels), raw_windows

    def _vanilla_matrix(raw_windows):
        feats = [np.concatenate([np.asarray(w, float).mean(axis=0),
                                 np.asarray(w, float).std(axis=0)])
                 for w in raw_windows]
        X = np.array(feats, dtype=float)
        X[np.isnan(X)] = 1; X[np.isinf(X)] = 1
        return X

    def _raw_feature_matrix(raw_windows):
        return np.array([np.asarray(w, float).mean(axis=0) for w in raw_windows])

    def _concept_means(X, labels):
        ids = sorted(np.unique(labels).tolist())
        cm = np.array([X[labels == c].mean(axis=0) for c in ids])
        return ids, cm

    # ---- main loop (single rep) ---------------------------------------------
    rows_summary = []
    for spec in SPECS:
        cell = spec['name']
        if KEEP_CHUNK_SIZE is not None and spec['chunk_size'] != KEEP_CHUNK_SIZE:
            continue

        Xbv, labels, raw_windows = _remf_and_raw(cell, REP_SEED)

        method_mats = {}
        for v in ABFS_VERSIONS:
            ids, cm = _concept_means(Xbv[v], labels)
            method_mats[f'ReMF_{v}'] = (ids, _pairwise_l2(cm))
        ids, cm = _concept_means(_vanilla_matrix(raw_windows), labels)
        method_mats['vanilla'] = (ids, _pairwise_l2(cm))
        ids, cm = _concept_means(_raw_feature_matrix(raw_windows), labels)
        method_mats['raw_features'] = (ids, _pairwise_l2(cm))
        raw_ref = method_mats['raw_features']

        # Komorniczak from cache (best family for this cell)
        komor_family = BEST_KOMOR_FAMILY.get(cell)
        Xk = yk = None
        if komor_family is None:
            print(f"  [komor] no best-family mapping for {cell} -- skipping komor.")
        else:
            Xk, yk = _load_komor_cache(cell, komor_family, REP_SEED)
            if Xk is None:
                print(f"  [komor] cache miss for {cell} / {komor_family} "
                      f"seed{REP_SEED} -- skipping komor for this cell.")
            else:
                if len(yk) != len(labels):
                    print(f"  [komor] window-count mismatch for {cell} "
                          f"(komor {len(yk)} vs remf {len(labels)}) "
                          f"-- using komor's own labels.")
                ids, cm = _concept_means(Xk, yk)
                method_mats[f'komorniczak_{komor_family}'] = (ids, _pairwise_l2(cm))

        # record numbers for every method (no per-method PNGs any more)
        for method, (ids_m, D) in method_mats.items():
            off = D[~np.eye(len(D), dtype=bool)]
            l2_mean = float(off.mean()) if off.size else 0.0
            l2_min = float(off.min()) if off.size else 0.0
            l2_max = float(off.max()) if off.size else 0.0
            rows_summary.append(dict(cell=cell, method=method,
                                     n_concepts=len(ids_m), rep=int(REP_SEED),
                                     l2_min=round(l2_min, 4),
                                     l2_max=round(l2_max, 4),
                                     l2_mean=round(l2_mean, 4)))
            print(f"  {cell:28s} {method:22s} mean L2 {l2_mean:.4f}")

        # ---- repr_comparison: Raw Features vs Best Komorniczak vs Best ReMF --
        best_remf = BEST_REMF_VARIANT.get(cell)
        best_komor = BEST_KOMOR_FAMILY.get(cell)
        remf_key = f"ReMF_{best_remf}"
        komor_key = f"komorniczak_{best_komor}"

        if remf_key not in method_mats:
            print(f"  missing {remf_key} for {cell} -- skipping repr_comparison")
            continue
        if komor_key not in method_mats:
            print(f"  missing {komor_key} for {cell} -- skipping repr_comparison")
            continue

        panels = [
            (raw_ref[1], raw_ref[0], "Raw Features"),
            (method_mats[komor_key][1], method_mats[komor_key][0],
             f"Komorniczak ({best_komor})"),
            (method_mats[remf_key][1], method_mats[remf_key][0],
             f"ReMF ({best_remf})"),
        ]
        _plot_repr_comparison(
            panels, cell,
            os.path.join(STREAM_DIR, f"repr_comparison_{cell}.png"))
        print(f"  {cell:28s} repr_comparison -> repr_comparison_{cell}.png")

    if rows_summary:
        out = os.path.join(RESULTS_DIR, 'concept_distance_metafeatures_exp3.csv')
        write_dict_csv(out, rows_summary)
        print(f"  Saved: {out}")

        

if args.concept_dist_features:
    print("\n" + "=" * 60)
    print("CONCEPT SEPARATION IN FEATURE SPACE")
    print("=" * 60)
 
    KEEP_CHUNK_SIZE = 100
    EXP = EXP_TAG
 
    # Replication seeds. Prefer the project's own list so the characterisation
    # describes the same streams that were evaluated; fall back to a derived
    # list if the module does not export one.
    try:
        from streams.generate_synthetic_streams import RANDOM_STATES as REP_SEEDS
        REP_SEEDS = [int(s) for s in REP_SEEDS]
        print(f"  seeds: RANDOM_STATES = {REP_SEEDS}")
    except ImportError:
        np.random.seed(SEED)
        REP_SEEDS = [int(SEED)] + [int(s) for s in
                                   np.random.randint(100, 10000, 4)]
        print(f"  seeds: derived from SEED = {REP_SEEDS}  "
              f"(module exports no RANDOM_STATES)")
 
    rows_means, rows_dist, rows_summary = [], [], []
 
    for spec in SPECS:
        cell = spec['name']
        cs = spec['chunk_size']
        if KEEP_CHUNK_SIZE is not None and cs != KEEP_CHUNK_SIZE:
            continue
 
        per_rep_pairs = {}       # (a, b) -> [l2 per rep]
        cm_rep0, uniq_rep0 = None, None
 
        for rep, seed in enumerate(REP_SEEDS):
            data, cpc = spec['builder'](seed)
            X = data[:, :-1].astype(float)
 
            labels_all = np.asarray(cpc)          # concept label PER WINDOW
            n_win = min(len(X) // cs, len(labels_all))
            if n_win == 0:
                continue
 
            means = np.array([X[i * cs:(i + 1) * cs].mean(axis=0)
                              for i in range(n_win)])
            concs = labels_all[:n_win]
            uniq = np.unique(concs)
            cm = np.array([means[concs == c].mean(axis=0) for c in uniq])
 
            if rep == 0:
                cm_rep0, uniq_rep0 = cm, uniq   # fingerprints: rep 0, see note
 
            d = []
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    l2 = float(np.linalg.norm(cm[i] - cm[j]))
                    d.append(l2)
                    per_rep_pairs.setdefault((int(uniq[i]), int(uniq[j])),
                                             []).append(l2)
            
        if not per_rep_pairs:
            print(f"  {cell}: no usable reps -- skipped")
            continue

        pair_means = np.array(
            [np.mean(vals) for vals in per_rep_pairs.values()],
            dtype=float
        )

        l2_min = float(np.min(pair_means))
        l2_max = float(np.max(pair_means))
        l2_mean = float(np.mean(pair_means))
        l2_mean_std = float(np.std(pair_means))

        print(f"  {cell:34s} n_feat={cm_rep0.shape[1]} "
            f"n_conc={len(uniq_rep0):2d}  L2 mean={l2_mean:.4f} "
            f"+/-{l2_mean_std:.4f}  (min={l2_min:.4f} max={l2_max:.4f}, "
            f"{len(REP_SEEDS)} reps)")

        rows_summary.append(dict(
            cell=cell,
            n_concepts=len(uniq_rep0),
            n_reps=len(REP_SEEDS),
            l2_min=round(l2_min, 4),
            l2_max=round(l2_max, 4),
            l2_mean=round(l2_mean, 4),
            l2_mean_std=round(l2_mean_std, 4)
        ))
 
        for (a, b), vals in sorted(per_rep_pairs.items()):
            rows_dist.append(dict(cell=cell, concept_a=a, concept_b=b,
                                  l2=round(float(np.mean(vals)), 4),
                                  l2_std=round(float(np.std(vals)), 4),
                                  n_reps=len(vals)))
 
        # Fingerprints come from rep 0 only, on purpose: for stream-learn the
        # concept layout differs per seed, so a cross-rep average would not be
        # a meaningful fingerprint. The distances above are the averaged part.
        for c, m_ in zip(uniq_rep0, cm_rep0):
            rows_means.append(dict(cell=cell, concept=int(c),
                                   **{f'f{k}': round(float(v), 4)
                                      for k, v in enumerate(m_)}))
 
    for rows, name in [
    (rows_means, 'concept_feature_means'),
    (rows_dist, 'concept_distances'),
    (rows_summary, 'concept_distance_summary')
    ]:
        out = os.path.join(RESULTS_DIR, f'{name}_exp3.csv')
        write_dict_csv(out, rows)
        print(f"  Saved: {out}")
 

print("\nAnalysis 3 complete.")