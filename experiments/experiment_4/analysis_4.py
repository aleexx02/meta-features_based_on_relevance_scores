# analysis_4.py
# ============================================================
# Analysis of Experiment 4 results (recurring SEA / STAGGER, chunk_size x n_drifts grid).
#
# The evaluate script produces results per grid cell
# {gen}_chunk{cs}_{drift}, shape (n_reps, n_windows, n_clfs). This
# script reads those and produces figures.
#
# Per-cell plots (sanity / PCA / SHAP / stream_analysis) need ONE
# realization of a stream. Since Exp 3 streams are not saved to disk
# (regenerated per cell), this script regenerates the rep-0
# (SEED) realization of each cell via the builder in
# exp4_specs(). Performance / metrics / gap figures average over the
# rep axis.
#
# To keep the per-cell plot set manageable (64 cells), the sanity /
# SHAP / stream_analysis / performance / metrics / gap figures are
# produced only for a REFERENCE cell (REF_CHUNK_SIZE, REF_N_DRIFTS) per
# (generator, drift) -- the chunk_size=200 cell, matching the rest of
# the project's default. The --grid flag is what spans the full
# chunk_size axis: it draws BA-vs-chunk_size sensitivity curves (ABFS
# raw v2.0 vs Komorniczak best-of-9) using every cell.
#
# Concept label = generative concept id (exact).
#
# Flags:
#   --sanity           relevance_scores / metafeatures_{version} / pca_{version}
#   --performance      trajectory_abfs / trajectory_komor   (mean over reps)
#   --shap             shap_all_clfs_{version}
#   --metrics          heatmap_f1 / heatmap_kappa            (mean over reps)
#   --stream_analysis  stream_drift_entropy / class_distribution (inline)
#   --gap              gap_heatmap_preq_exp4_{cell}          (mean over reps; fixed layout)
#   --grid             gap_grid_{gen}_{drift} (cs x n_drifts gap heatmap) +
#                      ba_vs_ndrifts_{gen}_chunk{cs}_{drift} curves
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
from streams.generate_synthetic_streams import exp4_specs, SEED, CHUNK_SIZES_EXP4 as CHUNK_SIZES
from classifier_sweep_prequential import BASE_CLFS_PREQUENTIAL
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
args = parser.parse_args()

EXP_TAG = 'exp4'
print(f"\nExperiment 4 analysis (recurring SEA/STAGGER, chunk_size x n_drifts grid)")
print(f"sanity={args.sanity} performance={args.performance} shap={args.shap} "
      f"metrics={args.metrics} gap={args.gap} stream_analysis={args.stream_analysis} "
      f"grid={args.grid}")


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_4', 'figures', 'analysis')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  CONFIG
# ============================================================
SPECS        = exp4_specs()
SPEC_BY_NAME = {s['name']: s for s in SPECS}
REF_CHUNK_SIZE = 200   # per-cell plots use this chunk_size...
REF_N_DRIFTS   = 7     # ...and this n_drifts (recurrence present) per (gen, drift)
from streams.generate_synthetic_streams import EXP4_N_DRIFTS

# (gen, drift) pairs, and the reference cell name for each
GEN_DRIFT_PAIRS = sorted({(s['gen_name'], s['transition']) for s in SPECS})
REF_CELLS = [f'{g}_chunk{REF_CHUNK_SIZE}_ndrift{REF_N_DRIFTS}_{d}' for (g, d) in GEN_DRIFT_PAIRS]

MEASURES = [
    'clustering', 'complexity', 'concept', 'general', 'info-theory',
    'itemset', 'landmarking', 'model-based', 'statistical',
]
ABFS_VERSIONS = ['aggstats', 'raw', 'raw_temporal']
ABFS_LABELS   = {'aggstats': 'Aggstats (v1.1)', 'raw': 'Raw scores (v2.0)',
                 'raw_temporal': 'Raw + temporal (v2.1)'}
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
    """Re-run ABFS on the rep-0 realization -> scores, {version: X}, concept labels."""
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

        fname = os.path.join(FIGURES_DIR, f'trajectory_abfs_{cell}.png')
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
            fig.suptitle(f'ABFS trajectories (mean over reps) -- {cell}\n({n_concepts} concepts, baseline={rb:.3f})', fontsize=12)
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
            ax.set_title(f'ABFS -- {ml}', fontsize=12)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            fig.suptitle(f'{ml} (mean over reps) -- {cell}\nfinal window | baseline={1/n_concepts:.3f}', fontsize=12)
            plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


# ============================================================
#  STREAM ANALYSIS  (reference cells, inline diagnostics)
# ============================================================
if args.stream_analysis:
    print("\n" + "="*60); print("STREAM ANALYSIS"); print("="*60)
    for cell in REF_CELLS:
        print(f"\n  {cell}")
        di, cd, le, cpc, spec = stream_diagnostics(cell)
        boundaries = boundaries_from_cpc(cpc)
        scores, _, _, _, _ = re_extract(cell)
        dr = np.linalg.norm(scores[1:] - scores[:-1], axis=1)
        dr = np.concatenate([[0], dr]); dr = dr / (np.max(dr) + 1e-10)

        fname = os.path.join(FIGURES_DIR, f'stream_drift_entropy_{cell}.png')
        if not os.path.exists(fname):
            di_n = di / (np.max(di) + 1e-10)
            fig, ax1 = plt.subplots(figsize=(14, 4))
            ax1.plot(di_n, color='steelblue', label='Drift intensity', linewidth=1.5)
            ax1.plot(dr, color='purple', label='ABFS relevance change', linewidth=1.2, alpha=0.7)
            ax1.set_ylabel('Normalized value')
            ax2 = ax1.twinx(); ax2.plot(le, color='darkorange', label='Label entropy', alpha=0.7)
            ax2.set_ylabel('Entropy', color='darkorange')
            for b in boundaries:
                ax1.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax1.set_xlabel('Window')
            l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, lab1 + lab2, loc='upper right')
            ax1.set_title(f'Drift vs ABFS dynamics -- {cell}')
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")

        fname = os.path.join(FIGURES_DIR, f'class_distribution_{cell}.png')
        if not os.path.exists(fname):
            fig, ax = plt.subplots(figsize=(14, 4))
            for c in range(cd.shape[1]):
                ax.plot(cd[:, c], label=f'class {c}', linewidth=1.2)
            for b in boundaries:
                ax.axvline(x=b, color='grey', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.set_xlabel('Window'); ax.set_ylabel('Proportion')
            ax.set_title(f'Class distribution over time -- {cell}'); ax.legend(ncol=4, fontsize=8)
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


if args.gap:
    print("\n" + "="*60); print("GAP HEATMAP (fixed layout)"); print("="*60)
    for cell in REF_CELLS:
 
        # ---- ABFS raw v2.0 (mean over reps), per classifier ----
        pr = load('preq_abfs_raw_ba', cell, optional=True)
        if pr is None:
            print(f"  {cell}: no ABFS raw -- skipping."); continue
        abfs_final = np.mean(pr[:, -1, :], axis=0)            # (n_clfs,)

        komor_rows = []
        for measure in MEASURES:
            d = load(f'preq_komor_{measure}_ba', cell, optional=True)
            if d is not None:
                komor_rows.append(np.mean(d[:, -1, :], axis=0))
        if not komor_rows:
            print(f"  {cell}: no Komorniczak -- skipping."); continue
        komor_best = np.nanmax(np.vstack(komor_rows), axis=0)  # (n_clfs,)
 
        gap  = abfs_final - komor_best
        vmax = np.nanmax(np.abs(gap)) if np.any(~np.isnan(gap)) else 1.0
        gap_heatmap(
            cell, gap, vmax,
            f'Gap (ABFS raw v2.0 minus best Komorniczak per classifier) -- {cell}')


# ============================================================
#  GRID  -- chunk_size x n_drifts gap heatmap + BA-vs-n_drifts curves
# ============================================================
if args.grid:
    print("\n" + "="*60); print("GRID: cs x n_drifts"); print("="*60)

    def best_abfs(cell):
        pr = load('preq_abfs_raw_ba', cell, optional=True)
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

    for (gen, drift) in GEN_DRIFT_PAIRS:
        # --- 2D gap heatmap: rows = chunk_size, cols = n_drifts ---
        fname = os.path.join(FIGURES_DIR, f'gap_grid_{gen}_{drift}.png')
        if not os.path.exists(fname):
            gap_grid = np.full((len(CHUNK_SIZES), len(EXP4_N_DRIFTS)), np.nan)
            for ri, cs in enumerate(CHUNK_SIZES):
                for ci, nd in enumerate(EXP4_N_DRIFTS):
                    cell = f'{gen}_chunk{cs}_ndrift{nd}_{drift}'
                    a = best_abfs(cell); k = best_komor(cell)
                    if a is not None and k is not None:
                        gap_grid[ri, ci] = a - k
            if np.any(~np.isnan(gap_grid)):
                vmax = np.nanmax(np.abs(gap_grid))
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(gap_grid, vmin=-vmax, vmax=vmax, cmap='RdBu', aspect='auto')
                for ri in range(len(CHUNK_SIZES)):
                    for ci in range(len(EXP4_N_DRIFTS)):
                        v = gap_grid[ri, ci]
                        if not np.isnan(v):
                            ax.text(ci, ri, f'{v:+.3f}', ha='center', va='center', fontsize=10,
                                    color='white' if abs(v) > vmax * 0.6 else 'black')
                ax.set_xticks(range(len(EXP4_N_DRIFTS))); ax.set_xticklabels(EXP4_N_DRIFTS)
                ax.set_yticks(range(len(CHUNK_SIZES))); ax.set_yticklabels(CHUNK_SIZES)
                ax.set_xlabel('n_drifts'); ax.set_ylabel('chunk_size')
                ax.set_title(f'Gap (ABFS raw v2.0 minus Komorniczak best) -- {gen}, {drift}')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Gap (BA)')
                fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(); print(f"  Saved: {fname}")
            else:
                print(f"  {gen}_{drift}: no grid data -- skipping heatmap.")

        # --- BA vs n_drifts, one curve set per chunk_size ---
        for cs in CHUNK_SIZES:
            fname = os.path.join(FIGURES_DIR, f'ba_vs_ndrifts_{gen}_chunk{cs}_{drift}.png')
            if os.path.exists(fname):
                continue
            xs, abfs_curve, komor_curve = [], [], []
            for nd in EXP4_N_DRIFTS:
                cell = f'{gen}_chunk{cs}_ndrift{nd}_{drift}'
                a = best_abfs(cell); k = best_komor(cell)
                if a is None or k is None:
                    continue
                xs.append(nd); abfs_curve.append(a); komor_curve.append(k)
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(xs, abfs_curve, 'o-', color='#911eb4', label='ABFS raw v2.0 (best clf)', linewidth=2)
            ax.plot(xs, komor_curve, 's-', color='#3cb44b', label='Komorniczak best-of-9 (best clf)', linewidth=2)
            ax.set_xticks(EXP4_N_DRIFTS); ax.set_xlabel('n_drifts (recurrence amount)')
            ax.set_ylabel('Final balanced accuracy (mean over reps)')
            ax.set_title(f'BA vs n_drifts -- {gen}, chunk_size={cs}, {drift}')
            ax.legend(); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(); print(f"  Saved: {fname}")


print("\nAnalysis 4 complete.")