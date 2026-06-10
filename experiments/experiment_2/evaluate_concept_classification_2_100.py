# evaluate_concept_classification_2.py
# ==============================================================================
# Experiment 2: Stream Configuration Sensitivity
#
# Tests how ABFS-based meta-features (raw scores v2.0) and all 9 Komorniczak
# measure groups respond to changes in:
#   - chunk_size   : 100, 200, 500, 1000
#   - n_informative: 3, 5, 10, 15 (n_features fixed at 20)
#
# 4x4 grid = 16 configurations x 2 drift types = 32 stream variants.
#
# Evaluation protocol: Prequential only (test-then-train per window).
# 5 replications per cell with different random seeds.
#
# Komorniczak baseline: all 9 measure groups re-extracted using pymfe
# on the same streams (cannot reuse Exp 1 files — chunk_size and
# n_informative vary across cells).
#
# Outputs saved to results/experiment_2/:
#   preq_abfs_ba_chunk{cs}_ninf{ni}_{drift}.npy    shape: (n_reps, n_windows, n_clfs)
#   preq_abfs_f1_chunk{cs}_ninf{ni}_{drift}.npy
#   preq_abfs_kappa_chunk{cs}_ninf{ni}_{drift}.npy
#   preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy
#   preq_komor_{measure}_f1_chunk{cs}_ninf{ni}_{drift}.npy
#   preq_komor_{measure}_kappa_chunk{cs}_ninf{ni}_{drift}.npy
#
#   Figures saved in results/experiment_2/figures/:
#     heatmap_combined_preq_chunk{cs}_ninf{ni}_{drift}.png
#       Left:  9 Komorniczak measure groups (rows) x classifiers (cols)
#       Right: ABFS raw scores v2.0 (1 row)        x classifiers (cols)
# ==============================================================================

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')    # experiments/
sys.path.append('../..') # project root

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL
from pymfe.mfe import MFE


# ============================================================
#  FIXED CONFIGURATION
# ============================================================

N_CHUNKS       = 5000
N_FEATURES     = 20
WARMUP_WINDOWS = 10
N_REPLICATIONS = 5

CHUNK_SIZES    = [100]
N_INFORMATIVES = [3, 5, 10, 15]

DRIFT_CONFIGS = [
    ('sudden',  20, 9999),  # 21 concepts
    ('gradual',  6,    5),  # 25 concepts
]

MEASURES = [
    'clustering',
    'complexity',
    'concept',
    'general',
    'info-theory',
    'itemset',
    'landmarking',
    'model-based',
    'statistical',
]

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
#  HELPERS
# ============================================================

def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing, stream.n_drifts)[1][::chunk_size]
    concept   = 0
    decreasing = True
    labels    = []
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


def get_concept_labels(stream, drift_type, n_chunks, chunk_size):
    if drift_type == 'sudden':
        cs = stream.concept_selector.copy()
        return np.array([
            int(np.bincount(cs[i*chunk_size:(i+1)*chunk_size]).argmax())
            for i in range(n_chunks)
        ])
    else:
        return assign_labels_gradual(stream, n_chunks, chunk_size)


def make_tag(chunk_size, n_informative, drift_type):
    return f"chunk{chunk_size}_ninf{n_informative}_{drift_type}"


def already_done_abfs(tag):
    """Return True if all 3 ABFS prequential result files exist."""
    prefixes = ['preq_abfs_ba', 'preq_abfs_f1', 'preq_abfs_kappa']
    return all(
        os.path.exists(os.path.join(RESULTS_DIR, f'{p}_{tag}.npy'))
        for p in prefixes
    )


def already_done_komor(tag, measure):
    """Return True if all 3 Komorniczak result files for this measure exist."""
    prefixes = [
        f'preq_komor_{measure}_ba',
        f'preq_komor_{measure}_f1',
        f'preq_komor_{measure}_kappa',
    ]
    return all(
        os.path.exists(os.path.join(RESULTS_DIR, f'{p}_{tag}.npy'))
        for p in prefixes
    )


def save(array, prefix, tag):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{tag}.npy')
    np.save(path, array)
    return path


# ============================================================
#  COMBINED HEATMAP
#  Left:  9 Komorniczak measure groups x classifiers
#  Right: ABFS raw scores v2.0         x classifiers
# ============================================================

def plot_combined_heatmap(tag, drift_type, n_concepts, pr_abfs_ba_arr):
    """
    Parameters
    ----------
    tag            : str  e.g. 'chunk200_ninf10_sudden'
    drift_type     : str
    n_concepts     : int
    pr_abfs_ba_arr : np.ndarray (n_reps, n_windows, n_clfs) — ABFS prequential BA
    """
    clf_names = [n for n, _ in BASE_CLFS_PREQUENTIAL]
    n_clfs    = len(clf_names)

    # ---- Komorniczak matrix (9 x n_clfs) ----
    komor_matrix     = np.full((len(MEASURES), n_clfs), np.nan)
    komor_std_matrix = np.full((len(MEASURES), n_clfs), np.nan)
    for m_id, measure in enumerate(MEASURES):
        path = os.path.join(RESULTS_DIR, f'preq_komor_{measure}_ba_{tag}.npy')
        if not os.path.exists(path):
            print(f"  Combined heatmap: missing {measure} — skipping plot.")
            return
        arr = np.load(path)  # (n_reps, n_windows, n_clfs)
        komor_matrix[m_id, :]     = np.mean(arr[:, -1, :], axis=0)
        komor_std_matrix[m_id, :] = np.std(arr[:, -1, :],  axis=0)

    # ---- ABFS matrix (1 x n_clfs) ----
    abfs_mean   = np.mean(pr_abfs_ba_arr[:, -1, :], axis=0)  # (n_clfs,)
    abfs_std    = np.std(pr_abfs_ba_arr[:, -1, :],  axis=0)
    abfs_matrix = abfs_mean[np.newaxis, :]
    abfs_std_matrix = abfs_std[np.newaxis, :]

    random_baseline = 1.0 / n_concepts

    fig, axes = plt.subplots(
        1, 2,
        figsize=(26, max(5, len(MEASURES) * 0.75)),
        gridspec_kw={'width_ratios': [3, 1.5]}
    )

    # ---- left: Komorniczak ----
    ax = axes[0]
    ax.imshow(komor_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i, measure in enumerate(MEASURES):
        for j in range(n_clfs):
            val = komor_matrix[i, j]
            std = komor_std_matrix[i, j]
            txt_color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}\n(±{std:.3f})',
                    ha='center', va='center', fontsize=10,
                    color=txt_color, linespacing=1.4)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(MEASURES)))
    ax.set_yticklabels(MEASURES, fontsize=10)
    ax.set_title('Komorniczak meta-features — balanced accuracy', fontsize=12)

    # ---- right: ABFS ----
    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for j in range(n_clfs):
        val = abfs_matrix[0, j]
        std = abfs_std_matrix[0, j]
        txt_color = 'white' if val > 0.6 else 'black'
        ax.text(j, 0, f'{val:.3f}\n(±{std:.3f})',
                ha='center', va='center', fontsize=10,
                color=txt_color, linespacing=1.4)
    ax.set_xticks(range(n_clfs))
    ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels(['Raw scores (v2.0)'], fontsize=10)
    ax.set_title('ABFS meta-features — balanced accuracy', fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f'Komorniczak vs ABFS — {drift_type} drift '
        f'({n_concepts} concepts) — {tag}\n'
        f'Prequential | mean ± std over {N_REPLICATIONS} replications '
        f'| random baseline = {random_baseline:.3f}',
        fontsize=13
    )
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR,
                            f'heatmap_comparison_komorniczak_ABFS_preq_{tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {out_path}")


# ============================================================
#  ABFS META-FEATURE EXTRACTION (raw scores v2.0)
# ============================================================

def extract_abfs_metafeatures(random_state, drift_type, n_drifts,
                               concept_sigmoid_spacing, chunk_size, n_informative):
    config = dict(
        n_drifts                = n_drifts,
        n_chunks                = N_CHUNKS,
        chunk_size              = chunk_size,
        n_features              = N_FEATURES,
        n_informative           = n_informative,
        n_redundant             = 0,
        n_repeated              = 0,
        concept_sigmoid_spacing = concept_sigmoid_spacing,
        random_state            = random_state,
    )

    stream = StreamGenerator(**config)

    # pass 1: concept labels
    abfs = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = chunk_size,
        class_window_size    = chunk_size,
    )
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

    concept_labels_all = get_concept_labels(stream, drift_type,
                                            N_CHUNKS, chunk_size)

    # pass 2: extract meta-features
    abfs = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = chunk_size,
        class_window_size    = chunk_size,
    )

    meta_features  = []
    concept_labels = []
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

        window_counter += 1

    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y


# ============================================================
#  KOMORNICZAK BASELINE EXTRACTION (one measure group at a time)
# ============================================================

def extract_komor_metafeatures(random_state, drift_type, n_drifts,
                                concept_sigmoid_spacing,
                                chunk_size, n_informative, measure):
    config = dict(
        n_drifts                = n_drifts,
        n_chunks                = N_CHUNKS,
        chunk_size              = chunk_size,
        n_features              = N_FEATURES,
        n_informative           = n_informative,
        n_redundant             = 0,
        n_repeated              = 0,
        concept_sigmoid_spacing = concept_sigmoid_spacing,
        random_state            = random_state,
    )

    stream = StreamGenerator(**config)

    # pass 1: concept labels
    abfs_dummy = ABFS_match(
        n_features           = N_FEATURES,
        categorical_features = [],
        accuracy_window_size = chunk_size,
        class_window_size    = chunk_size,
    )
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs_dummy.update(X_chunk[i], y_chunk[i])

    concept_labels_all = get_concept_labels(stream, drift_type,
                                            N_CHUNKS, chunk_size)

    # pass 2: pymfe features per chunk
    mfe = MFE(groups=[measure], suppress_warnings=True)

    meta_features  = []
    concept_labels = []

    stream.reset()
    window_counter = 0

    for X_chunk, y_chunk in stream:
        if window_counter < WARMUP_WINDOWS:
            window_counter += 1
            continue

        try:
            mfe.fit(X_chunk, y_chunk)
            _, ft_vals = mfe.extract(suppress_warnings=True)
            ft_vals = np.array(ft_vals, dtype=float)
            ft_vals[np.isnan(ft_vals)] = 0
            ft_vals[np.isinf(ft_vals)] = 0
        except Exception:
            ft_vals = np.zeros(1)

        meta_features.append(ft_vals)
        concept_labels.append(concept_labels_all[window_counter])
        window_counter += 1

    # pad to uniform width
    lengths    = [len(f) for f in meta_features]
    target_len = max(set(lengths), key=lengths.count)
    padded = []
    for f in meta_features:
        if len(f) == target_len:
            padded.append(f)
        elif len(f) < target_len:
            padded.append(np.concatenate([f, np.zeros(target_len - len(f))]))
        else:
            padded.append(f[:target_len])

    X = np.array(padded, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y


# ============================================================
#  MAIN SWEEP
# ============================================================

for drift_type, n_drifts, concept_sigmoid_spacing in DRIFT_CONFIGS:
    n_concepts      = 25 if drift_type == 'gradual' else n_drifts + 1
    random_baseline = 1 / n_concepts

    print(f"\n{'#'*70}")
    print(f"DRIFT TYPE: {drift_type.upper()} ({n_concepts} concepts, "
          f"random baseline={random_baseline:.3f})")
    print(f"{'#'*70}")

    for chunk_size in CHUNK_SIZES:
        for n_informative in N_INFORMATIVES:

            tag = make_tag(chunk_size, n_informative, drift_type)

            print(f"\n{'='*70}")
            print(f"chunk_size={chunk_size} | n_informative={n_informative}"
                  f" | drift={drift_type}  [tag: {tag}]")
            print(f"{'='*70}")

            # ---- ABFS prequential ----
            if already_done_abfs(tag):
                print("  ABFS results exist — loading from disk.")
                pr_abfs_ba_arr = np.load(
                    os.path.join(RESULTS_DIR, f'preq_abfs_ba_{tag}.npy'))
            else:
                pr_abfs_ba    = []
                pr_abfs_f1    = []
                pr_abfs_kappa = []

                for rep_id, rs in enumerate(RANDOM_STATES):
                    print(f"\n  [ABFS] Replication {rep_id+1}/{N_REPLICATIONS} "
                          f"(seed={rs})...")
                    X_abfs, y_abfs = extract_abfs_metafeatures(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative)

                    _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(
                        X_abfs, y_abfs)
                    pr_abfs_ba.append(tba)
                    pr_abfs_f1.append(tf1)
                    pr_abfs_kappa.append(tk)

                    print(f"    [Preq ABFS] " + "  ".join(
                        f"{n}={tba[-1, i]:.3f}"
                        for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

                pr_abfs_ba_arr = np.array(pr_abfs_ba)
                save(pr_abfs_ba_arr,           'preq_abfs_ba',    tag)
                save(np.array(pr_abfs_f1),     'preq_abfs_f1',    tag)
                save(np.array(pr_abfs_kappa),  'preq_abfs_kappa', tag)
                print(f"  Saved ABFS results for {tag}")

            # ---- Komorniczak: all 9 measure groups ----
            for measure in MEASURES:

                print(f"\n  --- Measure group: {measure} ---")

                if already_done_komor(tag, measure):
                    print(f"  Komorniczak [{measure}] exists — skipping.")
                    continue

                pr_komor_ba    = []
                pr_komor_f1    = []
                pr_komor_kappa = []

                for rep_id, rs in enumerate(RANDOM_STATES):
                    print(f"  [Komor {measure}] Replication {rep_id+1}/"
                          f"{N_REPLICATIONS} (seed={rs})...")
                    X_komor, y_komor = extract_komor_metafeatures(
                        rs, drift_type, n_drifts, concept_sigmoid_spacing,
                        chunk_size, n_informative, measure)

                    _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(
                        X_komor, y_komor)
                    pr_komor_ba.append(tba)
                    pr_komor_f1.append(tf1)
                    pr_komor_kappa.append(tk)

                    print(f"    [Preq Komor {measure}] " + "  ".join(
                        f"{n}={tba[-1, i]:.3f}"
                        for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

                save(np.array(pr_komor_ba),    f'preq_komor_{measure}_ba',    tag)
                save(np.array(pr_komor_f1),    f'preq_komor_{measure}_f1',    tag)
                save(np.array(pr_komor_kappa), f'preq_komor_{measure}_kappa', tag)
                print(f"  Saved Komorniczak [{measure}] results for {tag}")

            # ---- combined heatmap: after ABFS + all 9 measure groups ----
            plot_combined_heatmap(
                tag            = tag,
                drift_type     = drift_type,
                n_concepts     = n_concepts,
                pr_abfs_ba_arr = pr_abfs_ba_arr,
            )