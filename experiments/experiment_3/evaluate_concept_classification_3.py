#  evaluate_concept_classification_3.py
# ==============================================================================
# Experiment 3: SEA / STAGGER / LED Stream Evaluation (sequential concepts)
#
# Same core question as every experiment here (does ABFS discriminate
# concepts better than Komorniczak?), now on three classic named
# concept-drift generators from river.datasets.synth -- SEA, STAGGER,
# LED -- with SEQUENTIAL concepts (each concept appears once in order;
# the sole repeat is STAGGER's [0,1,2,0], since STAGGER only has 3
# classification functions). Recurring concepts are Experiment 4's job.
#
# Streams come from streams.generate_synthetic_streams.exp3_specs(),
# the single source of truth for the Exp 3 stream set (names, feature
# counts, concept counts, and a seed->(data, concept_per_chunk)
# builder per stream). 6 streams: {sea, stagger, led} x {sudden,
# gradual}.
#
# REPLICATIONS (key structural difference from Experiment 5's
# single-realization real streams): each stream is regenerated from
# N_REPLICATIONS different seeds and the per-window results are stacked
# into (n_reps, n_windows, n_clfs) -- exactly like Experiments 1c/2 do
# with StreamGenerator's random_state. This is only meaningful because
# the seed genuinely propagates into the river generators (see the
# header note in generate_synthetic_streams.py); otherwise sudden-stream
# reps would be byte-identical. RANDOM_STATES[0] = MASTER_SEED, so
# replication 0 reproduces exactly the on-disk stream that analysis_3.py
# loads.
#
# y_chunk (npy last column) vs concept label:
#   y_chunk = generator's REAL target -- binary for SEA/STAGGER,
#   10-class for LED. ABFS/Komorniczak compute meta-features against
#   this, same as everywhere. The CONCEPT label per window is
#   concept_per_chunk[i], the GENERATIVE concept id (exact, since we
#   built the stream -- not positional like Experiment 5).
#
# chunk_size=200, WARMUP_WINDOWS=0 (consistent with Experiment 5).
#
# Output: results/experiment_3/, shape (n_reps, n_windows, n_clfs)
#   preq_abfs_{version}_ba_{stream}.npy   / _f1_ / _kappa_
#   preq_komor_{measure}_ba_{stream}.npy  / _f1_ / _kappa_
#   concept_labels_{stream}.npy           (rep-0 concept labels)
# Figures: results/experiment_3/figures/
#   heatmap_comparison_komorniczak_ABFS_preq_exp3_{stream}.png
#
# Run from project root:
#   python experiments/experiment_3/evaluate_concept_classification_3.py
# ==============================================================================

import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from pymfe.mfe import MFE
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
)
from streams.generate_synthetic_streams import exp3_specs, CHUNK_SIZE, MASTER_SEED
from classifier_sweep_prequential import run_prequential_sweep, BASE_CLFS_PREQUENTIAL


# ============================================================
#  PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

KOMOR_CACHE_DIR = os.path.join(PROJECT_ROOT, 'external', 'komorniczak',
                               'results', 'synthetic_sea_stagger_led')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
FIGURES_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_3', 'figures')
os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs(FIGURES_DIR,     exist_ok=True)
os.makedirs(KOMOR_CACHE_DIR, exist_ok=True)


# ============================================================
#  CONFIGURATION
# ============================================================
WARMUP_WINDOWS = 0
N_REPLICATIONS = 5

SPECS = exp3_specs()

_seed_rng = np.random.RandomState(MASTER_SEED)
RANDOM_STATES = [MASTER_SEED] + [int(s) for s in
                                 _seed_rng.randint(100, 100000, N_REPLICATIONS - 1)]
print(f"Replication seeds: {RANDOM_STATES}")

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

EXPECTED_N_CLFS = len(BASE_CLFS_PREQUENTIAL)
EXP_TAG = 'exp3'


# ============================================================
#  HELPERS
# ============================================================

def already_done_abfs(stream_name, version):
    prefixes = [f'preq_abfs_{version}_ba', f'preq_abfs_{version}_f1',
                f'preq_abfs_{version}_kappa']
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if (arr.ndim != 3 or arr.shape[0] != N_REPLICATIONS
                or arr.shape[2] != EXPECTED_N_CLFS):
            print(f"  WARNING: {path} wrong shape {arr.shape} -- will rerun.")
            return False
    if not os.path.exists(os.path.join(RESULTS_DIR,
                                       f'concept_labels_{stream_name}.npy')):
        return False
    return True


def already_done_komor(stream_name, measure):
    prefixes = [f'preq_komor_{measure}_ba', f'preq_komor_{measure}_f1',
                f'preq_komor_{measure}_kappa']
    for p in prefixes:
        path = os.path.join(RESULTS_DIR, f'{p}_{stream_name}.npy')
        if not os.path.exists(path):
            return False
        arr = np.load(path)
        if (arr.ndim != 3 or arr.shape[0] != N_REPLICATIONS
                or arr.shape[2] != EXPECTED_N_CLFS):
            print(f"  WARNING: {path} wrong shape {arr.shape} -- will rerun.")
            return False
    return True


def save(array, prefix, stream_name):
    path = os.path.join(RESULTS_DIR, f'{prefix}_{stream_name}.npy')
    np.save(path, array)
    return path


def print_label_dist(name, y):
    dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  {name} label distribution: {dist}")
    return dist


def chunk_iter(data, concept_per_chunk):
    """Yield (chunk_idx, X_chunk, y_chunk, concept) over a regenerated
    in-memory stream array (data = (n_instances, n_features+1))."""
    X_full = data[:, :-1]
    y_full = data[:, -1]
    n_chunks = len(concept_per_chunk)
    for ci in range(n_chunks):
        s = ci * CHUNK_SIZE
        e = s + CHUNK_SIZE
        yield ci, X_full[s:e], y_full[s:e], int(concept_per_chunk[ci])


# ============================================================
#  COMBINED HEATMAP -- Komorniczak vs ABFS, mean final window over reps
# ============================================================

def plot_combined_heatmap(stream_name, n_concepts, tba_versions):
    clf_names = [n for n, _ in BASE_CLFS_PREQUENTIAL]
    n_clfs    = len(clf_names)

    komor_matrix = np.full((len(MEASURES), n_clfs), np.nan)
    for m_id, measure in enumerate(MEASURES):
        path = os.path.join(RESULTS_DIR,
                            f'preq_komor_{measure}_ba_{stream_name}.npy')
        if not os.path.exists(path):
            print(f"  Heatmap: missing {measure} -- skipping.")
            return
        komor_matrix[m_id, :] = np.mean(np.load(path)[:, -1, :], axis=0)

    abfs_matrix = np.full((len(ABFS_VERSIONS), n_clfs), np.nan)
    for v_id, version in enumerate(ABFS_VERSIONS):
        if version in tba_versions:
            abfs_matrix[v_id, :] = np.mean(tba_versions[version][:, -1, :], axis=0)
        else:
            path = os.path.join(RESULTS_DIR,
                                f'preq_abfs_{version}_ba_{stream_name}.npy')
            if not os.path.exists(path):
                print(f"  Heatmap: missing ABFS {version} -- skipping.")
                return
            abfs_matrix[v_id, :] = np.mean(np.load(path)[:, -1, :], axis=0)

    random_baseline = 1.0 / n_concepts
    fig, axes = plt.subplots(
        1, 2, figsize=(26, max(5, len(MEASURES) * 0.75)),
        gridspec_kw={'width_ratios': [3, 1.5]})

    ax = axes[0]
    ax.imshow(komor_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(len(MEASURES)):
        for j in range(n_clfs):
            val = komor_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if val > 0.6 else 'black')
    ax.set_xticks(range(n_clfs)); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(MEASURES))); ax.set_yticklabels(MEASURES, fontsize=10)
    ax.set_title('Komorniczak meta-features -- balanced accuracy', fontsize=12)

    ax = axes[1]
    im = ax.imshow(abfs_matrix, vmin=0.0, vmax=1.0, cmap='Blues', aspect='auto')
    for i in range(len(ABFS_VERSIONS)):
        for j in range(n_clfs):
            val = abfs_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color='white' if val > 0.6 else 'black')
    ax.set_xticks(range(n_clfs)); ax.set_xticklabels(clf_names, fontsize=10)
    ax.set_yticks(range(len(ABFS_VERSIONS)))
    ax.set_yticklabels([ABFS_LABELS[v] for v in ABFS_VERSIONS], fontsize=10)
    ax.set_title('ABFS meta-features -- balanced accuracy', fontsize=12)

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f'Komorniczak vs ABFS -- {stream_name}\n'
        f'Prequential  |  chunk_size={CHUNK_SIZE}  |  mean of {N_REPLICATIONS} reps  |  '
        f'random baseline = {random_baseline:.3f}',
        fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(
        FIGURES_DIR,
        f'heatmap_comparison_komorniczak_ABFS_preq_{EXP_TAG}_{stream_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {out_path}")


# ============================================================
#  ABFS EXTRACTION -- all 3 versions in one pass over one realization
# ============================================================

def extract_abfs_metafeatures(data, concept_per_chunk, n_features):
    abfs = ABFS_match(n_features=n_features, categorical_features=[],
                      accuracy_window_size=CHUNK_SIZE,
                      class_window_size=CHUNK_SIZE)
    mf_aggstats, mf_raw, mf_raw_temporal = [], [], []
    concept_labels = []
    wt_prev = None

    for ci, X_chunk, y_chunk, concept in chunk_iter(data, concept_per_chunk):
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
        wt          = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()
        t_since     = abfs.time_since_drift

        if ci >= WARMUP_WINDOWS:
            mf_aggstats.append(extract_metafeatures(wt, wt_prev, drift_count, t_since))
            mf_raw.append(extract_metafeatures_raw(wt))
            mf_raw_temporal.append(extract_metafeatures_raw_temporal(wt, wt_prev))
            concept_labels.append(concept)
        wt_prev = wt

    def clean(arr):
        a = np.array(arr, dtype=float)
        a[np.isnan(a)] = 0; a[np.isinf(a)] = 0
        return a

    return (clean(mf_aggstats), clean(mf_raw), clean(mf_raw_temporal),
            np.array(concept_labels))


# ============================================================
#  KOMORNICZAK EXTRACTION -- inline, cached per (stream, measure, seed)
# ============================================================

def extract_komor_metafeatures(stream_name, measure, seed, data, concept_per_chunk):
    cache_path = os.path.join(
        KOMOR_CACHE_DIR, f'komor_{stream_name}_{measure}_seed{seed}.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)

    mfe = MFE(groups=[measure], suppress_warnings=True)
    out = []
    for ci, X_chunk, y_chunk, concept in chunk_iter(data, concept_per_chunk):
        if ci < WARMUP_WINDOWS:
            continue
        try:
            mfe.fit(X_chunk, y_chunk)
            _, ft = mfe.extract(suppress_warnings=True)
            ft = np.array(ft, dtype=float)
            ft[np.isnan(ft)] = 0; ft[np.isinf(ft)] = 0
        except Exception as e:
            print(f"    chunk {ci}: pymfe failed ({e}) -- skipping.")
            continue
        out.append(np.append(ft, concept))

    result = np.array(out)
    np.save(cache_path, result)
    print(f"  [{measure} seed={seed}] cached: shape={result.shape}")
    return result


# ============================================================
#  MAIN SWEEP -- loop over all streams x all replications
# ============================================================

for spec in SPECS:
    stream_name = spec['name']
    n_features  = spec['n_features']
    builder     = spec['builder']

    print(f"\n{'='*70}")
    print(f"Stream   : {stream_name}")
    print(f"Generator: {spec['gen_name']}  |  Features: {n_features}  |  "
          f"Order: {spec['order']}  |  Concepts: {spec['n_concepts']}")
    print(f"{'='*70}")

    abfs_needed  = [v for v in ABFS_VERSIONS if not already_done_abfs(stream_name, v)]
    komor_needed = [m for m in MEASURES if not already_done_komor(stream_name, m)]

    tba_versions = {}
    n_concepts   = spec['n_concepts']

    if not abfs_needed and not komor_needed:
        print("  All results present -- loading ABFS BA for heatmap only.")
        for version in ABFS_VERSIONS:
            tba_versions[version] = np.load(
                os.path.join(RESULTS_DIR, f'preq_abfs_{version}_ba_{stream_name}.npy'))
        plot_combined_heatmap(stream_name, n_concepts, tba_versions)
        continue

    abfs_acc  = {v: {'ba': [], 'f1': [], 'kappa': []} for v in ABFS_VERSIONS}
    komor_acc = {m: {'ba': [], 'f1': [], 'kappa': []} for m in MEASURES}

    for rep_id, seed in enumerate(RANDOM_STATES):
        print(f"\n  --- Replication {rep_id+1}/{N_REPLICATIONS} (seed={seed}) ---")
        data, cpc = builder(seed)

        # ---- ABFS (all 3 versions, one pass) ----
        X_agg, X_raw, X_rt, y = extract_abfs_metafeatures(data, cpc, n_features)
        X_by_version = {'aggstats': X_agg, 'raw': X_raw, 'raw_temporal': X_rt}
        if rep_id == 0:
            n_concepts = len(np.unique(y))
            print(f"    aggstats={X_agg.shape} raw={X_raw.shape} raw_temporal={X_rt.shape}")
            print_label_dist('ABFS (rep0)', y)
            save(y, 'concept_labels', stream_name)

        for version in ABFS_VERSIONS:
            _, _, tba, _, _, tf1, _, _, tk = run_prequential_sweep(
                X_by_version[version], y)
            abfs_acc[version]['ba'].append(tba)
            abfs_acc[version]['f1'].append(tf1)
            abfs_acc[version]['kappa'].append(tk)

        # ---- Komorniczak (all 9 measures) ----
        for measure in MEASURES:
            komor_data = extract_komor_metafeatures(
                stream_name, measure, seed, data, cpc)
            Xk = komor_data[:, :-1]
            yk = komor_data[:, -1].astype(int)
            Xk[np.isnan(Xk)] = 0; Xk[np.isinf(Xk)] = 0
            _, _, tba_k, _, _, tf1_k, _, _, tk_k = run_prequential_sweep(Xk, yk)
            komor_acc[measure]['ba'].append(tba_k)
            komor_acc[measure]['f1'].append(tf1_k)
            komor_acc[measure]['kappa'].append(tk_k)

    # ---- stack across reps -> (n_reps, n_windows, n_clfs) and save ----
    for version in ABFS_VERSIONS:
        tba_arr = np.array(abfs_acc[version]['ba'])
        save(tba_arr,                              f'preq_abfs_{version}_ba',    stream_name)
        save(np.array(abfs_acc[version]['f1']),    f'preq_abfs_{version}_f1',    stream_name)
        save(np.array(abfs_acc[version]['kappa']), f'preq_abfs_{version}_kappa', stream_name)
        tba_versions[version] = tba_arr
        mean_final = np.mean(tba_arr[:, -1, :], axis=0)
        print(f"  [ABFS {version}, mean of {N_REPLICATIONS} reps] " + "  ".join(
            f"{n}={mean_final[i]:.3f}"
            for i, (n, _) in enumerate(BASE_CLFS_PREQUENTIAL)))

    for measure in MEASURES:
        save(np.array(komor_acc[measure]['ba']),    f'preq_komor_{measure}_ba',    stream_name)
        save(np.array(komor_acc[measure]['f1']),    f'preq_komor_{measure}_f1',    stream_name)
        save(np.array(komor_acc[measure]['kappa']), f'preq_komor_{measure}_kappa', stream_name)

    plot_combined_heatmap(stream_name, n_concepts, tba_versions)

print("\nExperiment 3 complete.")