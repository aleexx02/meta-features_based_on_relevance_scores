
# ============================================================
# Evaluation of our ABFS-based meta-features using the same
# experimental setup as Komorniczak et al. (2024):
#   - Same streams: same StreamGenerator configuration and seeds
#   - Same concept labelling: majority vote (sudden) and sigmoid
#     threshold method (gradual)
#   - Same evaluation protocol: classifier_sweep_komor.py

# The only difference with respect to their pipeline is the
# meta-features: instead of statistical descriptors computed
# by pymfe directly from the raw instances, we use relevance
# scores produced by ABFS — encoding which features are
# currently predictive and how that relevance is evolving
# over time.

# By controlling for everything except the meta-features,
# any difference in balanced accuracy observed when comparing
# against replication_check.py can be attributed solely to
# the meta-features themselves.

# Steps:
#   1. Generate a synthetic stream using StreamGenerator
#   2. Run ABFS to compute per-feature relevance scores
#   3. Extract meta-feature vectors from the relevance scores
#   4. Assign concept labels to each window
#   5. Run the classifier sweep (classifier_sweep_komor.py)
# ============================================================
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn import clone
from strlearn.streams import StreamGenerator
import warnings
import os
import sys
sys.path.append('..')
warnings.filterwarnings('ignore')

from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures,
    extract_metafeatures_raw,
    extract_metafeatures_raw_temporal,
    extract_metafeatures_raw_delta,
    extract_metafeatures_raw_cosine,
    MF_NAMES_AGGSTATS,
    MF_NAMES_RAW,
    MF_NAMES_RAW_TEMPORAL
)


from classifier_sweep_komor import run_classifier_sweep, print_results, BASE_CLFS

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)


# =============================================
#  CONFIGURATION (matching Komorniczak et al.)
# =============================================

#  Sudden drift:
#    concept_sigmoid_spacing=9999, n_drifts=20 → 21 concepts
#    label assignment: majority vote (equivalent to their
#    threshold method since sigmoid is a step function)
#
#  Gradual drift:
#    concept_sigmoid_spacing=5, n_drifts=6 → 25 concepts
#    label assignment: sigmoid threshold method from
#    E1_extract_synthetic.py (Komorniczak et al.)
#    4 stages per transition:
#      concept%4==0: static   (e >= 0.9 / e <= 0.1)
#      concept%4==1: early    (e >= 0.75 / e >= 0.1)
#      concept%4==2: central  (e >= 0.25 / e >= 0.25)
#      concept%4==3: late     (e >= 0.1  / e >= 0.75)
# ============================================================

DRIFT_TYPE = 'sudden' # 'sudden' or 'gradual'

N_CHUNKS = 5000
CHUNK_SIZE = 200
N_FEATURES = 10
WARMUP_WINDOWS = 10
N_REPLICATIONS = 5

if DRIFT_TYPE == 'sudden':
    N_DRIFTS = 20
    CONCEPT_SIGMOID_SPACING = 9999
elif DRIFT_TYPE == 'gradual':
    N_DRIFTS = 6
    CONCEPT_SIGMOID_SPACING = 5
else:
    raise ValueError(f"Unknown DRIFT_TYPE: '{DRIFT_TYPE}'")

# rows of the heatmap — one per meta-feature set
# MF_CONFIGS = [
#     ('aggstats', 'Aggregate stats (v1.1)', 8),
#     ('raw', 'Raw scores (v2.0)', 10),
#     ('raw_temporal', 'Raw + temporal (v2.1)', 12),
# ]

MF_CONFIGS = [
    ('raw',          'Raw only (v2.0)',        10),
    ('raw_delta',    'Raw + delta_mean',        11),
    ('raw_cosine',   'Raw + cosine_sim',        11),
    ('raw_temporal', 'Raw + both (v2.1)',       12),
]

base_clfs = [
    ('GNB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=11313)),
    ('DT',  DecisionTreeClassifier(random_state=11313)),
    ('MLP', MLPClassifier(random_state=11313))
]

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")


# ============================================================
#  HELPER — build extract_mf for a given MF_TYPE
# ============================================================

def make_extract_mf(mf_type):
    if mf_type == 'aggstats':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures(
                wt=wt, wt_prev=wt_prev,
                drift_count=drift_count,
                time_since_drift=time_since_drift)
    elif mf_type == 'raw':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw(wt)
    elif mf_type == 'raw_temporal':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw_temporal(wt, wt_prev=wt_prev)
    elif mf_type == 'raw_delta':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw_delta(wt, wt_prev=wt_prev)
    elif mf_type == 'raw_cosine':
        def extract_mf(wt, wt_prev, drift_count, time_since_drift):
            return extract_metafeatures_raw_cosine(wt, wt_prev=wt_prev)
    else:
        raise ValueError(f"Unknown MF_TYPE: '{mf_type}'")
    return extract_mf


# ============================================================
#  HELPER — sigmoid threshold label assignment
#  Used for gradual drift only (25 concepts).
# ============================================================
 
def assign_labels_gradual(stream, config):
    """
    Assign concept labels using the sigmoid threshold method
    from Komorniczak et al.
    Produces 25 concepts for n_drifts=6, n_chunks=5000.
 
    Returns
    -------
    labels : list of int, length n_chunks
        Concept label for each chunk.
    """
    # sigmoid value per chunk: same as their e[chunk]
    e = stream._sigmoid(
        stream.concept_sigmoid_spacing, stream.n_drifts
    )[1][::config['chunk_size']]
 
    concept = 0
    decreasing = True
    labels = []
 
    for chunk in range(config['n_chunks']):
        # threshold logic from Komorniczak et al.
        if decreasing:
            if concept % 4 == 0:
                if e[chunk] < 0.9:
                    concept += 1
            if concept % 4 == 1:
                if e[chunk] < 0.75:
                    concept += 1
            if concept % 4 == 2:
                if e[chunk] < 0.25:
                    concept += 1
            if concept % 4 == 3:
                if e[chunk] < 0.1:
                    concept += 1
                    decreasing = False
        else:
            if concept % 4 == 0:
                if e[chunk] > 0.1:
                    concept += 1
            if concept % 4 == 1:
                if e[chunk] > 0.25:
                    concept += 1
            if concept % 4 == 2:
                if e[chunk] > 0.75:
                    concept += 1
            if concept % 4 == 3:
                if e[chunk] > 0.9:
                    concept += 1
                    decreasing = True
 
        labels.append(concept)
 
    return labels


# ============================================================
#  HELPER — extract meta-features for one stream
# ============================================================

def extract_metafeatures_for_stream(random_state, extract_mf):
    config = {
        'n_drifts': N_DRIFTS,
        'n_chunks': N_CHUNKS,
        'chunk_size': CHUNK_SIZE,
        'n_features': N_FEATURES,
        'n_informative': N_FEATURES,
        'n_redundant': 0,
        'n_repeated': 0,
        'concept_sigmoid_spacing': CONCEPT_SIGMOID_SPACING,
        'random_state': random_state
    }
 
    stream = StreamGenerator(**config)

    abfs = ABFS_match(
        n_features=N_FEATURES,
        categorical_features=[],
        accuracy_window_size=CHUNK_SIZE,
        class_window_size=CHUNK_SIZE
    )

    # pass 1: run ABFS, save concept_selector
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

    # concept labels: method depends on drift type
    if DRIFT_TYPE == 'sudden':
        # majority vote from concept_selector
        # equivalent to their threshold method for sudden drift
        concept_selector_saved = stream.concept_selector.copy()
 
    elif DRIFT_TYPE == 'gradual':
        # sigmoid threshold method matching Komorniczak et al.
        all_chunk_labels = assign_labels_gradual(stream, config)

    # pass 2: extract meta-features
    abfs = ABFS_match(
        n_features=N_FEATURES,
        categorical_features=[],
        accuracy_window_size=CHUNK_SIZE,
        class_window_size=CHUNK_SIZE
    )

    meta_features  = []
    concept_labels = []
    window_indices = []
    wt_prev = None
    window_counter = 0

    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])

        wt = abfs.relevance_scores()
        drift_count = abfs.pop_drift_count()

        if window_counter >= WARMUP_WINDOWS:
            mf = extract_mf(wt, wt_prev, drift_count, abfs.time_since_drift)
            meta_features.append(mf)
            window_indices.append(window_counter)

        wt_prev = wt
        window_counter += 1

    # assign labels after extraction
    for idx in window_indices:
        if DRIFT_TYPE == 'sudden':
            chunk_start = idx * CHUNK_SIZE
            chunk_end = min((idx + 1) * CHUNK_SIZE, len(concept_selector_saved))
            chunk_concepts = concept_selector_saved[chunk_start:chunk_end]
            concept_labels.append(int(np.bincount(chunk_concepts).argmax()))
        elif DRIFT_TYPE == 'gradual':
            concept_labels.append(all_chunk_labels[idx])

    X = np.array(meta_features, dtype=float)
    y = np.array(concept_labels)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1
    return X, y


# ============================================================
#  MAIN — sweep across all MF_TYPES
# ============================================================

all_mean_ba = {}
all_std_ba = {}

for mf_type, mf_label, n_mf in MF_CONFIGS:

    print(f"\n{'='*60}")
    print(f"Meta-features: {mf_label} ({n_mf}) | Drift: {DRIFT_TYPE}")
    print(f"{'='*60}")

    extract_mf = make_extract_mf(mf_type)
    all_clf_res = []

    for rep_id, rs in enumerate(RANDOM_STATES):
        print(f"Replication {rep_id+1}/{N_REPLICATIONS} (seed={rs})...")
        X, y = extract_metafeatures_for_stream(rs, extract_mf)

        mean_ba, std_ba, clf_res = run_classifier_sweep(X, y)
        all_clf_res.append(clf_res)

        print(f"{'Clf':<6s} {'Mean BA':>8s}")
        for clf_id, (name, _) in enumerate(BASE_CLFS):
            print(f"{name:<6s} {mean_ba[clf_id]:>8.4f}")

    all_clf_res = np.array(all_clf_res) # shape  (N_REPLICATIONS, n_folds, n_clfs)

    save_path = f'results/clf_{mf_type}_{DRIFT_TYPE}.npy'
    np.save(save_path, clf_res)
    print(f"Saved to {save_path}")

    all_mean_ba[mf_type] = np.mean(clf_res, axis=(0, 1))
    all_std_ba[mf_type] = np.std(clf_res,  axis=(0, 1))


# ============================================================
#  SUMMARY TABLE
# ============================================================

clf_names = [name for name, _ in base_clfs]

if DRIFT_TYPE == 'sudden':
    random_baseline = 1 / (N_DRIFTS + 1) # 1/21
    n_concepts = N_DRIFTS + 1
elif DRIFT_TYPE == 'gradual':
    random_baseline = 1 / 25 # 25 concepts with transitions
    n_concepts = 25   

print(f"\n{'='*60}")
print(f"Summary — {DRIFT_TYPE} drift")
print(f"{'='*60}")
print(f"\n{'Meta-features':<25s}", end='')
for name in clf_names:
    print(f"{name:>10s}", end='')
print()
print('-' * (25 + 10 * len(clf_names)))
for mf_type, mf_label, _ in MF_CONFIGS:
    print(f"{mf_label:<25s}", end='')
    for clf_id in range(len(base_clfs)):
        print(f"{all_mean_ba[mf_type][clf_id]:>10.4f}", end='')
    print()
print(f"\nKomorniczak et al. (GNB, sudden): 0.8660")
print(f"Random baseline (1/{n_concepts}):  {random_baseline:.4f}")




# ============================================================
#  HEATMAP — replicating Figure 12 of paper by Komorniczak et al.
# ============================================================

n_mf_sets = len(MF_CONFIGS)
n_clfs = len(base_clfs)
matrix = np.zeros((n_mf_sets, n_clfs))
row_labels = [label for _, label, _ in MF_CONFIGS]

for row_idx, (mf_type, _, _) in enumerate(MF_CONFIGS):
    matrix[row_idx] = all_mean_ba[mf_type]

fig, ax = plt.subplots(figsize=(8, 3.5))
im = ax.imshow(matrix, vmin=0.05, vmax=1.0, cmap='Blues', aspect='auto')

for i in range(n_mf_sets):
    for j in range(n_clfs):
        val = matrix[i, j]
        txt_color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}',
                ha='center', va='center',
                fontsize=11, color=txt_color)

ax.set_xticks(range(n_clfs))
ax.set_xticklabels(clf_names, fontsize=11)
ax.set_yticks(range(n_mf_sets))
ax.set_yticklabels(row_labels, fontsize=10)
n_concepts = 25 if DRIFT_TYPE == 'gradual' else N_DRIFTS + 1
ax.set_title(
    f'Concept classification — balanced accuracy\n'
    f'ABFS meta-features, {DRIFT_TYPE} drift '
    f'({n_concepts} concepts)',
    fontsize=11
)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
plt.tight_layout()

heatmap_path = f'results/figures/heatmap_{DRIFT_TYPE}.png'
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nHeatmap saved to {heatmap_path}")

