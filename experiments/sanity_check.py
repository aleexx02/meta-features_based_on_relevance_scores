# ============================================================
#  Sanity check: can ABFS relevance scores detect concept drift?
#
#  Goals:
#    1. Verify that relevance scores change at drift moments
#    2. Verify that meta-features produce different vectors per concept
#    3. Verify that concepts are separable in meta-feature space (PCA)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
 
from streams.generators import (
    make_sea_sudden_drift, make_sea_gradual_drift,
    make_stagger_sudden_drift_01, make_stagger_gradual_drift,
    make_stagger_recurring, make_sea_stationary,
    make_sea_multi_drift, make_stagger_multi_drift,
    make_stagger_sudden_drift_02, make_stagger_sudden_drift_12
)
from abfs.abfs_implementation import ABFS_mismatch, ABFS_match
from metafeatures.mf_extraction import (
    extract_metafeatures, extract_metafeatures_raw, extract_metafeatures_raw_temporal,
    MF_NAMES_AGGSTATS, MF_NAMES_RAW, MF_NAMES_RAW_TEMPORAL
)
from strlearn.streams import StreamGenerator
 
os.makedirs('results/figures', exist_ok=True)


# ================
#  CONFIGURATION
# ================
 
# 1. Stream Selector
# options:
#   StreamLearn:  'SL_SUDDEN'  | 'SL_GRADUAL'
#   SEA:          'SEA_SUDDEN' | 'SEA_GRADUAL' | 'SEA_STATIONARY' | 'SEA_MULTI'
#   STAGGER:      'STG_01'     | 'STG_02'      | 'STG_12'
#                 'STG_GRADUAL'| 'STG_RECURRING'| 'STG_MULTI'
STREAM = 'SL_GRADUAL'
# whether this is a StreamLearn stream (chunk-based) or
# a SEA/STAGGER stream (instance-based, single drift)
IS_STREAMLEARN = STREAM.startswith('SL_')


# 2. Meta-feature Selector
# 'raw': 10 normalised raw relevance scores (use for StreamLearn)
# 'aggstats': 8 aggregate statistics (use for SEA / STAGGER)
MF_TYPE = None
 
# 3. Parameters for StreamLearn and SEA/STAGGER are separated below since they differ in how drift is generated and how time is measured (chunks vs instances).
# Common parameters for both StreamLearn and SEA/STAGGER
WINDOW_SIZE = 200
SCORE_INTERVAL = 100

# SL_SUDDEN: 21 concepts: 20 drifts with concept_sigmoid_spacing=9999 (sudden)
# SL_GRADUAL: 6 concepts: 5 drifts with concept_sigmoid_spacing=5 (gradual)
# StreamLearn parameters (only used for SL_SUDDEN / SL_GRADUAL)
if IS_STREAMLEARN:
    ABFS_CLASS = ABFS_match
    N_CHUNKS = 5000
    N_DRIFTS = 5   # 6 concepts
    RANDOM_STATE = 42
    N_FEATURES = 10
    WARMUP_WINDOWS = 10 # first 10 windows (2000 instances) are warmup — no meta-features extracted
    #MF_TYPE = 'raw' # 10 normalised raw relevance scores
    MF_TYPE = 'raw_temporal' # 10 normalised raw relevance scores + 2 temporal features (delta_mean, cosine_sim)
# SEA / STAGGER parameters (only used for non-StreamLearn streams)
else:
    DRIFT_POS = 5000 # instance index of drift point
    N_INSTANCES = 10000 # total instances to process
    N_FEATURES = 3 # number of features in SEA / STAGGER streams
    WARMUP_WINDOWS = 0 # no warmup for SEA / STAGGER since they have a single drift and we want to see the full trajectory
    MF_TYPE = 'aggstats' # 8 aggregate statistics (use for SEA / STAGGER)
    ABFS_CLASS = ABFS_mismatch # default to mismatch.
    # options: ABFS_match (accuracy_window = chunk_size), ABFS_mismatch (accuracy_window = 2000)



# ==============
#  STREAM SETUP
# ==============
# StreamLearn streams
# ==============

def make_stream():
    """Create a fresh stream instance. Call this before each iteration pass."""
    if STREAM == 'SL_SUDDEN':
        return StreamGenerator(
            n_drifts=N_DRIFTS, n_chunks=N_CHUNKS, chunk_size=WINDOW_SIZE,
            n_features=N_FEATURES, n_informative=N_FEATURES,
            n_redundant=0, n_repeated=0, random_state=RANDOM_STATE
        ), f"StreamLearn Sudden Drift (seed={RANDOM_STATE})", []

    elif STREAM == 'SL_GRADUAL':
        return StreamGenerator(
            n_drifts=N_DRIFTS, n_chunks=N_CHUNKS, chunk_size=WINDOW_SIZE,
            n_features=N_FEATURES, n_informative=N_FEATURES,
            n_redundant=0, n_repeated=0,
            concept_sigmoid_spacing=5, random_state=RANDOM_STATE
        ), f"StreamLearn Gradual Drift (seed={RANDOM_STATE})", []

    elif STREAM == 'SEA_SUDDEN':
        return make_sea_sudden_drift(drift_position=DRIFT_POS), \
               "SEA Sudden Boundary Drift", []

    elif STREAM == 'SEA_GRADUAL':
        return make_sea_gradual_drift(drift_position=DRIFT_POS), \
               "SEA Gradual Boundary Drift", []

    elif STREAM == 'SEA_STATIONARY':
        return make_sea_stationary(), "SEA Stationary", []

    elif STREAM == 'SEA_MULTI':
        return make_sea_multi_drift(), "SEA Multiple Boundary Drifts", []

    elif STREAM == 'STG_01':
        return make_stagger_sudden_drift_01(drift_position=DRIFT_POS), \
               "STAGGER Sudden Feature Drift (01)", [0, 1, 2]

    elif STREAM == 'STG_02':
        return make_stagger_sudden_drift_02(drift_position=DRIFT_POS), \
               "STAGGER Sudden Feature Drift (02)", [0, 1, 2]

    elif STREAM == 'STG_12':
        return make_stagger_sudden_drift_12(drift_position=DRIFT_POS), \
               "STAGGER Sudden Feature Drift (12)", [0, 1, 2]

    elif STREAM == 'STG_GRADUAL':
        return make_stagger_gradual_drift(drift_position=DRIFT_POS), \
               "STAGGER Gradual Feature Drift", [0, 1, 2]

    elif STREAM == 'STG_RECURRING':
        return make_stagger_recurring(), \
               "STAGGER Recurring Feature Drift", [0, 1, 2]

    elif STREAM == 'STG_MULTI':
        return make_stagger_multi_drift(), \
               "STAGGER Multiple Feature Drifts", [0, 1, 2]

    else:
        raise ValueError(f"Unknown STREAM: '{STREAM}'")


# Create stream
stream, stream_name, categorical_feats = make_stream()


# ==============
#  ABFS SETUP
# ==============
# ABFS_match: accuracy_window = chunk_size (StreamLearn)
# ABFS_mismatch: accuracy_window = 2000 (SEA / STAGGER)

def make_abfs(abfs_class, n_features, categorical_feats, chunk_size):
    if abfs_class == ABFS_match:
        return ABFS_match(n_features=n_features, categorical_features=categorical_feats,
        accuracy_window_size=chunk_size, class_window_size=chunk_size)
    elif abfs_class == ABFS_mismatch:
        return ABFS_mismatch(n_features=n_features, categorical_features=categorical_feats)
 

# Meta-feature function
if MF_TYPE == 'raw':
    def extract_mf(wt, wt_prev, drift_count, time_since_drift):
        return extract_metafeatures_raw(wt)
    MF_NAMES = MF_NAMES_RAW
    n_mf_cols = 5 # 2 x 5 = 10 subplots
    FEATURE_NAMES = [f'f{j+1}' for j in range(N_FEATURES)]
 
elif MF_TYPE == 'aggstats':
    def extract_mf(wt, wt_prev, drift_count, time_since_drift):
        return extract_metafeatures(wt=wt, wt_prev=wt_prev, drift_count=drift_count,
        time_since_drift=time_since_drift)
    MF_NAMES = MF_NAMES_AGGSTATS
    n_mf_cols = 4 # 2 x 4 = 8 subplots
    # feature names for plot legends
    if categorical_feats == [0, 1, 2]:
        FEATURE_NAMES = ['size', 'color', 'shape']
    else:
        FEATURE_NAMES = [f'f{j+1}' for j in range(N_FEATURES)]

elif MF_TYPE == 'raw_temporal':
    def extract_mf(wt, wt_prev, drift_count, time_since_drift):
        return extract_metafeatures_raw_temporal(wt=wt, wt_prev=wt_prev)
    MF_NAMES = MF_NAMES_RAW_TEMPORAL
    n_mf_cols = 6 # 2 x 6 = 12 subplots (10 raw + 2 temporal features)
    FEATURE_NAMES = [f'f{j+1}' for j in range(N_FEATURES)]
 
else:
    raise ValueError(f"Unknown MF_TYPE: '{MF_TYPE}'")


 
print(f"\nRunning sanity check:")
print(f"Stream: {stream_name}")
print(f"ABFS: {'ABFS_match' if IS_STREAMLEARN else 'ABFS_mismatch'}")
print(f"Meta-features: {MF_TYPE} ({len(MF_NAMES)} features)")


# ============================================================
#  STEP 3: run ABFS and track relevance scores over time
# ============================================================
 
abfs = make_abfs(ABFS_CLASS, N_FEATURES, categorical_feats, WINDOW_SIZE)
scores_over_time = []
drift_moments = [] # instance indices where drift was detected
instance_counter = 0
 
if IS_STREAMLEARN:
    stream.reset()
    for X_chunk, y_chunk in stream:
        for i in range(len(X_chunk)):
            abfs.update(X_chunk[i], y_chunk[i])
            if instance_counter % SCORE_INTERVAL == 0:
                scores_over_time.append(abfs.relevance_scores())
            instance_counter += 1
    # save before any reset — stream.reset() wipes concept_selector to zeros
    concept_selector_saved = stream.concept_selector.copy()
 
else:
    for i, (x, y) in enumerate(stream):
        if i >= N_INSTANCES:
            break
        x_arr = np.array(list(x.values()))
        abfs.update(x_arr, y)
        if abfs.drift_count > 0:
            drift_moments.append(i)
        if i % SCORE_INTERVAL == 0:
            scores_over_time.append(abfs.relevance_scores())
        instance_counter += 1
 
scores_over_time = np.array(scores_over_time)

 
# ============================================================
#  STEP 4: plot relevance scores over time
# ============================================================
 
fig, ax = plt.subplots(figsize=(14, 4))
for j in range(N_FEATURES):
    ax.plot(scores_over_time[:, j], label=f'{FEATURE_NAMES[j]} feature')
 
if IS_STREAMLEARN:
    # concept_selector is per instance: index with chunk_idx * WINDOW_SIZE
    prev_concept = int(np.bincount(concept_selector_saved[0:WINDOW_SIZE]).argmax())
    for chunk_idx in range(1, N_CHUNKS):
        chunk_start = chunk_idx * WINDOW_SIZE
        chunk_end = min((chunk_idx + 1) * WINDOW_SIZE, len(concept_selector_saved))
        if chunk_start >= len(concept_selector_saved):
            break
        chunk_concepts = concept_selector_saved[chunk_start:chunk_end]
        curr_concept = int(np.bincount(chunk_concepts).argmax())
        if curr_concept != prev_concept:
            ax.axvline(x=chunk_start // SCORE_INTERVAL, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            prev_concept = curr_concept
else:

    # true drift position on the x-axis (converted to score_interval units)
    ax.axvline(
        x=DRIFT_POS // SCORE_INTERVAL,
        color='red', linestyle='--', linewidth=1.5,
        label=f'true drift (instance {DRIFT_POS})'
    )

    # detected drift moments
    for dm in drift_moments:
        ax.axvline(
            x=dm // SCORE_INTERVAL,
            color='orange', linestyle=':', linewidth=1,
            alpha=0.6
        )
    if drift_moments:
        ax.axvline(
            x=drift_moments[0] // SCORE_INTERVAL,
            color='orange', linestyle=':', linewidth=1,
            alpha=0.6, label='detected drift'
        )

 
ax.axvline(x=-1, color='red', linestyle='--', linewidth=1.5, label='concept boundary')
ax.set_xlabel('Time (x100 instances)')
ax.set_ylabel('Relevance score')
ax.set_title(f'ABFS relevance scores over time — {stream_name}')
ax.legend()
fig.tight_layout()
fig.savefig(f'results/figures/relevance_scores_{stream_name.replace(" ", "_")}_{MF_TYPE}.png', dpi=150)
plt.show()
print(f"Plot saved: results/figures/relevance_scores_{stream_name.replace(" ", "_")}_{MF_TYPE}.png")
 




# ============================================================
#  STEP 5: extract meta-features per window
# ============================================================
 
abfs = make_abfs(ABFS_CLASS, N_FEATURES, categorical_feats, WINDOW_SIZE)
wt_prev = None
meta_features = []
concept_labels = []
window_indices = []
raw_vectors = []
window_counter = 0
 
if IS_STREAMLEARN:
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
            raw_vectors.append(wt)
 
        wt_prev = wt
        window_counter += 1
 
    # assign labels after full iteration using saved concept_selector
    # concept_selector is per instance: index with window * WINDOW_SIZE
    # majority vote across all instances in the chunk to assign concept label to the window
    for idx in window_indices:
        chunk_start = idx * WINDOW_SIZE
        chunk_end = min((idx + 1) * WINDOW_SIZE, len(concept_selector_saved))
        chunk_concepts = concept_selector_saved[chunk_start:chunk_end]
        concept_labels.append(int(np.bincount(chunk_concepts).argmax()))
 
else:
    stream, _, _ = make_stream()
    instance_buffer = 0
    for i, (x, y) in enumerate(stream):
        if i >= N_INSTANCES:
            break

        x_arr = np.array(list(x.values()))
        
        abfs.update(x_arr, y)
        instance_buffer += 1

        if instance_buffer == WINDOW_SIZE:
            wt = abfs.relevance_scores()
            drift_count = abfs.pop_drift_count()
            mf = extract_mf(wt, wt_prev, drift_count, abfs.time_since_drift)

            meta_features.append(mf)
            concept_labels.append(0 if i < DRIFT_POS else 1)
            window_indices.append(window_counter)
            raw_vectors.append(wt)

            wt_prev = wt
            instance_buffer = 0
            window_counter += 1
 
meta_features = np.array(meta_features)
concept_labels = np.array(concept_labels)
raw_vectors = np.array(raw_vectors)
unique_concepts = np.unique(concept_labels)
 
print(f"Unique concept labels: {list(unique_concepts)}")
 
 
# ============================================================
#  STEP 6: plot meta-features over windows
# ============================================================
 
fig, axes = plt.subplots(2, n_mf_cols, figsize=(4 * n_mf_cols, 6))
axes = axes.flatten()

for k, name in enumerate(MF_NAMES):
    axes[k].plot(meta_features[:, k], color='steelblue')
 
    if IS_STREAMLEARN:
        prev_concept = int(np.bincount(concept_selector_saved[WARMUP_WINDOWS*WINDOW_SIZE:(WARMUP_WINDOWS+1)*WINDOW_SIZE]).argmax())
        for chunk_idx in range(WARMUP_WINDOWS + 1, N_CHUNKS):
            chunk_start = chunk_idx * WINDOW_SIZE
            chunk_end = min((chunk_idx + 1) * WINDOW_SIZE, len(concept_selector_saved))
            if chunk_start >= len(concept_selector_saved):
                break
            chunk_concepts = concept_selector_saved[chunk_start:chunk_end]
            curr_concept = int(np.bincount(chunk_concepts).argmax())
            if curr_concept != prev_concept:
                drift_w = chunk_idx - WARMUP_WINDOWS
                axes[k].axvline(x=drift_w, color='red', linestyle='--', linewidth=1.5)
                prev_concept = curr_concept
    else:
        axes[k].axvline(x=DRIFT_POS // WINDOW_SIZE, color='red', linestyle='--', linewidth=1.5)
 
    axes[k].set_title(name, fontsize=10)
    axes[k].set_xlabel('Window')
    axes[k].set_ylabel('Value')
 
fig.suptitle(f'Meta-features over windows — {stream_name}', fontsize=12)
fig.tight_layout()
fig.savefig(f'results/figures/metafeatures_over_time_{stream_name.replace(" ", "_")}_{MF_TYPE}.png', dpi=150)
plt.show()
print(f"Plot saved: results/figures/metafeatures_over_time_{stream_name.replace(" ", "_")}_{MF_TYPE}.png")
 
 
# ============================================================
#  STEP 7: PCA
# ============================================================
 
pca = PCA(n_components=2)
projected = pca.fit_transform(meta_features)
if IS_STREAMLEARN:
    palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', "#5e5151", "#08332b",
    '#000000'
]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_concepts)}
else:
    colors = {0: 'steelblue', 1: 'coral'}
 
fig, ax = plt.subplots(figsize=(8, 5))
for c in unique_concepts:
    mask  = concept_labels == c
    label = f'concept {c}' if IS_STREAMLEARN else \
            ('concept A (before drift)' if c == 0 else 'concept B (after drift)')
    ax.scatter(projected[mask, 0], projected[mask, 1],
               color=colors[c], label=label,
               alpha=0.6, edgecolors='none', s=30)
 
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title(f'Meta-feature vectors projected to 2D — {stream_name}')
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig(f'results/figures/pca_{stream_name.replace(" ", "_")}_{MF_TYPE}.png', dpi=150)
plt.show()
print(f'Plot saved: results/figures/pca_{stream_name.replace(" ", "_")}_{MF_TYPE}.png')
 

# ============================================================
#  STEP 8: summary
# ============================================================
 
print(f"\n{'='*60}")
print(f"Sanity check summary")
print(f"{'='*60}")
print(f"Stream: {stream_name}")
print(f"ABFS: {'ABFS_match' if IS_STREAMLEARN else 'ABFS_mismatch'}")
print(f"Meta-features: {MF_TYPE} ({len(MF_NAMES)} features)")
print(f"Total windows: {len(meta_features)}")
print(f"Unique concepts: {len(unique_concepts)} {list(unique_concepts)}")
 
print(f"\nMeta-feature means per concept:")
print(f"{'':22s}", end='')
for c in unique_concepts:
    print(f"{'concept '+str(c):>14s}", end='')
print()
for k, name in enumerate(MF_NAMES):
    print(f"  {name:<22s}", end='')
    for c in unique_concepts:
        mean_val = meta_features[concept_labels == c, k].mean()
        print(f"{mean_val:>14.4f}", end='')
    print()
if raw_vectors.shape[0] > 0 and raw_vectors.ndim == 2: 
    print(f"\nMean raw relevance score per feature per concept:")
    print(f"{'':14s}", end='')
    for c in unique_concepts:
        print(f"{'concept '+str(c):>12s}", end='')
    print()
    for j in range(N_FEATURES):
        print(f"  f{j+1:<10d}", end='')
        for c in unique_concepts:
            mean_val = raw_vectors[concept_labels == c, j].mean()
            print(f"{mean_val:>12.4f}", end='')
        print()
else:
    print(f"\nRaw relevance vectors not available for {MF_TYPE} meta-features.")




# # ---------------- CONFIGURATION ---------------- 
# N_INSTANCES = 10000
# N_DRIFTS = 20
# DRIFT_POS = 5000
# WINDOW_SIZE = 200
# N_FEATURES = 3
# SCORE_INTERVAL = 100   # record scores every N instances

# STREAM = "SUDDEN_FEATURE_DRIFT12" # options: "SUDDEN_BOUNDARY_DRIFT", "GRADUAL_BOUNDARY_DRIFT", "SUDDEN_FEATURE_DRIFT01", "GRADUAL_FEATURE_DRIFT", "RECURRING_FEATURE_DRIFT", "STATIONARY", "MULTI_BOUNDARY", "MULTI_FEATURE", "SUDDEN_FEATURE_DRIFT02", "SUDDEN_FEATURE_DRIFT12"

# # ----------------   STREAM GENERATION ----------------
# if STREAM == 'SUDDEN_BOUNDARY_DRIFT':
#     stream = make_sea_sudden_drift(drift_position = DRIFT_POS)
#     stream_name = "SEA Sudden Boundary Drift"
# elif STREAM == 'GRADUAL_BOUNDARY_DRIFT':
#     stream = make_sea_gradual_drift(drift_position = DRIFT_POS)
#     stream_name = "SEA Gradual Boundary Drift"
# elif STREAM == 'SUDDEN_FEATURE_DRIFT01':
#     stream = make_stagger_sudden_drift_01(drift_position = DRIFT_POS)
#     stream_name = "STAGGER Sudden Feature Drift (01)"
# elif STREAM == 'GRADUAL_FEATURE_DRIFT':
#     stream = make_stagger_gradual_drift(drift_position = DRIFT_POS)
#     stream_name = "STAGGER Gradual Feature Drift"
# elif STREAM == 'RECURRING_FEATURE_DRIFT':
#     stream = make_stagger_recurring()
#     stream_name = "STAGGER Recurring Feature Drift"
# elif STREAM == 'STATIONARY':
#     stream = make_sea_stationary()
#     stream_name = "SEA Stationary"
# elif STREAM == 'MULTI_BOUNDARY':
#     stream = make_sea_multi_drift()
#     stream_name = "SEA Multiple Boundary Drifts"
# elif STREAM == 'MULTI_FEATURE':
#     stream = make_stagger_multi_drift()
#     stream_name = "STAGGER Multiple Feature Drifts"
# elif STREAM == 'SUDDEN_FEATURE_DRIFT02':
#     stream = make_stagger_sudden_drift_02(drift_position = DRIFT_POS)
#     stream_name = "STAGGER Sudden Feature Drift (02)"
# elif STREAM == 'SUDDEN_FEATURE_DRIFT12':
#     stream = make_stagger_sudden_drift_12(drift_position = DRIFT_POS)
#     stream_name = "STAGGER Sudden Feature Drift (12)"



# # ---------------- FEATURE TYPE CONFIGURATION ----------------
# # ABFS needs to know which features are categorical (for STAGGER) vs numeric (for SEA).
# IS_CATEGORICAL = STREAM in [
#     'SUDDEN_FEATURE_DRIFT01',
#     'SUDDEN_FEATURE_DRIFT02',
#     'SUDDEN_FEATURE_DRIFT12',
#     'GRADUAL_FEATURE_DRIFT',
#     'RECURRING_FEATURE_DRIFT',
#     'MULTI_FEATURE'
# ]
# CATEGORICAL_FEATURES = [0, 1, 2] if IS_CATEGORICAL else []

# # feature names for plot legends
# # descriptive for STAGGER (categorical), generic for SEA (numeric)
# if IS_CATEGORICAL:
#     FEATURE_NAMES = ['size', 'color', 'shape']
# else:
#     FEATURE_NAMES = ['f1', 'f2', 'f3']



# # ----------------  step 3 — run ABFS and track scores over time ---------------- 
# scores_over_time = [] # relevance score snapshot every SCORE_INTERVAL
# drift_moments = [] # instance indices where drift was detected
# abfs = ABFS_mismatch(n_features=N_FEATURES)
# for i, (x, y) in enumerate(stream):
#     if i >= N_INSTANCES:
#         break
    
#     x_arr = np.array(list(x.values()))

#     abfs.update(x_arr, y)

#     if abfs.drift_count > 0:
#         drift_moments.append(i)
    
#     if i % SCORE_INTERVAL == 0:
#         scores_over_time.append(abfs.relevance_scores())

# scores_over_time = np.array(scores_over_time)


# # ----------------  step 4 — plot relevance scores over time ---------------- 
# fig, ax = plt.subplots(figsize=(12, 4))

# for j in range(N_FEATURES):
#     ax.plot(scores_over_time[:, j], label=f'{FEATURE_NAMES[j]} feature')

# # true drift position on the x-axis (converted to score_interval units)
# ax.axvline(
#     x=DRIFT_POS // SCORE_INTERVAL,
#     color='red', linestyle='--', linewidth=1.5,
#     label=f'true drift (instance {DRIFT_POS})'
# )

# # # detected drift moments
# # for dm in drift_moments:
# #     ax.axvline(
# #         x=dm // SCORE_INTERVAL,
# #         color='orange', linestyle=':', linewidth=1,
# #         alpha=0.6
# #     )
# # if drift_moments:
# #     ax.axvline(
# #         x=drift_moments[0] // SCORE_INTERVAL,
# #         color='orange', linestyle=':', linewidth=1,
# #         alpha=0.6, label='detected drift'
# #     )

# ax.set_xlabel('Time (x100 instances)')
# ax.set_ylabel('Relevance score')
# ax.set_title(f'ABFS relevance scores over time — {stream_name} stream')
# ax.legend()
# fig.tight_layout()
# fig.savefig(f'results/figures/relevance_scores_{stream_name.replace(" ", "_")}.png')
# fig.show()
# print(f"Plot saved: results/figures/relevance_scores_{stream_name.replace(" ", "_")}.png")


# # ----------------  step 5 — extract meta-features per window and plot PCA ---------------- 
# # re-generate stream 
# if STREAM == 'SUDDEN_BOUNDARY_DRIFT':
#     stream = make_sea_sudden_drift(drift_position=DRIFT_POS)
# elif STREAM == 'GRADUAL_BOUNDARY_DRIFT':
#     stream = make_sea_gradual_drift(drift_position=DRIFT_POS)
# elif STREAM == 'SUDDEN_FEATURE_DRIFT01':
#     stream = make_stagger_sudden_drift_01(drift_position=DRIFT_POS)
# elif STREAM == 'GRADUAL_FEATURE_DRIFT':
#     stream = make_stagger_gradual_drift(drift_position = DRIFT_POS)
# elif STREAM == 'RECURRING_FEATURE_DRIFT':
#     stream = make_stagger_recurring()
# elif STREAM == 'STATIONARY':
#     stream = make_sea_stationary()
# elif STREAM == 'MULTI_BOUNDARY':
#     stream = make_sea_multi_drift()
# elif STREAM == 'MULTI_FEATURE':
#     stream = make_stagger_multi_drift()
# elif STREAM == 'SUDDEN_FEATURE_DRIFT02':
#     stream = make_stagger_sudden_drift_02(drift_position = DRIFT_POS)
# elif STREAM == 'SUDDEN_FEATURE_DRIFT12':
#     stream = make_stagger_sudden_drift_12(drift_position = DRIFT_POS)

# # step 5 — reinstantiate after regenerating stream
# abfs = ABFS_mismatch(n_features=N_FEATURES, categorical_features=CATEGORICAL_FEATURES)    
# wt_prev = None
# meta_features  = [] # one meta-feature vector per window
# concept_labels = [] # 0 = before drift, 1 = after drift
# window_indices = [] # which window number
# window_counter = 0
# instance_buffer = 0

# for i, (x, y) in enumerate(stream):
#     if i >= N_INSTANCES:
#         break

#     x_arr = np.array(list(x.values()))
    
#     abfs.update(x_arr, y)
#     instance_buffer += 1

#     if instance_buffer == WINDOW_SIZE:
#         wt = abfs.relevance_scores()
#         drift_count = abfs.pop_drift_count()      
        
#         mf = extract_metafeatures(wt=wt, wt_prev=wt_prev, drift_count=drift_count, time_since_drift=abfs.time_since_drift)

#         meta_features.append(mf)
#         concept_labels.append(0 if i < DRIFT_POS else 1)
#         window_indices.append(window_counter)

#         wt_prev = wt
#         instance_buffer = 0
#         window_counter += 1

# meta_features  = np.array(meta_features)
# concept_labels = np.array(concept_labels)


# # if instance_buffer == WINDOW_SIZE:
# #         wt = abfs.relevance_scores()
# #         drift_count = abfs.pop_drift_count()

# #         # only extract meta-features after warmup period
# #         # first WARMUP_WINDOWS windows have incomplete scoring window
# #         if window_counter >= WARMUP_WINDOWS:
# #             mf = extract_metafeatures(
# #                 wt=wt, wt_prev=wt_prev,
# #                 drift_count=drift_count,
# #                 time_since_drift=abfs.time_since_drift
# #             )
# #             meta_features.append(mf)
# #             concept_labels.append(0 if i < DRIFT_POS else 1)
# #             window_indices.append(window_counter)

# #         # always update wt_prev — even during warmup
# #         # so delta_mean is correct from the first kept window
# #         wt_prev = wt
# #         instance_buffer = 0
# #         window_counter += 1


# # ---------------- plot each meta-feature over windows ----------------
# fig, axes = plt.subplots(2, 4, figsize=(16, 6))
# axes = axes.flatten()

# drift_window = DRIFT_POS // WINDOW_SIZE

# for k, name in enumerate(MF_NAMES_AGGSTATS):
#     axes[k].plot(meta_features[:, k], color='steelblue')
#     axes[k].axvline(
#         x=drift_window,
#         color='red', linestyle='--', linewidth=1.5
#     )
#     axes[k].set_title(name, fontsize=10)
#     axes[k].set_xlabel('Window')
#     axes[k].set_ylabel('Value')

# fig.suptitle(f'Meta-features over windows — {stream_name} stream', fontsize=12)
# fig.tight_layout()
# fig.savefig(f'results/figures/metafeatures_over_time_{stream_name.replace(" ", "_")}.png', dpi=150)
# plt.show()
# print(f"Plot saved: results/figures/metafeatures_over_time_{stream_name.replace(" ", "_")}.png")


# # ---------------- PCA: do concepts cluster in meta-feature space? ----------------
# pca = PCA(n_components=2)
# projected = pca.fit_transform(meta_features)

# fig, ax = plt.subplots(figsize=(7, 5))
# colors = {0: 'steelblue', 1: 'coral'}
# labels = {0: 'concept A (before drift)', 1: 'concept B (after drift)'}

# for concept in [0, 1]:
#     mask = concept_labels == concept
#     ax.scatter(
#         projected[mask, 0],
#         projected[mask, 1],
#         c=colors[concept],
#         label=labels[concept],
#         alpha=0.75,
#         edgecolors='none',
#         s=40
#     )

# ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
# ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
# ax.set_title(f'Meta-feature vectors projected to 2D — {stream_name} stream')
# ax.legend()
# fig.tight_layout()
# fig.savefig(f'results/figures/pca_concept_separability_{stream_name.replace(" ", "_")}.png', dpi=150)
# plt.show()
# print(f"Plot saved: results/figures/pca_concept_separability_{stream_name.replace(" ", "_")}.png")


# # ----------------  step 6 — summary ----------------
# print("\n---------------- Sanity check summary ----------------")
# print(f"Total windows:       {len(meta_features)}")
# print(f"Windows concept A:   {(concept_labels == 0).sum()}")
# print(f"Windows concept B:   {(concept_labels == 1).sum()}")
# print(f"Drift detected at:   {drift_moments[:3]} (first 3 signals)")
# print(f"\nMeta-feature means per concept:")
# print(f"{'':20s} {'concept A (BEFORE DRIFT)':>12s} {'concept B (AFTER DRIFT)':>12s}")
# for k, name in enumerate(MF_NAMES_AGGSTATS):
#     mA = meta_features[concept_labels == 0, k].mean()
#     mB = meta_features[concept_labels == 1, k].mean()
#     print(f"  {name:<20s} {mA:>12.4f} {mB:>12.4f}")







# # ---------------- CONFIGURATION ----------------
# N_CHUNKS = 5000
# CHUNK_SIZE = 200
# N_FEATURES = 10
# N_DRIFTS = 5
# RANDOM_STATE = 42
# SCORE_INTERVAL = 100
# WARMUP_WINDOWS = 10
# DRIFT_TYPE = 'sudden'  # options: 'sudden', 'gradual'

# # ---------------- STREAM GENERATION ----------------
# if DRIFT_TYPE == 'sudden':
#     stream = StreamGenerator(n_drifts=N_DRIFTS, n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE, n_features=N_FEATURES, n_informative=N_FEATURES,
#     n_redundant=0, n_repeated=0, random_state=RANDOM_STATE)
#     stream_name = f"StreamLearn Sudden Drift (seed={RANDOM_STATE})"

# elif DRIFT_TYPE == 'gradual':
#     stream = StreamGenerator(n_drifts=6, n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE, n_features=N_FEATURES, n_informative=N_FEATURES,
#     n_redundant=0, n_repeated=0, concept_sigmoid_spacing=5, random_state=RANDOM_STATE)
#     stream_name = f"StreamLearn Gradual Drift (seed={RANDOM_STATE})"

# FEATURE_NAMES = [f'f{j+1}' for j in range(N_FEATURES)]

# # ---------------- step 3 — run ABFS and track scores over time ----------------
# abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[], accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
# scores_over_time = []
# instance_counter = 0

# stream.reset()
# for X_chunk, y_chunk in stream:
#     for i in range(len(X_chunk)):
#         abfs.update(X_chunk[i], y_chunk[i])
#         if instance_counter % SCORE_INTERVAL == 0:
#             scores_over_time.append(abfs.relevance_scores())
#         instance_counter += 1

# scores_over_time = np.array(scores_over_time)

# # concept_selector is fully populated here: we save it
# concept_selector_saved = stream.concept_selector.copy()


# # ----------------  step 4 — plot relevance scores over time ----------------
# fig, ax = plt.subplots(figsize=(14, 4))

# for j in range(N_FEATURES):
#     ax.plot(scores_over_time[:, j],
#             label=f'f{j+1}', linewidth=0.8, alpha=0.7)

# # mark true concept boundaries — check every CHUNK_SIZE instances
# prev_concept = int(concept_selector_saved[0])
# for chunk_idx in range(1, N_CHUNKS):
#     instance_idx = chunk_idx * CHUNK_SIZE
#     if instance_idx >= len(concept_selector_saved):
#         break
#     curr_concept = int(concept_selector_saved[instance_idx])
#     if curr_concept != prev_concept:
#         ax.axvline(x=instance_idx // SCORE_INTERVAL, color='red', linestyle='--',
#         linewidth=1.5, alpha=0.8)
#         prev_concept = curr_concept
        

# ax.axvline(x=-1, color='red', linestyle='--', linewidth=1.5, label='concept boundary')
# ax.set_xlabel('Time (x100 instances)')
# ax.set_ylabel('Relevance score')
# ax.set_title(f'ABFS relevance scores — {stream_name}')
# ax.legend(ncol=6, fontsize=8)
# fig.tight_layout()
# fig.savefig(
#     f'results/figures/relevance_scores_{stream_name.replace(" ", "_")}.png',
#     dpi=150)
# plt.show()
# print(f"Plot saved.")


# # ---------------- step 5 — extract meta-features per window ----------------
# abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[], accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
# wt_prev = None
# meta_features = []
# concept_labels = []
# window_indices = []
# raw_vectors = []
# window_counter = 0

# stream.reset()


# for X_chunk, y_chunk in stream:
#     for i in range(len(X_chunk)):
#         abfs.update(X_chunk[i], y_chunk[i])

#     wt = abfs.relevance_scores()
#     drift_count = abfs.pop_drift_count()

#     if window_counter >= WARMUP_WINDOWS:
#         # mf = extract_metafeatures( wt=wt, wt_prev=wt_prev, drift_count=drift_count,
#         # time_since_drift=abfs.time_since_drift)
#         mf = extract_metafeatures_raw(wt)
#         meta_features.append(mf)
#         window_indices.append(window_counter)
#         raw_vectors.append(wt)

#     wt_prev = wt
#     window_counter += 1

# for idx in window_indices:
#     concept_labels.append(int(concept_selector_saved[idx * CHUNK_SIZE]))

# # sanity check
# print(f"Unique concept labels: {np.unique(concept_labels)}")
# print(f"Expected: {{0, 1, 2, 3, 4, 5}}")


# meta_features = np.array(meta_features)
# concept_labels = np.array(concept_labels)
# raw_vectors = np.array(raw_vectors)

# # ---------------- plot each meta-feature over windows ----------------
# fig, axes = plt.subplots(2, 5, figsize=(20, 6))
# axes = axes.flatten()

# for k, name in enumerate(MF_NAMES_RAW):
#     axes[k].plot(meta_features[:, k], color='steelblue')

#     # mark concept boundaries — check every CHUNK_SIZE instances
#     prev_concept = int(concept_selector_saved[WARMUP_WINDOWS * CHUNK_SIZE])  # start checking after warmup
#     for chunk_idx in range(WARMUP_WINDOWS + 1, N_CHUNKS):
#         instance_idx = chunk_idx * CHUNK_SIZE
#         if instance_idx >= len(concept_selector_saved):
#             break

#         curr_concept = int(concept_selector_saved[instance_idx])
#         if curr_concept != prev_concept:
#             drift_w = chunk_idx - WARMUP_WINDOWS
#             axes[k].axvline(x=drift_w, color='red', linestyle='--', linewidth=1.5)
#             prev_concept = curr_concept

#     axes[k].set_title(name, fontsize=10)
#     axes[k].set_xlabel('Window')
#     axes[k].set_ylabel('Value')

# fig.suptitle(f'Meta-features over windows — {stream_name}', fontsize=12)
# fig.tight_layout()
# fig.savefig(
#     f'results/figures/metafeatures_over_time_{stream_name.replace(" ", "_")}.png',
#     dpi=150)
# plt.show()
# print(f"Plot saved.")

# # ---------------- PCA ----------------
# pca = PCA(n_components=2)
# projected = pca.fit_transform(meta_features)
# unique_concepts = np.unique(concept_labels)
# cmap = plt.cm.get_cmap('tab10', len(unique_concepts))

# fig, ax = plt.subplots(figsize=(8, 5))
# for c in unique_concepts:
#     mask = concept_labels == c
#     ax.scatter(projected[mask, 0], projected[mask, 1],
#                color=cmap(c), label=f'concept {c}',
#                alpha=0.6, edgecolors='none', s=30)

# ax.set_xlabel(
#     f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
# ax.set_ylabel(
#     f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
# ax.set_title(
#     f'Meta-feature vectors projected to 2D — {stream_name}')
# ax.legend(ncol=4, fontsize=8)
# fig.tight_layout()
# fig.savefig(
#     f'results/figures/pca_{stream_name.replace(" ", "_")}.png',
#     dpi=150)
# plt.show()
# print(f"Plot saved.")

# # ---------------- summary ----------------
# print(f"\n---------------- Sanity check summary ----------------")
# print(f"Stream:              {stream_name}")
# print(f"Total windows:       {len(meta_features)}")
# print(f"Unique concepts:     {len(unique_concepts)}")
# print(f"\nMeta-feature means per concept:")
# print(f"{'':20s}", end='')
# for c in unique_concepts:
#     print(f"{'concept '+str(c):>14s}", end='')
# print()
# for k, name in enumerate(MF_NAMES_RAW):
#     print(f"  {name:<20s}", end='')
#     for c in unique_concepts:
#         mean_val = meta_features[concept_labels == c, k].mean()
#         print(f"{mean_val:>14.4f}", end='')
#     print()