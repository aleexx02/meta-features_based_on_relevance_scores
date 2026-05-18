# ============================================================
#  Sanity check: can ABFS relevance scores detect concept drift?
#
#  Goals:
#    1. Verify that relevance scores change at drift moments
#    2. Verify that meta-features produce different vectors per concept
#    3. Verify that concepts are separable in meta-feature space (PCA)

# For StreamLearn streams, runs for all drift types
# (sudden, gradual) and all meta-feature sets (aggstats, raw, raw_temporal).
#
# For SEA/STAGGER streams, runs for all configured
# streams (SEA_SUDDEN, STG_01, STG_12, STG_02) using only aggstats
# meta-features.
#
# For each stream produces the following outputs:
#   1. relevance_scores_{stream_type}_{drift_type}.png
    #      ABFS relevance scores over time with drift markers.
    #      One per drift type since relevance scores do not depend
    #      on the meta-feature set, only on ABFS and the stream.
#   2. metafeatures_over_time_{stream_type}_{drift_type}_{mf_type}.png
    #      meta-feature evolution across windows with drift markers.
    #      One per meta-feature set per drift type.
#   3. pca_{stream_type}_{drift_type}_{mf_type}.png
    #      PCA projection of meta-feature vectors coloured by concept.
    #      One per meta-feature set per drift type.
#   4. Summary table printed to stdout: mean meta-feature values
#      per concept and absolute difference for binary concepts
#      (matches Table tab:sanity_initial in the report)

# Outputs saved to results/sanity_check/figures/
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
from plot_results import print_sanity_check_summary
from strlearn.streams import StreamGenerator


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# path to results folder
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results/sanity_check')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results/sanity_check', 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ================
#  CONFIGURATION
# ================
 
# set to True to run StreamLearn, False for SEA/STAGGER
RUN_STREAMLEARN = True

# StreamLearn configuration
SL_RANDOM_STATE = 42
SL_N_CHUNKS = 5000
SL_WINDOW_SIZE = 200
SL_N_FEATURES = 10
SL_WARMUP = 10
SL_SCORE_INTERVAL = 100

# SUDDEN: 6 concepts: 5 drifts with concept_sigmoid_spacing=9999 (sudden)
# GRADUAL: 6 concepts: 5 drifts with concept_sigmoid_spacing=5 (gradual)
SL_DRIFT_CONFIGS = [
    ('sudden', 5, 9999),
    ('gradual', 5, 5),
]

SL_MF_CONFIGS = [
    ('aggstats', MF_NAMES_AGGSTATS, 8),
    ('raw', MF_NAMES_RAW,  5),
    ('raw_temporal', MF_NAMES_RAW_TEMPORAL, 6),
]

# SEA / STAGGER configuration
DRIFT_POS = 5000
N_INSTANCES = 10000
N_FEATURES_SS = 3
SCORE_INTERVAL_SS = 100
WINDOW_SIZE_SS = 200

# streams to run: (stream_key, stream_type, drift_type, categorical_feats)
SEA_STAGGER_CONFIGS = [
    ('SEA_SUDDEN', 'SEA', 'sudden', []),
    ('STG_01', 'STAGGER', 'sudden_01', [0, 1, 2]),
    ('STG_12', 'STAGGER', 'sudden_12', [0, 1, 2]),
    ('STG_02', 'STAGGER', 'sudden_02', [0, 1, 2]),
]

# COLORS
palette = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#808000', '#c9a0dc', '#469990',
    '#7b3f91', '#9a6324', '#e6ac00', '#800000', '#2ecc71',
    '#556b2f', '#d2691e', '#000075', '#5e5151', '#08332b',
    '#000000'
]


# ============================================================
#  HELPERS
# ============================================================

def make_abfs_match(n_features, chunk_size):
    return ABFS_match(n_features=n_features, categorical_features=[], accuracy_window_size=chunk_size, class_window_size=chunk_size)


def make_abfs_mismatch(n_features, categorical_feats):
    return ABFS_mismatch(n_features=n_features, categorical_features=categorical_feats)


def plot_relevance_scores(scores_over_time, n_features, score_interval,drift_line, drift_moments, feature_names,title, filename):
    fig, ax = plt.subplots(figsize=(14, 4))
    for j in range(n_features):
        ax.plot(scores_over_time[:, j], label=f'{feature_names[j]}')
    if drift_line is not None:
        ax.axvline(x=drift_line // score_interval, color='red',linestyle='--', linewidth=1.5,label=f'true drift (instance {drift_line})')
    for dm in drift_moments:
        ax.axvline(x=dm // score_interval, color='orange',linestyle=':', linewidth=1, alpha=0.6)
    if drift_moments:
        ax.axvline(x=drift_moments[0] // score_interval, color='orange', linestyle=':', linewidth=1, alpha=0.6, label='detected drift')
    ax.set_xlabel(f'Time (x{score_interval} instances)')
    ax.set_ylabel('Relevance score')
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()
    print(f"\n*** ABFS relevance scores saved at:\n\t '{os.path.join(FIGURES_DIR, filename)}' ***")


def plot_metafeatures(meta_features, mf_names, n_mf_cols,drift_window, warmup, title, filename):
    fig, axes = plt.subplots(2, n_mf_cols, figsize=(4 * n_mf_cols, 6))
    axes = axes.flatten()
    for k, name in enumerate(mf_names):
        axes[k].plot(meta_features[:, k], color='steelblue')
        if drift_window is not None:
            dw = drift_window - warmup
            if dw > 0:
                axes[k].axvline(x=dw, color='red', linestyle='--', linewidth=1.0)
        axes[k].set_title(name, fontsize=9)
        axes[k].set_xlabel('Window')
        axes[k].set_ylabel('Value')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()
    print(f"\n*** Meta-features per window saved at:\n\t '{os.path.join(FIGURES_DIR, filename)}' ***")


def plot_pca(meta_features, concept_labels, title, filename):
    unique_concepts = np.unique(concept_labels)
    colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_concepts)}
    pca = PCA(n_components=2)
    projected = pca.fit_transform(meta_features)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in unique_concepts:
        mask = concept_labels == c
        label = f'concept {c}' if len(unique_concepts) > 2 else ('concept A (before drift)' if c == 0 else 'concept B (after drift)')
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   color=colors[c], label=label,
                   alpha=0.6, edgecolors='none', s=30)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150)
    plt.close()
    print(f"\n*** PCA plot saved at:\n\t '{os.path.join(FIGURES_DIR, filename)}' ***")



# ============================================================
#  STREAMLEARN STREAMS
# ============================================================

if RUN_STREAMLEARN:
    for drift_type, n_drifts, concept_sigmoid_spacing in SL_DRIFT_CONFIGS:
        print(f"\n{'='*60}")
        print(f"StreamLearn - {drift_type} drift (seed={SL_RANDOM_STATE})")
        print(f"{'='*60}")

        stream_type = f'StreamLearn_seed{SL_RANDOM_STATE}'

        config = {
            'n_drifts': n_drifts,
            'n_chunks': SL_N_CHUNKS,
            'chunk_size': SL_WINDOW_SIZE,
            'n_features': SL_N_FEATURES,
            'n_informative': SL_N_FEATURES,
            'n_redundant': 0,
            'n_repeated': 0,
            'concept_sigmoid_spacing': concept_sigmoid_spacing,
            'random_state': SL_RANDOM_STATE
        }

        stream = StreamGenerator(**config)

        # pass 1: relevance scores over time
        abfs = make_abfs_match(SL_N_FEATURES, SL_WINDOW_SIZE)
        scores_over_time = []
        instance_counter = 0
        stream.reset()
        for X_chunk, y_chunk in stream:
            for i in range(len(X_chunk)):
                abfs.update(X_chunk[i], y_chunk[i])
                if instance_counter % SL_SCORE_INTERVAL == 0:
                    scores_over_time.append(abfs.relevance_scores())
                instance_counter += 1
        concept_selector_saved = stream.concept_selector.copy()
        scores_over_time = np.array(scores_over_time)

        # concept labels and boundaries
        concept_labels_all = np.array([int(np.bincount(concept_selector_saved[i*SL_WINDOW_SIZE:(i+1)*SL_WINDOW_SIZE]).argmax())
            for i in range(SL_N_CHUNKS)])
        boundaries = [i for i in range(1, SL_N_CHUNKS) if concept_labels_all[i] != concept_labels_all[i-1]]

        # plot relevance scores, mark concept boundaries
        fig, ax = plt.subplots(figsize=(14, 4))
        for j in range(SL_N_FEATURES):
            ax.plot(scores_over_time[:, j], label=f'f{j+1}')
        for b in boundaries:
            ax.axvline(x=b * SL_WINDOW_SIZE // SL_SCORE_INTERVAL,
                       color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axvline(x=-1, color='red', linestyle='--', linewidth=1.0,
                   label='concept boundary')
        ax.set_xlabel(f'Time (x{SL_SCORE_INTERVAL} instances)')
        ax.set_ylabel('Relevance score')
        ax.set_title(f'ABFS relevance scores - StreamLearn {drift_type} (seed={SL_RANDOM_STATE})')
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        fname = f'relevance_scores_{stream_type}_{drift_type}.png'
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        plt.close()
        print(f"\n*** Plot of ABFS relevance scores for {drift_type} stream saved at:\n\t '{os.path.join(FIGURES_DIR, fname)}' ***")

        # pass 2: meta-features per MF type
        for mf_type, mf_names, n_mf_cols in SL_MF_CONFIGS:
            print(f"*** MF type: {mf_type} ***")
            print(f"{'-'*20}")

            abfs = make_abfs_match(SL_N_FEATURES, SL_WINDOW_SIZE)
            meta_features = []
            concept_labels = []
            wt_prev = None
            window_counter = 0

            stream.reset()
            for X_chunk, y_chunk in stream:
                for i in range(len(X_chunk)):
                    abfs.update(X_chunk[i], y_chunk[i])
                wt = abfs.relevance_scores()
                if window_counter >= SL_WARMUP:
                    if mf_type == 'aggstats':
                        mf = extract_metafeatures(wt=wt, wt_prev=wt_prev, drift_count=abfs.pop_drift_count(), time_since_drift=abfs.time_since_drift)
                    elif mf_type == 'raw':
                        mf = extract_metafeatures_raw(wt)
                    elif mf_type == 'raw_temporal':
                        mf = extract_metafeatures_raw_temporal(wt=wt, wt_prev=wt_prev)
                    meta_features.append(mf)
                    concept_labels.append(concept_labels_all[window_counter])
                wt_prev = wt
                window_counter += 1

            meta_features = np.array(meta_features)
            concept_labels = np.array(concept_labels)
            raw_vectors = meta_features[:, :SL_N_FEATURES] if mf_type in ('raw', 'raw_temporal') else np.array([])

            plot_metafeatures(meta_features, mf_names, n_mf_cols, boundaries[0] if boundaries else None,
                SL_WARMUP,title=f'Meta-features ({mf_type}) - StreamLearn {drift_type} (seed={SL_RANDOM_STATE})',
                filename=f'metafeatures_over_time_{stream_type}_{drift_type}_{mf_type}.png')

            plot_pca(meta_features, concept_labels,title=f'PCA ({mf_type}) - StreamLearn {drift_type} (seed={SL_RANDOM_STATE})',
                filename=f'pca_{stream_type}_{drift_type}_{mf_type}.png')

            print_sanity_check_summary(f'StreamLearn {drift_type} (seed={SL_RANDOM_STATE})',
                True, mf_type, mf_names, meta_features, concept_labels,raw_vectors, SL_N_FEATURES)



# ============================================================
#  SEA / STAGGER STREAMS
# ============================================================
else:
    def make_stream(stream_key, categorical_feats):
        if stream_key == 'SEA_SUDDEN':
            return make_sea_sudden_drift(drift_position=DRIFT_POS)
        elif stream_key == 'SEA_GRADUAL':
            return make_sea_gradual_drift(drift_position=DRIFT_POS)
        elif stream_key == 'STG_01':
            return make_stagger_sudden_drift_01(drift_position=DRIFT_POS)
        elif stream_key == 'STG_12':
            return make_stagger_sudden_drift_12(drift_position=DRIFT_POS)
        elif stream_key == 'STG_02':
            return make_stagger_sudden_drift_02(drift_position=DRIFT_POS)
        elif stream_key == 'STG_GRADUAL':
            return make_stagger_gradual_drift(drift_position=DRIFT_POS)
        elif stream_key == 'STG_RECURRING':
            return make_stagger_recurring()
        elif stream_key == 'STG_MULTI':
            return make_stagger_multi_drift()
        elif stream_key == 'SEA_STATIONARY':
            return make_sea_stationary()
        elif stream_key == 'SEA_MULTI':
            return make_sea_multi_drift()
        else:
            raise ValueError(f"Unknown stream: {stream_key}")

    for stream_key, stream_type, drift_type, categorical_feats in SEA_STAGGER_CONFIGS:
        print(f"\n{'='*60}")
        print(f"{stream_type} - {drift_type}")
        print(f"{'='*60}")

        mf_type = 'aggstats'
        mf_names = MF_NAMES_AGGSTATS
        n_mf_cols = 4

        feature_names = ['size', 'color', 'shape'] if categorical_feats == [0, 1, 2] else [f'f{j+1}' for j in range(N_FEATURES_SS)]

        stream = make_stream(stream_key, categorical_feats)

        abfs = make_abfs_mismatch(N_FEATURES_SS, categorical_feats)
        scores_over_time = []
        drift_moments = []
        meta_features = []
        concept_labels = []
        raw_vectors = []
        wt_prev = None
        instance_buffer = 0
        window_counter = 0

        for i, (x, y) in enumerate(stream):
            if i >= N_INSTANCES:
                break
            x_arr = np.array(list(x.values()))
            abfs.update(x_arr, y)
            if abfs.drift_count > 0:
                drift_moments.append(i)
            if i % SCORE_INTERVAL_SS == 0:
                scores_over_time.append(abfs.relevance_scores())
            instance_buffer += 1
            if instance_buffer == WINDOW_SIZE_SS:
                wt = abfs.relevance_scores()
                drift_count = abfs.pop_drift_count()
                mf = extract_metafeatures(wt=wt, wt_prev=wt_prev,drift_count=drift_count,time_since_drift=abfs.time_since_drift)
                meta_features.append(mf)
                concept_labels.append(0 if i < DRIFT_POS else 1)
                raw_vectors.append(wt)
                wt_prev = wt
                instance_buffer = 0
                window_counter += 1

        scores_over_time = np.array(scores_over_time)
        meta_features = np.array(meta_features)
        concept_labels = np.array(concept_labels)
        raw_vectors = np.array(raw_vectors)

        # drift window index
        drift_window = DRIFT_POS // WINDOW_SIZE_SS

        # relevance scores
        plot_relevance_scores(scores_over_time, N_FEATURES_SS, SCORE_INTERVAL_SS,drift_line=DRIFT_POS, drift_moments=drift_moments,
            feature_names=feature_names,title=f'ABFS relevance scores - {stream_type} {drift_type}',filename=f'relevance_scores_{stream_type}_{drift_type}_{mf_type}.png')

        # meta-features over windows
        plot_metafeatures(meta_features, mf_names, n_mf_cols,drift_window=drift_window, warmup=0,title=f'Meta-features ({mf_type}) - {stream_type} {drift_type}',
            filename=f'metafeatures_over_time_{stream_type}_{drift_type}_{mf_type}.png')

        # PCA
        plot_pca(meta_features, concept_labels,title=f'PCA ({mf_type}) - {stream_type} {drift_type}',
            filename=f'pca_{stream_type}_{drift_type}_{mf_type}.png')

        # summary table
        print_sanity_check_summary(f'{stream_type} {drift_type}', False, mf_type, mf_names,meta_features, concept_labels, raw_vectors, N_FEATURES_SS)