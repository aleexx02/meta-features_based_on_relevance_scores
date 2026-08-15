# ============================================================
#  Sanity check: can EMF relevance scores detect concept drift?
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
    #      EMF relevance scores over time with drift markers.
    #      One per drift type since relevance scores do not depend
    #      on the meta-feature set, only on EMF and the stream.
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
from collections import defaultdict
from river.datasets import synth as river_synth

from abfs.abfs_implementation import (
    EMF, ConfigResetWeightProp, ConfigReset, ConfigWeightProp,
)
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
RUN_STREAMLEARN = False

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


SEA_STAGGER_CONFIGS = [
    ('SEA_SUDDEN', 'SEA',     'sudden',    []),
    ('STG_12',     'STAGGER', 'sudden_12', [0, 1, 2]),
]
GROUND_TRUTH = {'SEA_SUDDEN': ({0,1}, {0,1}), 'STG_12': ({1,2}, {0})}

# Four settings of the SAME scoring mechanism (EMF), used to justify
# EMF's two design choices. Each is its own named class -- no flags.
# These are NOT "ABFS vs EMF": ABFS produces a feature subset, not scores.
CONFIGS = [
    ("orig",     ConfigResetWeightProp),  # reset + weight propagation (ABFS-style choices)
    ("noweight", ConfigReset),            # reset only (no weight propagation)
    ("noreset",  ConfigWeightProp),       # weight propagation only (no reset)
    ("emf",      EMF),                     # EMF: no reset, no weight propagation
]

# SEA / STAGGER configuration
DRIFT_POS = 5000
N_INSTANCES = 10000
N_FEATURES_SS = 3
SCORE_INTERVAL_SS = 100
WINDOW_SIZE_SS = 200


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

def make_emf(n_features, chunk_size):
    return EMF(n_features=n_features, categorical_features=[], accuracy_window_size=chunk_size, class_window_size=chunk_size)


def make_sea_sudden_drift(drift_position=5000, seed=42, noise=0.1):
    """SEA boundary drift: variant 0 (theta=8) -> variant 1 (theta=9).
       f1,f2 relevant, f3 irrelevant; only the decision threshold moves."""
    before = river_synth.SEA(variant=0, noise=noise, seed=seed)
    after  = river_synth.SEA(variant=1, noise=noise, seed=seed)
    for i, (x, y) in enumerate(before):
        if i >= drift_position:
            break
        yield x, int(y)
    for x, y in after:
        yield x, int(y)


def make_stagger_sudden_drift_12(drift_position=5000, seed=42):
    """STAGGER feature drift: concept 1 (color=green OR shape=circle)
       -> concept 2 (size in {medium,large}). Relevant feature set changes."""
    before = river_synth.STAGGER(classification_function=1, seed=seed)
    after  = river_synth.STAGGER(classification_function=2, seed=seed)
    for i, (x, y) in enumerate(before):
        if i >= drift_position:
            break
        yield x, int(y)
    for x, y in after:
        yield x, int(y)



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
    print(f"\n*** Relevance scores saved at:\n\t '{os.path.join(FIGURES_DIR, filename)}' ***")


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
        abfs = make_emf(SL_N_FEATURES, SL_WINDOW_SIZE)
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
        ax.set_title(f'EMF relevance scores - StreamLearn {drift_type} (seed={SL_RANDOM_STATE})')
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        fname = f'relevance_scores_{stream_type}_{drift_type}.png'
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
        plt.close()
        print(f"\n*** Plot of EMF relevance scores for {drift_type} stream saved at:\n\t '{os.path.join(FIGURES_DIR, fname)}' ***")

        # pass 2: meta-features per MF type
        for mf_type, mf_names, n_mf_cols in SL_MF_CONFIGS:
            print(f"*** MF type: {mf_type} ***")
            print(f"{'-'*20}")

            abfs = make_emf(SL_N_FEATURES, SL_WINDOW_SIZE)
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
        elif stream_key == 'STG_12':
            return make_stagger_sudden_drift_12(drift_position=DRIFT_POS)
        else:
            raise ValueError(f"Unknown stream: {stream_key}")

        
    for stream_key, stream_type, drift_type, categorical_feats in SEA_STAGGER_CONFIGS:
        print(f"\n{'='*60}")
        print(f"{stream_type} - {drift_type}")
        print(f"{'='*60}")

        mf_type   = 'aggstats'
        mf_names  = MF_NAMES_AGGSTATS
        n_mf_cols = 4
        feature_names = ['size', 'color', 'shape'] if categorical_feats == [0, 1, 2] \
            else [f'f{j+1}' for j in range(N_FEATURES_SS)]

        rows = []
        comparison = defaultdict(dict)

        for cfg_name, cfg_class in CONFIGS:
            print(f"\n  --- config: {cfg_name} ---")
            abfs = cfg_class(N_FEATURES_SS, categorical_features=categorical_feats)

            scores_over_time = []
            meta_features    = []
            concept_labels   = []
            raw_vectors      = [] 
            wt_prev          = None
            instance_buffer  = 0

            for i, (x, y) in enumerate(make_stream(stream_key, categorical_feats)):
                if i >= N_INSTANCES:
                    break
                abfs.update(np.array(list(x.values())), y)

                if i % SCORE_INTERVAL_SS == 0:
                    scores_over_time.append(abfs.relevance_scores())

                instance_buffer += 1
                if instance_buffer == WINDOW_SIZE_SS:
                    wt = abfs.relevance_scores()
                    mf = extract_metafeatures(wt=wt, wt_prev=wt_prev,
                                              drift_count=abfs.pop_drift_count(),
                                              time_since_drift=abfs.time_since_drift)
                    meta_features.append(mf)
                    concept_labels.append(0 if i < DRIFT_POS else 1)
                    raw_vectors.append(wt)
                    wt_prev = wt
                    instance_buffer = 0

            scores_over_time = np.array(scores_over_time)
            meta_features    = np.array(meta_features)
            concept_labels   = np.array(concept_labels)
            raw_vectors      = np.array(raw_vectors)
            
            # relevance scores — independent of any meta-feature version
            plot_relevance_scores(
                scores_over_time, N_FEATURES_SS, SCORE_INTERVAL_SS,
                drift_line=DRIFT_POS, drift_moments=[], feature_names=feature_names,
                title=f'Relevance scores - {stream_type} {drift_type} [{cfg_name}]',
                filename=f'relevance_scores_{stream_type}_{drift_type}_{cfg_name}.png')

            # PCA — on aggstats meta-features
            plot_pca(
                meta_features, concept_labels,
                title=f'PCA (aggstats) - {stream_type} {drift_type} [{cfg_name}]',
                filename=f'pca_{stream_type}_{drift_type}_{cfg_name}.png')

            if cfg_name == 'emf':
                print_sanity_check_summary(
                    f'{stream_type} {drift_type}', False, mf_type, mf_names,
                    meta_features, concept_labels, raw_vectors, N_FEATURES_SS)
                
            # comparison numbers
            dr = DRIFT_POS // SCORE_INTERVAL_SS
            rb, ra = GROUND_TRUTH[stream_key]
            a = scores_over_time[10:dr-2]; b = scores_over_time[dr+10:]
            ma, mb = a.mean(0), b.mean(0)
            allf = set(range(N_FEATURES_SS))
            sep = lambda m, rel: float(m[sorted(rel)].mean() - (m[sorted(allf-rel)].mean() if allf-rel else 0))
            rows.append((cfg_name, sep(ma, rb), sep(mb, ra),
                         float(np.linalg.norm(mb-ma)),
                         float(0.5*(a.std(0).mean()+b.std(0).mean()))))

            ma, mb = a.mean(0), b.mean(0)

            comparison[cfg_name]["before"] = ma
            comparison[cfg_name]["after"] = mb
            comparison[cfg_name]["dominant_before"] = feature_names[np.argmax(ma)]
            comparison[cfg_name]["dominant_after"]  = feature_names[np.argmax(mb)]


        print("\n")
        print("=" * 80)
        print(f"{stream_type} {drift_type} - Mean relevance scores")
        print("=" * 80)

        for phase in ["before", "after"]:

            print(f"\n{phase.upper()} DRIFT")
            header = f"{'config':12s}"
            for f in feature_names:
                header += f"{f:>12s}"
            print(header)

            for cfg_name in comparison:
                row = f"{cfg_name:12s}"
                vals = comparison[cfg_name][phase]

                for v in vals:
                    row += f"{v:12.3f}"

                print(row)

        print("\nEXPECTED RELEVANT FEATURES")
    
        if stream_key == "STG_12":
            print("Before drift: color, shape")
            print("After drift : size")
        elif stream_key == "SEA_SUDDEN":
            print("Before drift: f1, f2")
            print("After drift : f1, f2")


        print("\nDOMINANT FEATURES")
        print(f"{'config':12s}{'before':>15s}{'after':>15s}")

        for cfg_name in comparison:
            print(
                f"{cfg_name:12s}"
                f"{comparison[cfg_name]['dominant_before']:>15s}"
                f"{comparison[cfg_name]['dominant_after']:>15s}"
            )