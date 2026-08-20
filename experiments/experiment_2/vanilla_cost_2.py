# vanilla_cost_2.py
# ==============================================================================
# Thin adapter: supplies Experiment 2's stream-loading to the shared
# vanilla_and_cost runner. Place in experiments/experiment_2/ and run.
# The other experiments follow the SAME pattern -- only the window_provider /
# cost functions change, because only the stream source changes.
# ==============================================================================

import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')     # experiments/
sys.path.append('../..')  # project root

from vanilla_and_cost import (
    ExperimentSpec,
    run_experiment,
    vanilla_features_from_windows
)

from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from streams.generate_synthetic_streams import (
    get_exp2_concept_labels as get_concept_labels,
    EXP2_N_CHUNKS       as N_CHUNKS,
    EXP2_N_FEATURES     as N_FEATURES,
    EXP2_CHUNK_SIZES    as CHUNK_SIZES,
    EXP2_N_INFORMATIVES as N_INFORMATIVES,
    EXP2_DRIFT_CONFIGS  as DRIFT_CONFIGS,
)
from pymfe.mfe import MFE

WARMUP = 10
np.random.seed(1233)
SEEDS = list(np.random.randint(100, 10000, 5))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')

# --- cell tag <-> config mapping (so a cell string can rebuild its stream) ---
CELL_CONFIG = {}   # tag -> (drift_type, n_drifts, spacing, n_concepts, chunk_size, n_informative)
for drift_type, n_drifts, spacing, n_concepts in DRIFT_CONFIGS:
    for cs in CHUNK_SIZES:
        for ni in N_INFORMATIVES:
            tag = f"chunk{cs}_ninf{ni}_{drift_type}"
            CELL_CONFIG[tag] = (drift_type, n_drifts, spacing, n_concepts, cs, ni)

CELLS = list(CELL_CONFIG.keys())


def _build_stream_and_labels(cfg, seed):
    drift_type, n_drifts, spacing, _, cs, ni = cfg
    config = dict(n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=cs,
                  n_features=N_FEATURES, n_informative=ni,
                  n_redundant=0, n_repeated=0,
                  concept_sigmoid_spacing=spacing, random_state=seed)
    stream = StreamGenerator(**config)
    # concept labels need a full ReMF pass first (matches the real evaluator)
    dummy = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                       accuracy_window_size=cs, class_window_size=cs)
    stream.reset()
    for Xc, yc in stream:
        for i in range(len(Xc)):
            dummy.update(Xc[i], yc[i])
    labels_all = get_concept_labels(stream, drift_type, N_CHUNKS, cs)
    return stream, labels_all


def window_provider(cell, seed):
    cfg = CELL_CONFIG[cell]
    stream, labels_all = _build_stream_and_labels(cfg, seed)
    windows, labels = [], []
    wc = 0
    stream.reset()
    for Xc, yc in stream:
        if wc >= WARMUP:
            windows.append(np.asarray(Xc, dtype=float))
            labels.append(labels_all[wc])
        wc += 1
    return windows, labels


def cost_vanilla(cell, seed):

    windows, _ = window_provider(cell, seed)

    def _extract():

        Xv = vanilla_features_from_windows(windows)

        return len(Xv)

    return _extract

def cost_remf(cell, seed):
    cfg = CELL_CONFIG[cell]
    drift_type, n_drifts, spacing, _, cs, ni = cfg
    config = dict(n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=cs,
                  n_features=N_FEATURES, n_informative=ni, n_redundant=0,
                  n_repeated=0, concept_sigmoid_spacing=spacing, random_state=seed)
    stream = StreamGenerator(**config)
 
    def _extract():
        remf = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                          accuracy_window_size=cs, class_window_size=cs)
        n = 0
        stream.reset()
        for Xc, yc in stream:
            for i in range(len(Xc)):
                remf.update(Xc[i], yc[i])
            _ = extract_metafeatures_raw(remf.relevance_scores())
            n += 1
        return n
 
    return _extract


def cost_komor(cell, seed, measure='statistical'):
    cfg = CELL_CONFIG[cell]
    drift_type, n_drifts, spacing, _, cs, ni = cfg
    config = dict(n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=cs,
                  n_features=N_FEATURES, n_informative=ni, n_redundant=0,
                  n_repeated=0, concept_sigmoid_spacing=spacing, random_state=seed)
    stream = StreamGenerator(**config)
 
    def _extract():
        mfe = MFE(groups=[measure], suppress_warnings=True)
        n = 0
        stream.reset()
        for Xc, yc in stream:
            try:
                mfe.fit(Xc, yc)
                mfe.extract(suppress_warnings=True)
            except Exception:
                pass
            n += 1
        return n
 
    return _extract


def n_features_of(cell):
    return N_FEATURES


if __name__ == '__main__':
    spec = ExperimentSpec(
    'exp2',
    RESULTS_DIR,
    CELLS,
    SEEDS,
    window_provider,
    cost_remf,
    cost_komor,
    cost_vanilla,
    n_features_of)

    run_experiment(spec, do_vanilla=False, do_cost=True)