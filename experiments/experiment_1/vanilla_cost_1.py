# vanilla_cost_1.py

import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..'); sys.path.append('../..')
 
from vanilla_and_cost import ExperimentSpec, run_experiment
from strlearn.streams import StreamGenerator
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
# ADJUST these imports to the real 1 config exposed by generate_synthetic_streams:
from streams.generate_synthetic_streams import (
    get_exp2_concept_labels as get_concept_labels,   # 1 uses same labeler
    EXP2_DRIFT_CONFIGS as DRIFT_CONFIGS,   # 1 baseline: use the drift configs
)
from pymfe.mfe import MFE
 
WARMUP = 10
CHUNK_SIZE_1 = 200          # 1 reference chunk size (adjust if different)
N_INFORMATIVE_1 = 10        # 1 baseline n_informative (adjust)
N_FEATURES = 10
N_CHUNKS = 5000


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_1')
 
np.random.seed(1233)
SEEDS = list(np.random.randint(100, 10000, 5))
 
# one cell per drift type (the 1 baseline)
CELL_CONFIG = {}
for drift_type, n_drifts, spacing, n_concepts in DRIFT_CONFIGS:
    tag = f"{drift_type}"
    CELL_CONFIG[tag] = (drift_type, n_drifts, spacing, n_concepts)
CELLS = list(CELL_CONFIG.keys())
 
def _build(cfg, seed):
    drift_type, n_drifts, spacing, _ = cfg
    config = dict(n_drifts=n_drifts, n_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE_1,
                  n_features=N_FEATURES, n_informative=N_INFORMATIVE_1,
                  n_redundant=0, n_repeated=0,
                  concept_sigmoid_spacing=spacing, random_state=seed)
    stream = StreamGenerator(**config)
    dummy = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                       accuracy_window_size=CHUNK_SIZE_1, class_window_size=CHUNK_SIZE_1)
    stream.reset()
    for Xc, yc in stream:
        for i in range(len(Xc)):
            dummy.update(Xc[i], yc[i])
    labels_all = get_concept_labels(stream, drift_type, N_CHUNKS, CHUNK_SIZE_1)
    return stream, labels_all
 
def window_provider(cell, seed):
    stream, labels_all = _build(CELL_CONFIG[cell], seed)
    windows, labels = [], []; wc = 0
    stream.reset()
    for Xc, yc in stream:
        if wc >= WARMUP:
            windows.append(np.asarray(Xc, dtype=float))
            labels.append(labels_all[wc])
        wc += 1
    return windows, labels
 
def cost_abfs(cell, seed):
    # stream built HERE, outside the timed closure — so tracemalloc measures
    # only the extraction, not the stream array
    stream, _ = _build(CELL_CONFIG[cell], seed)

    def _extract():
        abfs = ABFS_match(n_features=N_FEATURES, categorical_features=[],
                          accuracy_window_size=CHUNK_SIZE_1,
                          class_window_size=CHUNK_SIZE_1)
        n = 0
        stream.reset()
        for Xc, yc in stream:
            for i in range(len(Xc)):
                abfs.update(Xc[i], yc[i])
            _ = extract_metafeatures_raw(abfs.relevance_scores())
            n += 1
        return n

    return _extract
 
def cost_komor(cell, seed, measure='statistical'):
    stream, _ = _build(CELL_CONFIG[cell], seed)

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
    spec = ExperimentSpec('exp1', RESULTS_DIR, CELLS, SEEDS,
                          window_provider, cost_abfs, cost_komor, n_features_of)
    run_experiment(spec, do_vanilla=False, do_cost=True)
