#vanilla_cost_4.py

import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..'); sys.path.append('../..')
 
from vanilla_and_cost import ExperimentSpec, run_experiment
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from streams.generate_synthetic_streams import (
    exp4_specs, RIVER_N_FEATURES, SEED)
from pymfe.mfe import MFE
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
 
SPECS = {s['name']: s for s in exp4_specs()}
CELLS = list(SPECS.keys())
SEEDS = [SEED]
WARMUP = 0
 
def _rebuild(cell, seed):
    s = SPECS[cell]
    data, cpc = s['builder'](seed)
    return data, cpc, s
 
def window_provider(cell, seed):
    data, cpc, s = _rebuild(cell, seed)
    cs = s['chunk_size']
    windows, labels = [], []
    for w in range(len(cpc)):
        block = data[w*cs:(w+1)*cs, :-1]
        if len(block) == 0: break
        windows.append(np.asarray(block, dtype=float))
        labels.append(int(cpc[w]))
    return windows, labels
 
def cost_abfs(cell, seed):
    data, cpc, s = _rebuild(cell, seed)
    cs = s['chunk_size']; nf = RIVER_N_FEATURES[s['gen_name']]
    abfs = ABFS_match(n_features=nf, categorical_features=[],
                      accuracy_window_size=cs, class_window_size=cs)
    n = 0
    for w in range(len(cpc)):
        block = data[w*cs:(w+1)*cs]
        if len(block) == 0: break
        for row in block:
            abfs.update(row[:-1], int(row[-1]))
        _ = extract_metafeatures_raw(abfs.relevance_scores())
        n += 1
    return n
 
def cost_komor(cell, seed, measure='statistical'):
    data, cpc, s = _rebuild(cell, seed)
    cs = s['chunk_size']
    mfe = MFE(groups=[measure], suppress_warnings=True)
    n = 0
    for w in range(len(cpc)):
        block = data[w*cs:(w+1)*cs]
        if len(block) == 0: break
        try:
            mfe.fit(block[:, :-1], block[:, -1].astype(int))
            mfe.extract(suppress_warnings=True)
        except Exception:
            pass
        n += 1
    return n
 
def n_features_of(cell):
    return RIVER_N_FEATURES[SPECS[cell]['gen_name']]
 
if __name__ == '__main__':
    spec = ExperimentSpec('exp4', RESULTS_DIR, CELLS, SEEDS,
                          window_provider, cost_abfs, cost_komor, n_features_of)
    run_experiment(spec)