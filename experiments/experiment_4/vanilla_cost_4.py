#vanilla_cost_4.py

import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))        # experiments/
sys.path.insert(0, os.path.join(_HERE, '..', '..'))  # repo root
 
from vanilla_and_cost import (
    ExperimentSpec,
    run_experiment,
    vanilla_features_from_windows
)
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from streams.generate_synthetic_streams import (
    exp4_specs, RIVER_N_FEATURES, SEED)
from pymfe.mfe import MFE
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
 
SPECS = {s['name']: s for s in exp4_specs()}
CELLS = list(SPECS.keys())
np.random.seed(SEED)
SEEDS = list(np.random.randint(100, 10000, 5))     # matches RANDOM_STATES in the evaluator
WARMUP = 0


COST_CELLS = [
    'sea_chunk100_ndrift1_sudden',
    'stagger_chunk100_ndrift1_sudden',   # <-- add
]


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

def cost_vanilla(cell, seed):

    windows, _ = window_provider(cell, seed)

    def _extract():

        Xv = vanilla_features_from_windows(windows)

        return len(Xv)

    return _extract

def cost_remf(cell, seed):
    data, cpc, s = _rebuild(cell, seed)
    cs = s['chunk_size']
    nf = RIVER_N_FEATURES[s['gen_name']]
 
    def _extract():
        remf = ABFS_match(n_features=nf, categorical_features=[],
                          accuracy_window_size=cs, class_window_size=cs)
        n = 0
        for w in range(len(cpc)):
            block = data[w*cs:(w+1)*cs]
            if len(block) == 0:
                break
            for row in block:
                remf.update(row[:-1], int(row[-1]))
            _ = extract_metafeatures_raw(remf.relevance_scores())
            n += 1
        return n
 
    return _extract
 
def cost_komor(cell, seed, measure='statistical'):
    data, cpc, s = _rebuild(cell, seed)
    cs = s['chunk_size']
 
    def _extract():
        mfe = MFE(groups=[measure], suppress_warnings=True)
        n = 0
        for w in range(len(cpc)):
            block = data[w*cs:(w+1)*cs]
            if len(block) == 0:
                break
            try:
                mfe.fit(block[:, :-1], block[:, -1].astype(int))
                mfe.extract(suppress_warnings=True)
            except Exception:
                pass
            n += 1
        return n
 
    return _extract
 
def n_features_of(cell):
    return RIVER_N_FEATURES[SPECS[cell]['gen_name']]
 
if __name__ == '__main__':
    spec = ExperimentSpec(
    'exp4',
    RESULTS_DIR,
    CELLS,
    SEEDS,
    window_provider,
    cost_remf,
    cost_komor,
    cost_vanilla,
    n_features_of,
    cost_cells=COST_CELLS
    )
    
    run_experiment(spec, do_vanilla=True, do_cost=True)