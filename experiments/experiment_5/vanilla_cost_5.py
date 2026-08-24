#vanilla_cost_5.py

import os, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.append('..'); sys.path.append('../..')
 
import strlearn as sl
from vanilla_and_cost import (
    ExperimentSpec,
    run_experiment,
    vanilla_features_from_windows
)
from abfs.abfs_implementation import ABFS_match
from metafeatures.mf_extraction import extract_metafeatures_raw
from streams.generate_real_streams import REAL_STREAMS, N_FEATURES, CHUNK_SIZE
from pymfe.mfe import MFE
 
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
REAL_STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')
REAL_GT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_gt')
RESULTS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'experiment_5')
 
CELLS = list(REAL_STREAMS)          # stream names are the 'cells'
SEEDS = [0]                          # single realization; seed value unused for real data
 
def _drift_chunks(stream_name):
    return np.load(os.path.join(REAL_GT_DIR, f'{stream_name}.npy'))
 
def window_provider(cell, seed):
    stream_path = os.path.join(REAL_STREAM_DIR, f'{cell}.npy')
    drift_chunks = _drift_chunks(cell)
    stream = sl.streams.NPYParser(stream_path, chunk_size=CHUNK_SIZE, n_chunks=100000)
    windows, labels = [], []
    for chunk_idx in range(100000):
        concept = int(np.sum(drift_chunks <= chunk_idx))
        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break
        if len(X_chunk) == 0:
            break
        windows.append(np.asarray(X_chunk, dtype=float))   # features (target is separate)
        labels.append(concept)
    return windows, labels


def _read_chunks(cell):
    """Read all (X, y) chunks for a real stream once, OUTSIDE any timer."""
    stream_path = os.path.join(REAL_STREAM_DIR, f'{cell}.npy')
    stream = sl.streams.NPYParser(stream_path, chunk_size=CHUNK_SIZE, n_chunks=100000)
    chunks = []
    for _ in range(100000):
        try:
            X_chunk, y_chunk = stream.get_chunk()
        except Exception:
            break
        if len(X_chunk) == 0:
            break
        chunks.append((np.asarray(X_chunk, float), np.asarray(y_chunk)))
    return chunks


def cost_vanilla(cell, seed):
    chunks = _read_chunks(cell)

    def _extract():
        n = 0

        for X_chunk, _ in chunks:

            feat = np.concatenate([
                X_chunk.mean(axis=0),
                X_chunk.std(axis=0)
            ])

            feat[np.isnan(feat)] = 1
            feat[np.isinf(feat)] = 1

            n += 1

        return n

    return _extract


def cost_remf(cell, seed):
    chunks = _read_chunks(cell)          # file I/O OUTSIDE timer
    nf = N_FEATURES[cell]
    def _extract():
        remf = ABFS_match(n_features=nf, categorical_features=[],
                          accuracy_window_size=CHUNK_SIZE, class_window_size=CHUNK_SIZE)
        n = 0
        for X_chunk, y_chunk in chunks:
            for i in range(len(X_chunk)):
                remf.update(X_chunk[i], y_chunk[i])
            _ = extract_metafeatures_raw(remf.relevance_scores())
            n += 1
        return n
    return _extract


def cost_komor(cell, seed, measure='statistical'):
    chunks = _read_chunks(cell)          # file I/O OUTSIDE timer
    def _extract():
        mfe = MFE(groups=[measure], suppress_warnings=True)
        n = 0
        for X_chunk, y_chunk in chunks:
            try:
                mfe.fit(X_chunk, y_chunk)
                mfe.extract(suppress_warnings=True)
            except Exception:
                pass
            n += 1
        return n
    return _extract


 
def n_features_of(cell):
    return N_FEATURES[cell]
 
if __name__ == '__main__':
    spec = ExperimentSpec(
    'exp5',
    RESULTS_DIR,
    CELLS,
    SEEDS,
    window_provider,
    cost_remf,
    cost_komor,
    cost_vanilla,
    n_features_of,
    cost_cells=CELLS
    )
    
    run_experiment(spec, do_vanilla=True, do_cost=True)