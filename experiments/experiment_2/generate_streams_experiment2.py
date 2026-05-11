# ============================================================
# Generates synthetic streams for Experiment 2 and saves them as .npz files
# replication_check.py and evaluate_concept_classification.py load these streams
# to guarantee they use identical data.
#
# For each drift type and replication, saves a .npz file
# containing:
#   - X_chunks: shape (n_chunks, chunk_size, n_features)
#   - y_chunks: shape (n_chunks, chunk_size)
#   - concept_labels: shape (n_chunks,): one label per chunk
#
# Output: experiments/experiment_2/streams/stream_{drift_type}_rep{i}.npz
# ============================================================

import numpy as np
from strlearn.streams import StreamGenerator
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMS_DIR = os.path.join(SCRIPT_DIR, 'streams')
os.makedirs(STREAMS_DIR, exist_ok=True)

# ================
#  CONFIGURATION 
# ================

N_CHUNKS = 5000
CHUNK_SIZE = 200
N_FEATURES = 10
N_REPLICATIONS = 5

DRIFT_CONFIGS = [
    ('sudden',  20, 9999),
    ('gradual',  6,    5),
]

np.random.seed(1233)
RANDOM_STATES = np.random.randint(100, 10000, N_REPLICATIONS)
print(f"Random states: {RANDOM_STATES}")