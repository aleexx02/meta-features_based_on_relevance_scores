import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))

import numpy as np
from strlearn.streams import StreamGenerator

# --- Experiment 1c: 10 features, all informative, chunk 200 ---
from streams.generate_synthetic_streams import (
    EXP2_N_CHUNKS, EXP2_DRIFT_CONFIGS)   # or exp1c's own N_CHUNKS/DRIFT_CONFIGS if different

drift_type, n_drifts, spacing, n_concepts = EXP2_DRIFT_CONFIGS[0]   # sudden

s = StreamGenerator(n_drifts=n_drifts, n_chunks=EXP2_N_CHUNKS, chunk_size=200,
                    n_features=10, n_informative=10,
                    n_redundant=0, n_repeated=0,
                    concept_sigmoid_spacing=spacing, random_state=8393)
s.reset()
X = np.vstack([Xc for Xc, yc in s])
print(f"\nExp 1c (stream-learn, 10 feat, all informative, chunk200):")
print(f"  overall min={X.min():.3f}  max={X.max():.3f}  mean={X.mean():.3f}")
print(f"  per-feature mean: {X.mean(axis=0).round(3)}")
print(f"  per-feature min:  {X.min(axis=0).round(3)}")
print(f"  per-feature max:  {X.max(axis=0).round(3)}")

# --- Experiment 2: 20 features, chunk 100, sweep n_informative ---
from streams.generate_synthetic_streams import EXP2_N_FEATURES, EXP2_N_INFORMATIVES

for ninf in EXP2_N_INFORMATIVES:          # 3, 5, 10, 15
    s = StreamGenerator(n_drifts=n_drifts, n_chunks=EXP2_N_CHUNKS, chunk_size=100,
                        n_features=EXP2_N_FEATURES, n_informative=ninf,
                        n_redundant=0, n_repeated=0,
                        concept_sigmoid_spacing=spacing, random_state=8393)
    s.reset()
    X = np.vstack([Xc for Xc, yc in s])
    print(f"\nExp 2 (stream-learn, 20 feat, n_informative={ninf}, chunk100):")
    print(f"  overall min={X.min():.3f}  max={X.max():.3f}  mean={X.mean():.3f}")
    print(f"  per-feature std:  {X.std(axis=0).round(3)}")


# --- river generators (Exp 3/4): SEA, STAGGER, LED ---
from streams.generate_synthetic_streams import exp3_specs, SEED
SPECS = {s['name']: s for s in exp3_specs()}

for cell in ['sea_chunk100_sudden', 'stagger_chunk100_sudden', 'led_chunk100_sudden']:
    spec = SPECS[cell]
    data, cpc = spec['builder'](SEED)
    X = data[:, :-1]
    print(f"\n{cell}:  ({X.shape[1]} features, {X.shape[0]} instances)")
    print(f"  overall min={X.min():.3f}  max={X.max():.3f}  mean={X.mean():.3f}")
    print(f"  per-feature mean: {X.mean(axis=0).round(3)}")
    print(f"  per-feature min:  {X.min(axis=0).round(3)}")
    print(f"  per-feature max:  {X.max(axis=0).round(3)}")


import os
import numpy as np
import strlearn as sl
from streams.generate_real_streams import REAL_STREAMS, N_FEATURES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')

# use whatever CHUNK_SIZE your Exp 5 uses; the range doesn't depend on it
CHUNK = 100

for name in ['INSECTS-abrupt_balanced', 'SPAM']:
    path = os.path.join(STREAM_DIR, f'{name}.npy')
    stream = sl.streams.NPYParser(path, chunk_size=CHUNK, n_chunks=100000)
    chunks = []
    for _ in range(100000):
        try:
            Xc, yc = stream.get_chunk()
        except Exception:
            break
        if len(Xc) == 0:
            break
        chunks.append(Xc)
    X = np.vstack(chunks)
    nz = (X != 0).mean()          # fraction of non-zero entries (sparsity check)
    print(f"\n{name}:  ({X.shape[1]} features, {X.shape[0]} instances)")
    print(f"  overall min={X.min():.3f}  max={X.max():.3f}  mean={X.mean():.3f}")
    print(f"  fraction non-zero: {nz:.3f}")