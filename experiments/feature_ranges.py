# feature_ranges.py
# ==============================================================================
# Measures the raw feature ranges (min/max/mean/std + sparsity) of every stream
# used in the thesis, across all five experiments. Writes one CSV summarising
# all of them.
#
# Purpose: the concept-distance numbers (concept_distances_expN.csv) are measured
# on the RAW feature values, whose scale differs per generator. This script
# documents those scales so the distances can be interpreted correctly, and it
# also reports the fraction of non-zero entries (relevant to the SPAM
# sparsity-vs-dimensionality question).
#
# Place in experiments/ and run ON THE CLUSTER (the real streams of Experiment 5
# are pre-saved .npy files that only exist there).
# ==============================================================================

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np

# experiments/ is one level below the repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

rows = []


def record(label, X, n_features):
    nz = float((X != 0).mean())
    r = dict(stream=label, n_features=int(n_features),
             min=round(float(X.min()), 3), max=round(float(X.max()), 3),
             mean=round(float(X.mean()), 3), std=round(float(X.std()), 3),
             frac_nonzero=round(nz, 3))
    rows.append(r)
    print(f"{label:34s}  {int(n_features):3d} feat  "
          f"range [{r['min']:.2f}, {r['max']:.2f}]  "
          f"mean {r['mean']:.3f}  std {r['std']:.3f}  "
          f"nonzero {r['frac_nonzero']:.3f}")


# ---------------------------------------------------------------- stream-learn (1c, 2)
from strlearn.streams import StreamGenerator
from streams.generate_synthetic_streams import (
    EXP2_N_CHUNKS, EXP2_DRIFT_CONFIGS, EXP2_N_FEATURES)

drift_type, n_drifts, spacing, _ = EXP2_DRIFT_CONFIGS[0]   # sudden


def sl_stream(nf, ninf, cs):
    s = StreamGenerator(n_drifts=n_drifts, n_chunks=EXP2_N_CHUNKS, chunk_size=cs,
                        n_features=nf, n_informative=ninf,
                        n_redundant=0, n_repeated=0,
                        concept_sigmoid_spacing=spacing, random_state=8393)
    s.reset()
    return np.vstack([Xc for Xc, yc in s])


print("=== stream-learn (Experiments 1c, 2) ===")
record('exp1c stream-learn (10f, 10inf)', sl_stream(10, 10, 200), 10)
record('exp2 stream-learn (20f, 10inf)',  sl_stream(20, 10, 100), 20)


# ---------------------------------------------------------------- river (3, 4)
print("\n=== river (Experiments 3, 4) ===")
from streams.generate_synthetic_streams import exp3_specs, SEED
SPECS = {s['name']: s for s in exp3_specs()}
for cell in ['sea_chunk100_sudden', 'stagger_chunk100_sudden', 'led_chunk100_sudden']:
    data, cpc = SPECS[cell]['builder'](SEED)
    gen = cell.split('_')[0]
    record(f'exp3 {gen}', data[:, :-1], data.shape[1] - 1)


# ---------------------------------------------------------------- real (5)
print("\n=== real (Experiment 5) ===")
import strlearn as sl
from streams.generate_real_streams import REAL_STREAMS
STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')

for name in REAL_STREAMS:
    path = os.path.join(STREAM_DIR, f'{name}.npy')
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    stream = sl.streams.NPYParser(path, chunk_size=100, n_chunks=100000)
    chunks = []
    for _ in range(100000):
        try:
            Xc, yc = stream.get_chunk()
        except Exception:
            break
        if len(Xc) == 0:
            break
        chunks.append(Xc)
    if chunks:
        X = np.vstack(chunks)
        record(f'exp5 {name}', X, X.shape[1])


# ---------------------------------------------------------------- save
out_dir = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'feature_ranges_all_experiments.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {out}")