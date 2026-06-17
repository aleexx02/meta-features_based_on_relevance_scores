# streams/generate_semi_synthetic_streams.py
# ============================================================
# I prepare semi-synthetic streams with INJECTED drift for a separate
# experiment (Experiment 4), distinct from the genuinely annotated
# INSECTS streams in generate_real_streams.py.
#
# Why these streams need injected drift:
#   Electricity and covtype have no published ground truth drift
#   location anywhere in the literature I could verify. Multiple
#   independent benchmark papers confirm this — for example Lukats
#   et al. (2024) explicitly separate real-world streams into "known
#   drift ground truth" (the INSECTS family) vs. "unknown ground
#   truth" (electricity, covtype, NOAA, and others), placing
#   electricity and covtype in the unknown group. Using them as if
#   their natural drift were annotated would mean fabricating a claim
#   I cannot verify.
#
# What I do instead:
#   I sort instances by class label into contiguous blocks. Each block
#   boundary is then a drift point BY CONSTRUCTION — fully known,
#   because I created it. This is the same idea used to build
#   poker-lsn from the otherwise driftless poker hand dataset
#   (Losing et al., 2016). The "concept" here is just the class label
#   itself, since each block is, by construction, dominated by one
#   class.
#
#   This is NOT a claim about real concept drift in electricity or
#   covtype — it tests whether ABFS and Komorniczak meta-features can
#   recover known, controlled drift injected into real feature
#   distributions, as opposed to the fully synthetic feature
#   distributions used in Experiments 1 and 2.
#
# Source: USP DS Repository (Souza et al., 2020), "Old datasets" folder
# https://sites.google.com/view/uspdsrepository
#
# Output format:
#   data/semi_synthetic/streams/{stream}.npy
#       shape: (n_instances, n_features + 1)
#       last column = integer concept label (0-indexed)
#       features min-max normalised to [0, 1] where needed
#
#   data/semi_synthetic/streams_gt/{stream}.npy
#       shape: (n_drifts,)
#       chunk indices where a block boundary (injected drift) occurs
# ============================================================
 
import numpy as np
import os
import collections
 
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
 
USP_OLD = os.path.expanduser(
    '~/usp_ds_repository/USP DS Repository/Old datasets')
 
OUT_STREAMS = os.path.join(PROJECT_ROOT, 'data', 'semi_synthetic', 'streams')
OUT_GT      = os.path.join(PROJECT_ROOT, 'data', 'semi_synthetic', 'streams_gt')
 
os.makedirs(OUT_STREAMS, exist_ok=True)
os.makedirs(OUT_GT,      exist_ok=True)
 
CHUNK_SIZE = 200  # consistent with the rest of the project
 
 
# ============================================================
#  HELPERS
# ============================================================
 
def load_csv(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return [l.strip() for l in lines if l.strip()]
 
 
def minmax_normalise(X):
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    rng     = col_max - col_min
    rng[rng == 0] = 1.0
    return (X - col_min) / rng
 
 
def already_done(stream_name):
    sp = os.path.join(OUT_STREAMS, f'{stream_name}.npy')
    gp = os.path.join(OUT_GT,      f'{stream_name}.npy')
    if os.path.exists(sp) and os.path.exists(gp):
        d = np.load(sp)
        print(f"  EXISTS : {stream_name}.npy  shape={d.shape}")
        return True
    return False
 
 
def save_stream(stream_name, X, y, normalise=False):
    if normalise:
        X = minmax_normalise(X)
    result = np.hstack([X, y.reshape(-1, 1)])
    np.save(os.path.join(OUT_STREAMS, f'{stream_name}.npy'), result)
    counts = dict(collections.Counter(y.tolist()))
    print(f"  SAVED  : {stream_name}.npy")
    print(f"           shape={result.shape}  "
          f"chunks@{CHUNK_SIZE}={result.shape[0]//CHUNK_SIZE}  "
          f"classes={counts}")
    return result
 
 
def save_gt(stream_name, drift_chunks):
    np.save(os.path.join(OUT_GT, f'{stream_name}.npy'),
           np.array(drift_chunks))
    print(f"  GT     : drift_chunks={drift_chunks}  "
          f"n_concepts={len(drift_chunks)+1}  "
          f"random_baseline={1/(len(drift_chunks)+1):.3f}")
 
 
def build_sorted_drift_stream(X, y_raw):
    """
    Sort instances by class label to create contiguous blocks, each
    block being one "concept" (the class itself). Returns
    (X_sorted, y_concept, drift_chunks).
    """
    classes = sorted(np.unique(y_raw))
    order   = np.argsort(y_raw, kind='stable')
    X_sorted = X[order]
    y_sorted_raw = y_raw[order]
 
    label_map = {c: i for i, c in enumerate(classes)}
    y_concept = np.array([label_map[c] for c in y_sorted_raw])
 
    n_chunks = X_sorted.shape[0] // CHUNK_SIZE
    chunk_concepts = [
        int(np.bincount(
            y_concept[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]).argmax())
        for i in range(n_chunks)
    ]
    drift_chunks = [i for i in range(1, n_chunks)
                    if chunk_concepts[i] != chunk_concepts[i-1]]
 
    return X_sorted, y_concept, drift_chunks
 
 
# ============================================================
#  ELECTRICITY (injected drift)
# ============================================================
 
print("=" * 60)
print("Electricity (semi-synthetic, injected drift)")
print("=" * 60)
 
if not already_done('electricity'):
    csv_path = os.path.join(USP_OLD, 'Electricity.csv')
    rows = []
    for line in load_csv(csv_path):
        parts    = line.split(',')
        features = [float(x) for x in parts[:-1]]
        label    = 0 if parts[-1].strip() == '-1' else 1
        rows.append(features + [label])
    data  = np.array(rows, dtype=float)
    X_raw, y_raw = data[:, :-1], data[:, -1].astype(int)
 
    X_sorted, y_concept, drift_chunks = build_sorted_drift_stream(X_raw, y_raw)
 
    save_stream('electricity', X_sorted, y_concept, normalise=True)
    save_gt('electricity', drift_chunks)
 
 
# ============================================================
#  FOREST COVER TYPE (injected drift) — all 7 classes
# ============================================================
 
print("\n" + "=" * 60)
print("Covtype (semi-synthetic, injected drift, all 7 classes)")
print("=" * 60)
 
if not already_done('covtype'):
    csv_path = os.path.join(USP_OLD, 'ForestCoverType.csv')
    rows = []
    for line in load_csv(csv_path):
        parts    = line.split(',')
        features = [float(x) for x in parts[:-1]]
        label    = int(parts[-1].strip()) - 1   # remap 1-7 -> 0-6
        rows.append(features + [label])
    data  = np.array(rows, dtype=float)
    X_raw, y_raw = data[:, :-1], data[:, -1].astype(int)
 
    print(f"  Original class counts: "
          f"{dict(collections.Counter(y_raw.tolist()))}")
 
    X_sorted, y_concept, drift_chunks = build_sorted_drift_stream(X_raw, y_raw)
 
    # already normalised in the USP version
    save_stream('covtype', X_sorted, y_concept, normalise=False)
    save_gt('covtype', drift_chunks)
 
 
# ============================================================
#  VERIFICATION
# ============================================================
 
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
 
all_ok = True
for stream_name in ['electricity', 'covtype']:
    sp = os.path.join(OUT_STREAMS, f'{stream_name}.npy')
    gp = os.path.join(OUT_GT,      f'{stream_name}.npy')
    if os.path.exists(sp) and os.path.exists(gp):
        d  = np.load(sp)
        g  = np.load(gp)
        nc = len(g) + 1
        rb = 1.0 / nc
        print(f"  OK  {stream_name}")
        print(f"      shape={d.shape}  chunks@{CHUNK_SIZE}="
              f"{d.shape[0]//CHUNK_SIZE}  concepts={nc}  "
              f"random_baseline={rb:.3f}")
        print(f"      classes={np.unique(d[:,-1]).tolist()}  "
              f"drift_chunks={g.tolist()}")
    else:
        print(f"  MISSING: {stream_name}")
        all_ok = False
 
print()
print("ALL OK" if all_ok else "SOME FILES MISSING — check paths above")