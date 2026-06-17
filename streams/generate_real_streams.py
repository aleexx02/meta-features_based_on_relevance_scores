# streams/generate_real_streams.py
# ============================================================
# INSECTS streams for Experiment 3.
#
# Sourced from the USP DS Repository. Drift chunk
# indices come directly from Table 2 of that paper, which reports the change points in raw
# instance numbers. I convert these to chunk indices by dividing by
# chunk_size=200.
#
# IMPORTANT — what the saved npy's last column is, and what "concept"
# means here (these are two DIFFERENT things, kept deliberately
# separate):
#
#   The last column of data/real/annotated_streams/{stream}.npy is the
#   SPECIES label (mapped through INSECTS_SPECIES_MAP below), i.e. the
#   actual classification target ABFS and Komorniczak operate on when
#   computing per-window meta-features. This is the same role the
#   class label plays in any other stream (e.g. SEA/STAGGER in
#   Experiments 1-2) — it is NOT the concept/drift label.
#
#   The CONCEPT label — which segment of the stream a window belongs
#   to, relative to Table 2's documented change points — is computed
#   downstream, in evaluate_concept_classification_3.py and
#   analysis_3.py, directly from the drift_chunks ground truth file
#   saved here. It is purely positional: concept = number of change
#   points passed so far. Table 2 only tells us WHERE a change point
#   occurs, not WHAT the stream changes into, so this label says
#   nothing about the semantic content of each segment, only its
#   position relative to known boundaries. Each segment is treated as
#   a distinct concept even if two segments happen to be statistically
#   similar; no recurring-concept structure is assumed or detected.
#
#   Why this separation matters: if the concept label were saved as
#   the npy's last column (an earlier version of this script did
#   this), ABFS's accuracy-based relevance mechanism and Komorniczak's
#   pymfe extraction would both be computing meta-features against the
#   very thing being evaluated downstream — i.e. "which segment is
#   this" used as the training signal for finding out "which segment
#   is this." Saving the species label instead keeps ABFS/Komorniczak
#   doing the same kind of task they do everywhere else in this
#   project (classify the real target), with concept-discrimination
#   tested purely through how their derived meta-features behave
#   relative to the independently-known drift boundaries.
#
# Only the rows from Table 2 with a discrete, citable change point are
# used:
#   Abrupt (bal.)               5 change points -> 6 concepts
#   Abrupt (imbal.)             5 change points -> 6 concepts
#   Incremental-gradual (bal.)  1 change point  -> 2 concepts
#   Incremental-gradual (imbal.) 1 change point -> 2 concepts
#
# Output format:
#   data/real/annotated_streams/{stream}.npy
#       shape: (n_instances, n_features + 1)
#       last column = integer SPECIES label (0-indexed, via
#       INSECTS_SPECIES_MAP) — NOT the concept label, see above
#       features min-max normalised to [0, 1]
#
#   data/real/annotated_streams_gt/{stream}.npy
#       shape: (n_drifts,)
#       chunk indices where a known change point occurs
#
#   data/real/annotated_streams_analysis/{stream}_*.npy
#       per-chunk diagnostics (class_distribution, feature_means,
#       feature_stds, drift_intensity, label_entropy) — computed on
#       the SAVED (post-normalisation) array, not the raw features,
#       so they describe exactly what the downstream pipeline sees.
# ============================================================
 
 
import numpy as np
import os
import collections
from scipy.stats import entropy
 
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
 
USP_INSECTS = os.path.expanduser(
    '~/usp_ds_repository/USP DS Repository/INSECTS')
 
OUT_STREAMS  = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')
OUT_GT       = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_gt')
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_analysis')
 
os.makedirs(OUT_STREAMS,  exist_ok=True)
os.makedirs(OUT_GT,       exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
 
CHUNK_SIZE = 200
 
 
# ============================================================
#  HELPERS
# ============================================================
 
def minmax_normalise(X):
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    rng     = col_max - col_min
    rng[rng == 0] = 1.0
    return (X - col_min) / rng
 
 
def load_csv(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return [l.strip() for l in lines if l.strip()]
 
 
def already_done(stream_name):
    sp = os.path.join(OUT_STREAMS, f'{stream_name}.npy')
    gp = os.path.join(OUT_GT,      f'{stream_name}.npy')
    if os.path.exists(sp) and os.path.exists(gp):
        d = np.load(sp)
        print(f"  EXISTS : {stream_name}.npy  shape={d.shape}")
        return True
    return False
 
 
def analysis_already_done(stream_name):
    suffixes = ['class_distribution', 'feature_means', 'feature_stds',
                'drift_intensity', 'label_entropy']
    return all(
        os.path.exists(os.path.join(ANALYSIS_DIR, f'{stream_name}_{s}.npy'))
        for s in suffixes
    )
 
 
def save_stream(stream_name, X, y, normalise=False):
    if normalise:
        X = minmax_normalise(X)
    result = np.hstack([X, y.reshape(-1, 1)])
    np.save(os.path.join(OUT_STREAMS, f'{stream_name}.npy'), result)
    counts = dict(collections.Counter(y.tolist()))
    print(f"  SAVED  : {stream_name}.npy")
    print(f"           shape={result.shape}  "
          f"chunks@{CHUNK_SIZE}={result.shape[0]//CHUNK_SIZE}  "
          f"species_classes={counts}")
    return result
 
 
def save_gt(stream_name, drift_chunks):
    np.save(os.path.join(OUT_GT, f'{stream_name}.npy'),
           np.array(drift_chunks))
    print(f"  GT     : drift_chunks={drift_chunks}  "
          f"n_concepts={len(drift_chunks)+1}  "
          f"random_baseline={1/(len(drift_chunks)+1):.3f}")
 
 
def run_stream_analysis(stream_name, X_stream, y_stream):
    """
    Per-chunk diagnostics: class distribution, feature means/stds,
    drift intensity (mean-shift magnitude between consecutive chunks),
    label entropy. X_stream/y_stream MUST be the already-saved
    (post-normalisation) arrays — callers should reload from disk
    rather than reuse pre-normalisation in-memory arrays, so these
    diagnostics describe exactly what the rest of the pipeline sees.
    """
    print(f"\n  [Stream analysis] {stream_name}")
 
    n_instances = X_stream.shape[0]
    n_chunks    = n_instances // CHUNK_SIZE
    n_classes   = int(np.max(y_stream)) + 1
 
    class_distribution = []
    feature_means      = []
    feature_stds       = []
    drift_intensity    = []
    label_entropy      = []
 
    prev_mean = None
 
    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end   = start + CHUNK_SIZE
 
        X_chunk = X_stream[start:end]
        y_chunk = y_stream[start:end]
 
        counts = np.bincount(y_chunk, minlength=n_classes)
        probs  = counts / np.sum(counts)
        class_distribution.append(probs)
 
        mean = np.mean(X_chunk, axis=0)
        std  = np.std(X_chunk, axis=0)
        feature_means.append(mean)
        feature_stds.append(std)
 
        if prev_mean is None:
            drift_intensity.append(0.0)
        else:
            drift_intensity.append(np.linalg.norm(mean - prev_mean))
        prev_mean = mean
 
        label_entropy.append(entropy(probs + 1e-10))
 
    class_distribution = np.array(class_distribution)
    feature_means      = np.array(feature_means)
    feature_stds       = np.array(feature_stds)
    drift_intensity    = np.array(drift_intensity)
    label_entropy       = np.array(label_entropy)
 
    np.save(os.path.join(ANALYSIS_DIR, f'{stream_name}_class_distribution.npy'),
            class_distribution)
    np.save(os.path.join(ANALYSIS_DIR, f'{stream_name}_feature_means.npy'),
            feature_means)
    np.save(os.path.join(ANALYSIS_DIR, f'{stream_name}_feature_stds.npy'),
            feature_stds)
    np.save(os.path.join(ANALYSIS_DIR, f'{stream_name}_drift_intensity.npy'),
            drift_intensity)
    np.save(os.path.join(ANALYSIS_DIR, f'{stream_name}_label_entropy.npy'),
            label_entropy)
 
    print(f"    avg drift: {np.mean(drift_intensity):.4f}  "
          f"max drift: {np.max(drift_intensity):.4f}")
    print(f"    avg entropy: {np.mean(label_entropy):.4f}")
    print(f"  [Stream analysis] Saved to {ANALYSIS_DIR}")
 
 
# ============================================================
#  INSECTS — genuinely annotated (Table 2, Souza et al. 2020)
# ============================================================
 
print("=" * 60)
print("INSECTS streams (Table 2, Souza et al. 2020)")
print("=" * 60)
 
# (csv filename, change points in raw instance numbers)
INSECTS_STREAMS = {
    'INSECTS-abrupt_balanced': (
        'INSECTS abrupt_balanced.csv',
        [14352, 19500, 33240, 38682, 39510]),
    'INSECTS-abrupt_imbalanced': (
        'INSECTS abrupt_imbalanced.csv',
        [83859, 128651, 182320, 242883, 268380]),
    'INSECTS-incgradual_balanced': (
        'INSECTS gradual_balanced.csv',
        [14028]),
    'INSECTS-incgradual_imbalanced': (
        'INSECTS gradual_imbalanced.csv',
        [58159]),
}
 
# INSECTS species ID -> 0-indexed encoding. THIS is what gets saved as
# the npy's last column (the classification target) — see header note.
INSECTS_SPECIES_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 11: 4, 12: 5}
 
for stream_name, (csv_fname, change_points) in INSECTS_STREAMS.items():
    print(f"\n  {stream_name}")
 
    stream_already_done = already_done(stream_name)
 
    if not stream_already_done:
        csv_path = os.path.join(USP_INSECTS, csv_fname)
        if not os.path.exists(csv_path):
            print(f"  MISSING: {csv_path}"); continue
 
        rows       = []
        labels_raw = []
        for idx, line in enumerate(load_csv(csv_path)):
            parts        = line.split(',')
            features     = [float(x) for x in parts[:-1]]
            species_code = int(float(parts[-1]))
            rows.append(features)
            labels_raw.append(species_code)
 
        X = np.array(rows, dtype=float)
 
        try:
            y_species = np.array(
                [INSECTS_SPECIES_MAP[c] for c in labels_raw])
        except KeyError as e:
            raise ValueError(
                f"{stream_name}: species code {e.args[0]} not found in "
                f"INSECTS_SPECIES_MAP — update the map or check that "
                f"'{csv_fname}' is the expected file/format.")
 
        drift_chunks = [p // CHUNK_SIZE for p in change_points]
 
        save_stream(stream_name, X, y_species, normalise=True)
        save_gt(stream_name, drift_chunks)
 
    # ---- stream analysis ----
    # Runs independently of whether the stream npy was just generated
    # above or already existed, so deleting only the analysis folder
    # (e.g. to recompute diagnostics) doesn't require regenerating the
    # stream itself.
    if analysis_already_done(stream_name):
        print(f"  [Stream analysis] {stream_name}: already exists — skipping.")
        continue
 
    saved_path = os.path.join(OUT_STREAMS, f'{stream_name}.npy')
    if not os.path.exists(saved_path):
        print(f"  [Stream analysis] {stream_name}: stream npy missing — skipping.")
        continue
 
    # Reload from disk rather than reusing the in-memory pre-save
    # array: save_stream() normalises X internally without mutating
    # the caller's variable, so re-reading guarantees the diagnostics
    # below describe exactly what's on disk (post-normalisation),
    # matching what the rest of the pipeline actually consumes.
    saved    = np.load(saved_path)
    X_stream = saved[:, :-1]
    y_stream = saved[:, -1].astype(int)
 
    run_stream_analysis(stream_name, X_stream, y_stream)
 
 
# ============================================================
#  VERIFICATION
# ============================================================
 
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
 
all_ok = True
for stream_name in INSECTS_STREAMS:
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
        print(f"      drift_chunks={g.tolist()}")
    else:
        print(f"  MISSING: {stream_name}")
        all_ok = False
 
print()
print("ALL OK" if all_ok else "SOME FILES MISSING — check paths above")



