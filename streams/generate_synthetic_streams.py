# streams/generate_synthetic_streams.py

# Synthetic stream generator script: produces synthetic streams used
# across several different experiments -- recurring-concept streams,
# StreamGenerator (strlearn) streams for Experiment 2, and river-based
# SEA, STAGGER, and LED streams with drift.

# Saves to disk (data/synthetic/streams/, streams_gt/):
#   - 4 recurring-concept variants
#   - 6 river-based variants: SEA, STAGGER, LED, each sudden + gradual

import numpy as np
import os
from sklearn.datasets import make_classification
from strlearn.streams import StreamGenerator
from river.datasets import synth as river_synth

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

STREAM_DIR = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'streams')
GT_DIR     = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'streams_gt')
os.makedirs(STREAM_DIR, exist_ok=True)
os.makedirs(GT_DIR,     exist_ok=True)

# ============================================================
#  SHARED CONFIG
# ============================================================
CHUNK_SIZE  = 200
MASTER_SEED = 1233
np.random.seed(MASTER_SEED)


# ============================================================
#  RECURRING CONCEPTS — independent make_classification pools
# ============================================================
N_FEATURES        = 10
N_INFORMATIVE     = 10
SEGMENT_CHUNKS    = 50
N_SEGMENTS        = 100
TRANSITION_CHUNKS = 10
N_CONCEPTS        = 4
POOL_SIZE         = 200_000

CONCEPT_SEEDS = np.random.randint(100, 10000, N_CONCEPTS)


def make_concept_generators():
    """K independent, reusable concept sources. Recurring 'concept 2'
    later in the stream means re-sampling more rows from this SAME
    pool, never regenerating it."""
    generators = []
    for k in range(N_CONCEPTS):
        X, y = make_classification(
            n_samples=POOL_SIZE, n_features=N_FEATURES,
            n_informative=N_INFORMATIVE, n_redundant=0, n_repeated=0,
            n_classes=2, class_sep=1.0, random_state=int(CONCEPT_SEEDS[k]),
        )
        generators.append({'X': X, 'y': y, 'cursor': 0})
    return generators


def draw(generators, k, n):
    """Draw n fresh instances from concept k's pool, advancing its
    cursor with wraparound."""
    g = generators[k]
    N = len(g['y'])
    idx = (np.arange(g['cursor'], g['cursor'] + n)) % N
    g['cursor'] = (g['cursor'] + n) % N
    return g['X'][idx], g['y'][idx]


def build_schedule(recurrence, rng):
    n_cycles   = N_SEGMENTS // N_CONCEPTS
    base_order = list(range(N_CONCEPTS))
    schedule   = []
    for _ in range(n_cycles):
        if recurrence == 'fixed':
            schedule.extend(base_order)
        elif recurrence == 'random':
            order = base_order.copy()
            rng.shuffle(order)
            schedule.extend(order)
        else:
            raise ValueError(recurrence)
    return schedule


def build_stream(recurrence, transition, seed):
    rng        = np.random.RandomState(seed)
    generators = make_concept_generators()
    schedule   = build_schedule(recurrence, rng)

    X_all, y_all, concept_per_chunk = [], [], []

    for seg_id, concept_id in enumerate(schedule):
        next_concept_id = (schedule[seg_id + 1]
                           if seg_id + 1 < len(schedule) else None)

        for c in range(SEGMENT_CHUNKS):
            in_transition = (
                transition == 'gradual'
                and next_concept_id is not None
                and c >= SEGMENT_CHUNKS - TRANSITION_CHUNKS
            )

            if not in_transition:
                Xc, yc = draw(generators, concept_id, CHUNK_SIZE)
                concept_per_chunk.append(concept_id)
            else:
                steps    = c - (SEGMENT_CHUNKS - TRANSITION_CHUNKS)
                progress = (steps + 0.5) / TRANSITION_CHUNKS
                p_new    = 1.0 / (1.0 + np.exp(-12 * (progress - 0.5)))

                mask_new = rng.random(CHUNK_SIZE) < p_new
                n_old, n_new = int((~mask_new).sum()), int(mask_new.sum())
                Xa, ya = draw(generators, concept_id,      n_old)
                Xb, yb = draw(generators, next_concept_id, n_new)

                Xc = np.empty((CHUNK_SIZE, N_FEATURES))
                yc = np.empty(CHUNK_SIZE, dtype=int)
                Xc[~mask_new], yc[~mask_new] = Xa, ya
                Xc[mask_new],  yc[mask_new]  = Xb, yb
                concept_per_chunk.append(
                    concept_id if mask_new.mean() < 0.5 else next_concept_id)

            X_all.append(Xc)
            y_all.append(yc)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    data  = np.column_stack([X_all, y_all])
    return data, np.array(concept_per_chunk)


RECURRING_VARIANTS = [
    ('fixed',  'sudden'),
    ('fixed',  'gradual'),
    ('random', 'sudden'),
    ('random', 'gradual'),
]


# ============================================================
#  SEA / STAGGER / LED — river-based generators with drift
#  Recipe: 4 concepts, 3 drifts, abrupt at instances
#  10000/20000/30000, gradual centered at 9500/20000/30500 with
#  width 1000 -- translated to chunk_size=200 units.
# ============================================================
DRIFT_CHUNKS_ABRUPT     = [50, 100, 150]
DRIFT_CHUNKS_GRADUAL    = [48, 100, 153]
RIVER_TRANSITION_CHUNKS = 5
RIVER_TOTAL_CHUNKS      = 200

SEA_VARIANT_ORDER     = [0, 1, 2, 3]
STAGGER_VARIANT_ORDER = [0, 1, 2, 0]
LED_VARIANT_ORDER     = [0, 1, 2, 3]


class CategoricalEncoder:
    """Stable string->int encoding for categorical river features
    (e.g. STAGGER's size/shape/colour). One instance must be shared
    across an ENTIRE stream build -- a fresh per-chunk mapping would
    silently scramble which integer represents which category."""
    def __init__(self):
        self.maps = {}

    def encode(self, key, value):
        if isinstance(value, (int, float)):
            return float(value)
        if key not in self.maps:
            self.maps[key] = {}
        if value not in self.maps[key]:
            self.maps[key][value] = len(self.maps[key])
        return float(self.maps[key][value])


def river_stream_to_arrays(dataset, n, encoder):
    X, y = [], []
    for x, label in dataset.take(n):
        X.append([encoder.encode(k, v) for k, v in x.items()])
        y.append(int(label))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def build_river_drift_stream(make_gen, variant_order, drift_chunks, transition, seed):
    rng     = np.random.RandomState(seed)
    encoder = CategoricalEncoder()
    boundaries = [0] + list(drift_chunks) + [RIVER_TOTAL_CHUNKS]

    X_all, y_all, concept_per_chunk = [], [], []

    for seg_id, variant in enumerate(variant_order):
        seg_start, seg_end = boundaries[seg_id], boundaries[seg_id + 1]
        next_variant = (variant_order[seg_id + 1]
                        if seg_id + 1 < len(variant_order) else None)
        gen_current = make_gen(variant)
        gen_next    = make_gen(next_variant) if next_variant is not None else None

        for c in range(seg_start, seg_end):
            in_transition = (
                transition == 'gradual'
                and gen_next is not None
                and c >= seg_end - RIVER_TRANSITION_CHUNKS
            )
            if not in_transition:
                Xc, yc = river_stream_to_arrays(gen_current, CHUNK_SIZE, encoder)
                concept_per_chunk.append(variant)
            else:
                steps    = c - (seg_end - RIVER_TRANSITION_CHUNKS)
                progress = (steps + 0.5) / RIVER_TRANSITION_CHUNKS
                p_new    = 1.0 / (1.0 + np.exp(-12 * (progress - 0.5)))

                mask_new = rng.random(CHUNK_SIZE) < p_new
                n_old, n_new = int((~mask_new).sum()), int(mask_new.sum())
                Xa, ya = river_stream_to_arrays(gen_current, n_old, encoder)
                Xb, yb = river_stream_to_arrays(gen_next,    n_new, encoder)

                Xc = np.empty((CHUNK_SIZE, Xa.shape[1]))
                yc = np.empty(CHUNK_SIZE, dtype=int)
                Xc[~mask_new], yc[~mask_new] = Xa, ya
                Xc[mask_new],  yc[mask_new]  = Xb, yb
                concept_per_chunk.append(
                    variant if mask_new.mean() < 0.5 else next_variant)

            X_all.append(Xc)
            y_all.append(yc)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    return np.column_stack([X_all, y_all]), np.array(concept_per_chunk)


RIVER_VARIANTS = [
    ('sea',     'sudden',  lambda v: river_synth.SEA(seed=MASTER_SEED, variant=v),
                SEA_VARIANT_ORDER, DRIFT_CHUNKS_ABRUPT),
    ('sea',     'gradual', lambda v: river_synth.SEA(seed=MASTER_SEED, variant=v),
                SEA_VARIANT_ORDER, DRIFT_CHUNKS_GRADUAL),
    ('stagger', 'sudden',  lambda v: river_synth.STAGGER(seed=MASTER_SEED, variant=v),
                STAGGER_VARIANT_ORDER, DRIFT_CHUNKS_ABRUPT),
    ('stagger', 'gradual', lambda v: river_synth.STAGGER(seed=MASTER_SEED, variant=v),
                STAGGER_VARIANT_ORDER, DRIFT_CHUNKS_GRADUAL),
    ('led',     'sudden',  lambda v: river_synth.LED(seed=MASTER_SEED + v,
                                noise_percentage=0.1, irrelevant_features=True),
                LED_VARIANT_ORDER, DRIFT_CHUNKS_ABRUPT),
    ('led',     'gradual', lambda v: river_synth.LED(seed=MASTER_SEED + v,
                                noise_percentage=0.1, irrelevant_features=True),
                LED_VARIANT_ORDER, DRIFT_CHUNKS_GRADUAL),
]


# ============================================================
#  Experiment 2 — StreamGenerator (strlearn) with VARYING parameters
#  NOT pre-saved to disk: the full 4x4 grid x 2 drift types x 5
#  replications runs into tens of GB at chunk_size up to 1000, and
#  regenerating from a fixed seed is cheap -- the expensive part is
#  the ABFS/Komorniczak feature EXTRACTION downstream, not generation.
#  Imported by evaluate_concept_classification_2.py instead of
#  duplicated there.
# ============================================================
EXP2_N_CHUNKS       = 5000
EXP2_N_FEATURES     = 20
EXP2_CHUNK_SIZES    = [100, 200, 500, 1000]
EXP2_N_INFORMATIVES = [3, 5, 10, 15]
# n_concepts (4th element) isn't derivable by a simple n_drifts+1
# formula for gradual drift, due to the mod-4 cycling logic in
# assign_labels_gradual -- stored directly rather than re-derived
# differently in each consuming script.
EXP2_DRIFT_CONFIGS  = [
    ('sudden',  20, 9999, 21),
    ('gradual',  6,    5, 25),
]


def assign_labels_gradual(stream, n_chunks, chunk_size):
    e = stream._sigmoid(stream.concept_sigmoid_spacing, stream.n_drifts)[1][::chunk_size]
    concept, decreasing, labels = 0, True, []
    for chunk in range(n_chunks):
        if decreasing:
            if concept % 4 == 0 and e[chunk] < 0.9:  concept += 1
            if concept % 4 == 1 and e[chunk] < 0.75: concept += 1
            if concept % 4 == 2 and e[chunk] < 0.25: concept += 1
            if concept % 4 == 3 and e[chunk] < 0.1:
                concept += 1; decreasing = False
        else:
            if concept % 4 == 0 and e[chunk] > 0.1:  concept += 1
            if concept % 4 == 1 and e[chunk] > 0.25: concept += 1
            if concept % 4 == 2 and e[chunk] > 0.75: concept += 1
            if concept % 4 == 3 and e[chunk] > 0.9:
                concept += 1; decreasing = True
        labels.append(concept)
    return np.array(labels)


def get_exp2_concept_labels(stream, drift_type, n_chunks, chunk_size):
    if drift_type == 'sudden':
        cs = stream.concept_selector.copy()
        return np.array([
            int(np.bincount(cs[i*chunk_size:(i+1)*chunk_size]).argmax())
            for i in range(n_chunks)
        ])
    return assign_labels_gradual(stream, n_chunks, chunk_size)


def build_exp2_stream(random_state, drift_type, n_drifts,
                       concept_sigmoid_spacing, chunk_size, n_informative):
    config = dict(
        n_drifts=n_drifts, n_chunks=EXP2_N_CHUNKS, chunk_size=chunk_size,
        n_features=EXP2_N_FEATURES, n_informative=n_informative,
        n_redundant=0, n_repeated=0,
        concept_sigmoid_spacing=concept_sigmoid_spacing,
        random_state=random_state,
    )
    stream = StreamGenerator(**config)
    concept_labels = get_exp2_concept_labels(
        stream, drift_type, EXP2_N_CHUNKS, chunk_size)

    stream.reset()
    X_all, y_all = [], []
    for X_chunk, y_chunk in stream:
        X_all.append(X_chunk)
        y_all.append(y_chunk)

    return np.vstack(X_all), np.concatenate(y_all), concept_labels


# ============================================================
#  EXECUTION — only runs when this file is executed directly,
#  never on import (evaluate_concept_classification_2.py imports
#  assign_labels_gradual / get_exp2_concept_labels above and must
#  NOT trigger generation as a side effect of that import)
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("Recurring-concept streams")
    print("=" * 60)

    for recurrence, transition in RECURRING_VARIANTS:
        stream_name = f'recurring_{recurrence}_{transition}'
        stream_path = os.path.join(STREAM_DIR, f'{stream_name}.npy')
        gt_path     = os.path.join(GT_DIR,     f'{stream_name}.npy')

        print(f"\n{stream_name}")
        if os.path.exists(stream_path) and os.path.exists(gt_path):
            print(f"  EXISTS — skipping.")
            continue

        data, concept_per_chunk = build_stream(recurrence, transition, MASTER_SEED)
        np.save(stream_path, data)
        np.save(gt_path, concept_per_chunk)
        print(f"  SAVED  : {stream_name}.npy  shape={data.shape}")

    print(f"\n{'='*60}")
    print("River-based streams: SEA, STAGGER, LED")
    print(f"{'='*60}")

    for name, transition, make_gen, variant_order, drift_chunks in RIVER_VARIANTS:
        stream_name = f'{name}_{transition}'
        stream_path = os.path.join(STREAM_DIR, f'{stream_name}.npy')
        gt_path     = os.path.join(GT_DIR,     f'{stream_name}.npy')

        print(f"\n{stream_name}")
        if os.path.exists(stream_path) and os.path.exists(gt_path):
            print(f"  EXISTS — skipping.")
            continue

        data, concept_per_chunk = build_river_drift_stream(
            make_gen, variant_order, drift_chunks, transition, MASTER_SEED)
        np.save(stream_path, data)
        np.save(gt_path, concept_per_chunk)
        print(f"  SAVED  : {stream_name}.npy  shape={data.shape}")

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    ALL_SAVED_STREAMS = (
        [f'recurring_{r}_{t}' for r, t in RECURRING_VARIANTS] +
        [f'{n}_{t}' for n, t, *_ in RIVER_VARIANTS]
    )

    for stream_name in ALL_SAVED_STREAMS:
        sp = os.path.join(STREAM_DIR, f'{stream_name}.npy')
        gp = os.path.join(GT_DIR,     f'{stream_name}.npy')
        if not (os.path.exists(sp) and os.path.exists(gp)):
            print(f"  MISSING: {stream_name}")
            continue
        data = np.load(sp)
        cpc  = np.load(gp)
        counts = dict(zip(*np.unique(cpc, return_counts=True)))
        n_boundaries = int(np.sum(np.diff(cpc) != 0))
        print(f"  {stream_name}")
        print(f"      shape={data.shape}  n_chunks={len(cpc)}  "
              f"per-concept chunk counts={counts}  n_boundaries={n_boundaries}")

    print("\nALL DONE.")