# streams/generate_synthetic_streams.py

# Builder library for the synthetic-stream experiments. As of the grid
# redesign, this file does NOT pre-save Experiment 3/4 streams to disk
# (the way an earlier version did) -- the grid streams are regenerated
# on the fly per grid cell by the evaluate scripts, exactly like
# Experiment 2's StreamGenerator streams already are. There is nothing
# to "run" here to produce data files; the __main__ block only
# verifies that the builders work and prints the grid layout.
#
#   Experiment 2 — strlearn StreamGenerator, grid chunk_size x
#     n_informative. Helpers (build_exp2_stream / get_exp2_concept_labels)
#     imported by evaluate_concept_classification_2.py. UNCHANGED.
#
#   Experiment 3 — river SEA / STAGGER / LED, SEQUENTIAL concepts,
#     500,000 instances per stream, swept over chunk_size
#     {100,200,500}. No drift-count axis: each generator runs its
#     fixed concept order once (SEA [0,1,2,3], STAGGER [0,1,2,0], LED
#     [0,1,2,3]). The chunk_size sweep is the one grid axis -- the
#     river analogue of "how does window size affect discrimination",
#     parallel to one axis of Experiment 2.
#
#   Experiment 4 — river SEA / STAGGER, RECURRING concepts, full grid
#     chunk_size {100,200,500} x n_drifts {1,3,7,15}, 500,000
#     instances per stream. Concepts CYCLE through the generator's
#     concept set (segment i -> concept i % n_concepts), so as n_drifts
#     grows the concepts recur more and more. This is the only way to
#     get a drift-count axis out of river: SEA has exactly 4 concepts
#     and STAGGER exactly 3 (fixed labelling rules, not seed-tunable),
#     so beyond a few drifts the concepts MUST repeat -- recurrence is
#     forced by the generators, not chosen. At low n_drifts the streams
#     barely recur (n_drifts=3 on SEA = [0,1,2,3], each concept once);
#     at high n_drifts they recur heavily (n_drifts=15 = 16 segments,
#     4 cycles of SEA's 4 concepts).
#
# ------------------------------------------------------------------
#  Two river facts, both verified empirically against the installed
#  library before being relied on here:
#
#  1. STAGGER takes `classification_function`, NOT `variant` (only SEA
#     takes `variant`). The factories below use the right keyword.
#
#  2. SEED PROPAGATION. The seed passed to the build_* functions flows
#     into the river generators (via the factory closures) and into
#     every random choice (gradual transition masks). The evaluate
#     scripts regenerate each grid cell once per replication seed and
#     stack into (n_reps, n_windows, n_clfs); if the seed did not reach
#     the generators, every replication of a sudden cell would be
#     byte-identical and the error bars would be a meaningless zero.
#
# Target class (npy last column) vs concept label: the last column is
# the generator's REAL target (binary for SEA/STAGGER, 10-class digit
# for LED) -- what ABFS/Komorniczak classify per window. The concept
# label is concept_per_chunk[i], the GENERATIVE concept id, which
# repeats on recurrence. Drift boundaries = np.diff(concept_per_chunk)
# != 0.

import numpy as np
import os
from strlearn.streams import StreamGenerator
from river.datasets import synth as river_synth

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# ============================================================
#  SHARED CONFIG
# ============================================================
SEED      = 1233
TOTAL_INSTANCES  = 500_000          # per Experiment 3/4 stream (per grid cell)
RIVER_TRANSITION_FRAC = 0.10        # gradual transition width as a fraction of
                                    # each segment (scales with segment length,
                                    # so it stays "10% of the segment" at every
                                    # chunk_size / n_drifts combination instead
                                    # of a fixed chunk count that could overrun
                                    # short segments)
np.random.seed(SEED)


# ============================================================
#  RIVER GENERATOR FACTORIES (seed-aware)
# ============================================================

def make_sea(seed, concept_idx):
    # SEA: 3 numeric features (first 2 relevant), binary target,
    # 4 thresholds via `variant` (0-3).
    return river_synth.SEA(seed=seed, variant=concept_idx)

def make_stagger(seed, concept_idx):
    # STAGGER: 3 categorical features (size/color/shape), binary target,
    # 3 boolean rules via `classification_function` (0-2).
    return river_synth.STAGGER(seed=seed, classification_function=concept_idx)

def make_led(seed, concept_idx):
    # LED: 24 features (7 real segments + 17 irrelevant), 10-class digit.
    # Concepts differentiated by per-concept seed offset.
    return river_synth.LED(seed=seed + concept_idx,
                           noise_percentage=0.1, irrelevant_features=True)

GENERATOR_FACTORY = {'sea': make_sea, 'stagger': make_stagger, 'led': make_led}
GENERATOR_N_CONCEPTS = {'sea': 4, 'stagger': 3, 'led': 4}
RIVER_N_FEATURES     = {'sea': 3, 'stagger': 3, 'led': 24}
RIVER_N_TARGET_CLASSES = {'sea': 2, 'stagger': 2, 'led': 10}


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


# ============================================================
#  CORE GRID BUILDER  (shared by Experiments 3 and 4)
# ============================================================

def build_river_grid_stream(gen_name, concept_order, transition, chunk_size,
                            seed, total_instances=TOTAL_INSTANCES):
    """
    Build one river stream of ~total_instances instances, windowed at
    chunk_size, walking through concept_order (repeats in this list =
    recurrence). Drift boundaries split the chunks evenly across the
    segments. Gradual transitions sigmoid-blend the last
    RIVER_TRANSITION_FRAC of each segment into the next.

    Returns: data (n_chunks*chunk_size, n_features+1), concept_per_chunk.
    """
    factory  = GENERATOR_FACTORY[gen_name]
    make_gen = lambda c: factory(seed, c)
    rng      = np.random.RandomState(seed)
    encoder  = CategoricalEncoder()

    n_chunks   = total_instances // chunk_size
    n_segments = len(concept_order)
    boundaries = [round(i * n_chunks / n_segments) for i in range(n_segments + 1)]

    X_all, y_all, concept_per_chunk = [], [], []

    for seg_id, concept in enumerate(concept_order):
        seg_start, seg_end = boundaries[seg_id], boundaries[seg_id + 1]
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue
        next_concept = (concept_order[seg_id + 1]
                        if seg_id + 1 < n_segments else None)
        gen_current = make_gen(concept)
        gen_next    = make_gen(next_concept) if next_concept is not None else None

        trans_chunks = (max(1, int(RIVER_TRANSITION_FRAC * seg_len))
                        if (transition == 'gradual' and gen_next is not None) else 0)

        for c in range(seg_start, seg_end):
            in_transition = trans_chunks > 0 and c >= seg_end - trans_chunks
            if not in_transition:
                Xc, yc = river_stream_to_arrays(gen_current, chunk_size, encoder)
                concept_per_chunk.append(concept)
            else:
                steps    = c - (seg_end - trans_chunks)
                progress = (steps + 0.5) / trans_chunks
                p_new    = 1.0 / (1.0 + np.exp(-12 * (progress - 0.5)))
                mask_new = rng.random(chunk_size) < p_new
                n_old, n_new = int((~mask_new).sum()), int(mask_new.sum())
                Xa, ya = river_stream_to_arrays(gen_current, n_old, encoder)
                Xb, yb = river_stream_to_arrays(gen_next,    n_new, encoder)
                Xc = np.empty((chunk_size, Xa.shape[1] if n_old else Xb.shape[1]))
                yc = np.empty(chunk_size, dtype=int)
                # Assign OLD samples safely
                if n_old > 0:
                    if Xa.ndim == 1:
                        Xa = Xa.reshape(-1, Xc.shape[1])
                    if ya.ndim == 0:
                        ya = np.array([ya])
                    elif ya.ndim > 1:
                        ya = ya.ravel()

                    if Xa.shape[0] != n_old or ya.shape[0] != n_old:
                        raise ValueError(
                            f"[build_river_grid_stream OLD] mismatch:\n"
                            f"mask_old selects {n_old} rows\n"
                            f"Xa shape: {Xa.shape}, ya shape: {ya.shape}"
                        )

                    Xc[~mask_new] = Xa
                    yc[~mask_new] = ya


                # Assign NEW samples safely
                if n_new > 0:
                    if Xb.ndim == 1:
                        Xb = Xb.reshape(-1, Xc.shape[1])
                    if yb.ndim == 0:
                        yb = np.array([yb])
                    elif yb.ndim > 1:
                        yb = yb.ravel()

                    if Xb.shape[0] != n_new or yb.shape[0] != n_new:
                        raise ValueError(
                            f"[build_river_grid_stream NEW] mismatch:\n"
                            f"mask_new selects {n_new} rows\n"
                            f"Xb shape: {Xb.shape}, yb shape: {yb.shape}"
                        )

                    Xc[mask_new] = Xb
                    yc[mask_new] = yb
                concept_per_chunk.append(
                    concept if mask_new.mean() < 0.5 else next_concept)

            X_all.append(Xc)
            y_all.append(yc)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    return np.column_stack([X_all, y_all]), np.array(concept_per_chunk)


# ============================================================
#  EXPERIMENT 3 — SEA / STAGGER / LED, SEQUENTIAL, chunk_size sweep
# ============================================================
EXP3_GENERATORS  = ['sea', 'stagger', 'led']
EXP3_TRANSITIONS = ['sudden', 'gradual']
EXP3_ORDERS = {
    'sea':     [0, 1, 2, 3],   # 4 concepts, 3 drifts
    'stagger': [0, 1, 2, 0],   # 3 unique concepts (0 recurs once), 3 drifts
    'led':     [0, 1, 2, 3],   # 4 concepts, 3 drifts
}
CHUNK_SIZES_EXP3 = [100, 200, 500]


def build_exp3_stream(gen_name, transition, chunk_size, seed):
    """seed->(data, concept_per_chunk) for one Experiment 3 cell.
    Sequential concept order, fixed per generator; the only swept axis
    is chunk_size."""
    return build_river_grid_stream(
        gen_name, EXP3_ORDERS[gen_name], transition, chunk_size, seed)


def exp3_specs():
    """One dict per Experiment 3 grid cell (generator x transition x
    chunk_size). Single source of truth for evaluate_concept_
    classification_3.py and analysis_3.py."""
    specs = []
    for gen_name in EXP3_GENERATORS:
        order = EXP3_ORDERS[gen_name]
        for transition in EXP3_TRANSITIONS:
            for cs in CHUNK_SIZES_EXP3:
                name = f'{gen_name}_chunk{cs}_{transition}'
                specs.append({
                    'name':       name,
                    'gen_name':   gen_name,
                    'transition': transition,
                    'chunk_size': cs,
                    'order':      order,
                    'n_features': RIVER_N_FEATURES[gen_name],
                    'n_concepts': len(set(order)),
                    'builder': (lambda seed, g=gen_name, t=transition, c=cs:
                                build_exp3_stream(g, t, c, seed)),
                })
    return specs


# ============================================================
#  EXPERIMENT 4 — SEA / STAGGER, RECURRING, full chunk_size x n_drifts grid
# ============================================================
EXP4_GENERATORS  = ['sea', 'stagger']
EXP4_TRANSITIONS = ['sudden', 'gradual']
EXP4_N_DRIFTS    = [1, 3, 7, 15]   # concept switches over the stream;
                                   # n_segments = n_drifts + 1
CHUNK_SIZES_EXP4 = [100, 200, 500]


def cycling_order(gen_name, n_drifts):
    """Concept order that cycles through the generator's concept set:
    segment i -> concept (i % n_concepts). With n_drifts switches there
    are n_drifts+1 segments. Recurrence appears once n_drifts+1 exceeds
    the generator's concept count."""
    n = GENERATOR_N_CONCEPTS[gen_name]
    return [i % n for i in range(n_drifts + 1)]


def build_exp4_stream(gen_name, n_drifts, transition, chunk_size, seed):
    """seed->(data, concept_per_chunk) for one Experiment 4 grid cell."""
    order = cycling_order(gen_name, n_drifts)
    return build_river_grid_stream(gen_name, order, transition, chunk_size, seed)


def exp4_specs():
    """One dict per Experiment 4 grid cell (generator x transition x
    chunk_size x n_drifts). Single source of truth for evaluate_concept_
    classification_4.py and analysis_4.py.

    n_concepts is the number of DISTINCT concepts that actually appear
    in the cell: min(n_drifts+1, generator concept count). At
    n_drifts=1 only 2 concepts appear (baseline 1/2); once n_drifts+1
    reaches the generator's concept count, all of them appear (SEA 4,
    STAGGER 3) and further drifts add recurrence, not new concepts."""
    specs = []
    for gen_name in EXP4_GENERATORS:
        n_gen = GENERATOR_N_CONCEPTS[gen_name]
        for transition in EXP4_TRANSITIONS:
            for cs in CHUNK_SIZES_EXP4:
                for nd in EXP4_N_DRIFTS:
                    name = f'{gen_name}_chunk{cs}_ndrift{nd}_{transition}'
                    order = cycling_order(gen_name, nd)
                    specs.append({
                        'name':       name,
                        'gen_name':   gen_name,
                        'transition': transition,
                        'chunk_size': cs,
                        'n_drifts':   nd,
                        'order':      order,
                        'n_features': RIVER_N_FEATURES[gen_name],
                        'n_concepts': len(set(order)),
                        'builder': (lambda seed, g=gen_name, n=nd, t=transition, c=cs:
                                    build_exp4_stream(g, n, t, c, seed)),
                    })
    return specs


# ============================================================
#  Experiment 2 — StreamGenerator (strlearn), VARYING parameters
#  Unchanged. Imported by evaluate_concept_classification_2.py; no disk
#  side effects on import.
# ============================================================
EXP2_N_CHUNKS       = 5000
EXP2_N_FEATURES     = 20
EXP2_CHUNK_SIZES    = [100, 200, 500, 1000]
EXP2_N_INFORMATIVES = [3, 5, 10, 15]
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
#  __main__ — VERIFICATION ONLY (no disk saving; grid streams are
#  regenerated per cell by the evaluate scripts).
# ============================================================
if __name__ == "__main__":
    print("Grid streams are regenerated at evaluation time (like Experiment 2);")
    print("this script saves nothing. Verifying builders + printing grid layout.\n")

    print("=" * 64)
    print(f"Experiment 3 grid: {len(exp3_specs())} cells "
          f"(generators {EXP3_GENERATORS} x transitions {EXP3_TRANSITIONS} "
          f"x chunk_sizes {CHUNK_SIZES_EXP3})")
    print(f"  {TOTAL_INSTANCES} instances per stream, sequential concepts.")
    print("=" * 64)
    # build one small smoke-test cell per generator at a tiny instance count
    for gen_name in EXP3_GENERATORS:
        data, cpc = build_river_grid_stream(
            gen_name, EXP3_ORDERS[gen_name], 'sudden', 100, SEED,
            total_instances=2000)
        segs = [int(cpc[i]) for i in range(len(cpc)) if i == 0 or cpc[i] != cpc[i-1]]
        print(f"  {gen_name:8s} n_feat={RIVER_N_FEATURES[gen_name]:2d} "
              f"order={EXP3_ORDERS[gen_name]} -> segment seq {segs} "
              f"({len(set(cpc.tolist()))} unique concepts)")

    print("\n" + "=" * 64)
    print(f"Experiment 4 grid: {len(exp4_specs())} cells "
          f"(generators {EXP4_GENERATORS} x transitions {EXP4_TRANSITIONS} "
          f"x chunk_sizes {CHUNK_SIZES_EXP4} x n_drifts {EXP4_N_DRIFTS})")
    print(f"  {TOTAL_INSTANCES} instances per stream, recurring (cycling) concepts.")
    print("=" * 64)
    for gen_name in EXP4_GENERATORS:
        for nd in EXP4_N_DRIFTS:
            order = cycling_order(gen_name, nd)
            n_unique = len(set(order))
            recurs = n_unique < len(order)
            print(f"  {gen_name:8s} n_drifts={nd:2d} -> order {order} "
                  f"({n_unique} unique, baseline {1/n_unique:.3f}, "
                  f"{'RECURS' if recurs else 'no recurrence'})")

    print("\nVerification complete (no files written).")