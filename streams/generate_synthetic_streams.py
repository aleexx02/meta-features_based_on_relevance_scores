# streams/generate_synthetic_streams.py

# Synthetic stream generator script: produces synthetic streams used
# across several different experiments.
#
#   Experiment 2 — strlearn StreamGenerator streams with VARYING
#     chunk_size / n_informative (NOT pre-saved; regenerated on the fly
#     by evaluate_concept_classification_2.py via the importable
#     build_exp2_stream / get_exp2_concept_labels helpers below).
#
#   Experiment 3 — river SEA / STAGGER / LED streams, SEQUENTIAL
#     concepts (each concept appears once in order; the only repeat is
#     STAGGER's [0,1,2,0], see note there). Saved to disk.
#
#   Experiment 4 — RECURRING-concept streams, built from SEA and
#     STAGGER (NOT make_classification, NOT strlearn). Concepts cycle
#     TWICE so each one reappears later in the stream, which is the
#     whole point: it lets us test whether ABFS recognises a
#     previously-seen concept (same relevance signature on recurrence)
#     rather than only detecting that *some* change happened. Saved to
#     disk.
#
# ------------------------------------------------------------------
#  TWO THINGS WORTH KNOWING UP FRONT (both verified empirically
#  against the installed river / strlearn before being relied on):
#
#  1. river's STAGGER takes `classification_function`, NOT `variant`
#     (only SEA takes `variant`). Passing variant= to STAGGER raises
#     TypeError at generation time. The factories below use the right
#     keyword per generator.
#
#  2. SEED PROPAGATION. The `seed` passed to the build_* functions
#     now flows all the way into the river data generators (via the
#     factory closures) and into every random choice (transition
#     masks, recurring-order shuffles). This matters because the
#     evaluate scripts for Experiments 3/4 regenerate each stream
#     once per replication seed (like Experiments 1c/2 do with
#     StreamGenerator's random_state) and stack the results into a
#     (n_reps, n_windows, n_clfs) array. If seed did NOT propagate
#     into generation (an earlier version of this file had make_gen
#     lambdas hardcoding MASTER_SEED), every replication of a SUDDEN
#     stream would be byte-identical and the across-rep variance /
#     error bars would be a meaningless zero. Running the __main__
#     block here saves ONE canonical realization at seed=MASTER_SEED;
#     that realization is exactly replication 0 in the evaluate
#     scripts (which put MASTER_SEED first in their seed list), so the
#     on-disk streams stay consistent with rep 0 and with whatever
#     the analysis scripts load.
#
# Saves to disk (data/synthetic/streams/, streams_gt/):
#   Experiment 3:  sea_{sudden,gradual}, stagger_{sudden,gradual},
#                  led_{sudden,gradual}                       (6 streams)
#   Experiment 4:  recurring_{sea,stagger}_{fixed,random}_{sudden,gradual}
#                                                             (8 streams)
#
#   {name}.npy      -> (n_instances, n_features + 1), last column = the
#                      generator's REAL class target (binary for
#                      SEA/STAGGER, 10-class digit for LED). NOT the
#                      concept label.
#   streams_gt/{name}.npy -> (n_chunks,) concept_per_chunk: the
#                      GENERATIVE concept id active in each chunk. For
#                      recurring streams the same id reappears (that's
#                      the recurrence); drift boundaries are recoverable
#                      as np.diff(concept_per_chunk) != 0.

import numpy as np
import os
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
#  RIVER GENERATOR FACTORIES  (seed-aware -- see header note 2)
#  Each factory: (seed, concept_idx) -> a river dataset for that
#  concept. The seed flows in here, so regenerating with a different
#  seed produces genuinely different data (not just a different
#  transition mask).
# ============================================================

def make_sea(seed, concept_idx):
    # SEA: 3 numeric features (first 2 relevant), binary target,
    # 4 thresholds selectable via `variant` (0-3).
    return river_synth.SEA(seed=seed, variant=concept_idx)

def make_stagger(seed, concept_idx):
    # STAGGER: 3 categorical features (size/color/shape), binary
    # target, 3 boolean rules via `classification_function` (0-2).
    # NOTE: classification_function, NOT variant (that's the bug fix).
    return river_synth.STAGGER(seed=seed, classification_function=concept_idx)

def make_led(seed, concept_idx):
    # LED: 24 features (7 real display segments + 17 irrelevant),
    # 10-class target (the digit). LED has no built-in "variant"; its
    # concepts here are distinct random LED instances differentiated by
    # a per-concept seed offset, which is how the original version of
    # this file did it too.
    return river_synth.LED(seed=seed + concept_idx,
                           noise_percentage=0.1, irrelevant_features=True)

GENERATOR_FACTORY = {
    'sea':     make_sea,
    'stagger': make_stagger,
    'led':     make_led,
}

# Number of distinct concepts each generator offers, and feature count.
GENERATOR_N_CONCEPTS = {'sea': 4, 'stagger': 3, 'led': 4}
RIVER_N_FEATURES     = {'sea': 3, 'stagger': 3, 'led': 24}

# Real class-target cardinality (the npy's last column), distinct from
# the concept count above -- used by the characteristics tables / docs.
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


def river_stream_to_arrays(dataset, n, encoder, n_features):
    X, y = [], []
    for x, label in dataset.take(n):
        X.append([encoder.encode(k, v) for k, v in x.items()])
        y.append(int(label))

    if n == 0:
        # Ensure correct 2D shape when empty
        return (
            np.empty((0, n_features), dtype=float),
            np.empty((0,), dtype=int)
        )

    return np.array(X, dtype=float), np.array(y, dtype=int)


RIVER_TRANSITION_CHUNKS = 5
RIVER_TOTAL_CHUNKS      = 200


def build_river_drift_stream(gen_name, concept_order, drift_chunks,
                             transition, seed):

    factory  = GENERATOR_FACTORY[gen_name]
    make_gen = lambda c: factory(seed, c)

    rng     = np.random.RandomState(seed)
    encoder = CategoricalEncoder()
    boundaries = [0] + list(drift_chunks) + [RIVER_TOTAL_CHUNKS]

    n_features = RIVER_N_FEATURES[gen_name]

    X_all, y_all, concept_per_chunk = [], [], []

    for seg_id, concept in enumerate(concept_order):
        seg_start, seg_end = boundaries[seg_id], boundaries[seg_id + 1]
        next_concept = (
            concept_order[seg_id + 1]
            if seg_id + 1 < len(concept_order) else None
        )

        gen_current = make_gen(concept)
        gen_next = make_gen(next_concept) if next_concept is not None else None

        for c in range(seg_start, seg_end):
            in_transition = (
                transition == 'gradual'
                and gen_next is not None
                and c >= seg_end - RIVER_TRANSITION_CHUNKS
            )

            if not in_transition:
                Xc, yc = river_stream_to_arrays(
                    gen_current, CHUNK_SIZE, encoder, n_features
                )
                concept_per_chunk.append(concept)

            else:
                steps = c - (seg_end - RIVER_TRANSITION_CHUNKS)
                progress = (steps + 0.5) / RIVER_TRANSITION_CHUNKS
                p_new = 1.0 / (1.0 + np.exp(-12 * (progress - 0.5)))

                mask_new = rng.random(CHUNK_SIZE) < p_new
                n_old = int((~mask_new).sum())
                n_new = int(mask_new.sum())

                Xa, ya = river_stream_to_arrays(
                    gen_current, n_old, encoder, n_features
                )
                Xb, yb = river_stream_to_arrays(
                    gen_next, n_new, encoder, n_features
                )

                Xc = np.empty((CHUNK_SIZE, n_features))
                yc = np.empty(CHUNK_SIZE, dtype=int)

                # assignments now always safe
                Xc[~mask_new], yc[~mask_new] = Xa, ya
                Xc[mask_new],  yc[mask_new]  = Xb, yb

                concept_per_chunk.append(
                    concept if mask_new.mean() < 0.5 else next_concept
                )

            X_all.append(Xc)
            y_all.append(yc)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    return np.column_stack([X_all, y_all]), np.array(concept_per_chunk)


def even_boundaries(n_segments, total_chunks=RIVER_TOTAL_CHUNKS):
    """Inner drift-chunk boundaries that partition total_chunks into
    n_segments as evenly as possible (segments differ by at most 1)."""
    return [round(i * total_chunks / n_segments) for i in range(1, n_segments)]


# ============================================================
#  EXPERIMENT 3 — SEA / STAGGER / LED, SEQUENTIAL concepts
#  4 segments, 3 drifts. Drift chunk positions match the original
#  recipe (abrupt at 50/100/150; gradual nudged to 48/100/153 so the
#  blend straddles the boundary). STAGGER's order is [0,1,2,0] -- it
#  only has 3 classification functions, so the 4th segment reuses
#  concept 0; that single repeat means STAGGER has 3 UNIQUE concepts
#  here (random baseline 1/3), not 4. SEA and LED have 4 distinct
#  concepts (baseline 1/4).
# ============================================================
DRIFT_CHUNKS_ABRUPT  = [50, 100, 150]
DRIFT_CHUNKS_GRADUAL = [48, 100, 153]

SEA_ORDER     = [0, 1, 2, 3]
STAGGER_ORDER = [0, 1, 2, 0]
LED_ORDER     = [0, 1, 2, 3]

# (gen_name, concept_order)
EXP3_GENERATORS = [
    ('sea',     SEA_ORDER),
    ('stagger', STAGGER_ORDER),
    ('led',     LED_ORDER),
]


def exp3_specs():
    """Returns list of dicts fully describing each Experiment 3 stream,
    including a seed->(data, concept_per_chunk) builder. Imported by
    evaluate_concept_classification_3.py and analysis_3.py so the
    stream set has a single source of truth."""
    specs = []
    for gen_name, order in EXP3_GENERATORS:
        for transition, drifts in [('sudden',  DRIFT_CHUNKS_ABRUPT),
                                   ('gradual', DRIFT_CHUNKS_GRADUAL)]:
            name = f'{gen_name}_{transition}'
            specs.append({
                'name':         name,
                'gen_name':     gen_name,
                'order':        order,
                'drift_chunks': drifts,
                'transition':   transition,
                'n_features':   RIVER_N_FEATURES[gen_name],
                'n_concepts':   len(set(order)),
                'builder': (lambda seed, g=gen_name, o=order, d=drifts, t=transition:
                            build_river_drift_stream(g, o, d, t, seed)),
            })
    return specs


# ============================================================
#  EXPERIMENT 4 — RECURRING concepts via SEA and STAGGER
#  Each generator's concepts cycle TWICE (so every concept reappears),
#  crossed with fixed/random ordering and sudden/gradual transitions.
#
#    SEA     (4 concepts): 8 segments -> 7 drifts
#    STAGGER (3 concepts): 6 segments -> 5 drifts
#
#  fixed  = the second cycle repeats the first cycle's order exactly
#           (e.g. SEA 0,1,2,3,0,1,2,3) -- predictable recurrence.
#  random = the second cycle is a reshuffle of the concept set, with
#           an adjacency guard so the seam between cycles is still a
#           real drift (the reshuffle can't start with the concept the
#           first cycle ended on) -- unpredictable recurrence, a
#           harder test. The shuffle is driven by the seed, so it
#           ALSO varies across replications (not just the data).
#
#  Concept label = generative concept id, so a recurring concept gets
#  the SAME label both times it appears (the entire point of Exp 4).
# ============================================================
EXP4_N_CYCLES   = 2
EXP4_GENERATORS = ['sea', 'stagger']
EXP4_RECURRENCES = ['fixed']   # 'random' available but dropped for now (simple fixed cycling only)
EXP4_TRANSITIONS = ['sudden', 'gradual']


def build_recurring_order(gen_name, recurrence, rng):
    """Concept order with EXP4_N_CYCLES cycles. First cycle is always
    the natural order; subsequent cycles repeat it (fixed) or reshuffle
    it (random, with an adjacency guard at the cycle seam)."""
    n    = GENERATOR_N_CONCEPTS[gen_name]
    base = list(range(n))
    order = list(base)
    for _ in range(EXP4_N_CYCLES - 1):
        if recurrence == 'fixed':
            nxt = list(base)
        elif recurrence == 'random':
            nxt = base.copy()
            rng.shuffle(nxt)
            # guard: don't let the seam collapse (no drift) when the
            # reshuffled cycle would start on the concept we just ended
            while nxt[0] == order[-1] and n > 1:
                rng.shuffle(nxt)
        else:
            raise ValueError(recurrence)
        order.extend(nxt)
    return order


def build_exp4_stream(gen_name, recurrence, transition, seed):
    """seed->(data, concept_per_chunk) for one Experiment 4 stream.
    Uses a seed-derived RNG for the (random) recurrence order, then
    delegates the actual instance generation to build_river_drift_stream
    (which independently seeds its transition-mask RNG from the same
    seed). Drift boundaries partition the 200 chunks evenly across the
    segments."""
    order_rng = np.random.RandomState(seed * 2 + 1)
    order     = build_recurring_order(gen_name, recurrence, order_rng)
    drifts    = even_boundaries(len(order))
    return build_river_drift_stream(gen_name, order, drifts, transition, seed)


def exp4_specs():
    """Returns list of dicts fully describing each Experiment 4 stream,
    including a seed->(data, concept_per_chunk) builder. Single source
    of truth for evaluate_concept_classification_4.py / analysis_4.py.

    n_concepts is the count of DISTINCT concepts (= the generator's
    concept count: SEA 4, STAGGER 3); the streams are 'recurring'
    because those concepts each appear in multiple segments, not
    because there are extra concepts."""
    specs = []
    for gen_name in EXP4_GENERATORS:
        for recurrence in EXP4_RECURRENCES:
            for transition in EXP4_TRANSITIONS:
                name = f'recurring_{gen_name}_{recurrence}_{transition}'
                n_concepts = GENERATOR_N_CONCEPTS[gen_name]
                specs.append({
                    'name':       name,
                    'gen_name':   gen_name,
                    'recurrence': recurrence,
                    'transition': transition,
                    'n_features': RIVER_N_FEATURES[gen_name],
                    'n_concepts': n_concepts,
                    'n_cycles':   EXP4_N_CYCLES,
                    'builder': (lambda seed, g=gen_name, r=recurrence, t=transition:
                                build_exp4_stream(g, r, t, seed)),
                })
    return specs


# ============================================================
#  Experiment 2 — StreamGenerator (strlearn) with VARYING parameters
#  NOT pre-saved to disk: the full 4x4 grid x 2 drift types x 5
#  replications runs into tens of GB at chunk_size up to 1000, and
#  regenerating from a fixed seed is cheap -- the expensive part is
#  the ABFS/Komorniczak feature EXTRACTION downstream, not generation.
#  Imported by evaluate_concept_classification_2.py instead of
#  duplicated there.  (Unchanged from the previous version of this
#  file -- left exactly as-is so Experiment 2 keeps reproducing.)
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
#  NOT trigger generation as a side effect of that import). Saves the
#  canonical seed=MASTER_SEED realization, which equals replication 0
#  in the Exp 3 / Exp 4 evaluate scripts.
# ============================================================
if __name__ == "__main__":

    def save_if_absent(stream_name, builder):
        stream_path = os.path.join(STREAM_DIR, f'{stream_name}.npy')
        gt_path     = os.path.join(GT_DIR,     f'{stream_name}.npy')
        print(f"\n{stream_name}")
        if os.path.exists(stream_path) and os.path.exists(gt_path):
            print(f"  EXISTS — skipping.")
            return
        data, concept_per_chunk = builder(MASTER_SEED)
        np.save(stream_path, data)
        np.save(gt_path, concept_per_chunk)
        n_boundaries = int(np.sum(np.diff(concept_per_chunk) != 0))
        counts = dict(zip(*np.unique(concept_per_chunk, return_counts=True)))
        print(f"  SAVED  : shape={data.shape}  n_chunks={len(concept_per_chunk)}  "
              f"unique_concepts={sorted(counts)}  n_boundaries={n_boundaries}")

    print("=" * 60)
    print("Experiment 3: SEA / STAGGER / LED (sequential concepts)")
    print("=" * 60)
    for spec in exp3_specs():
        save_if_absent(spec['name'], spec['builder'])

    print(f"\n{'='*60}")
    print("Experiment 4: recurring concepts (SEA, STAGGER)")
    print(f"{'='*60}")
    for spec in exp4_specs():
        save_if_absent(spec['name'], spec['builder'])

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    for spec in exp3_specs() + exp4_specs():
        stream_name = spec['name']
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
              f"n_concepts={len(counts)}  per-concept chunks={counts}  "
              f"n_boundaries={n_boundaries}  baseline={1/len(counts):.3f}")

    print("\nALL DONE.")