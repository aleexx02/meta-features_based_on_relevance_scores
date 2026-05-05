from river import datasets
import numpy as np


"""
Stream generators for sanity check and full experimental evaluation.

Core contrast:
    SEA (boundary drift only) → meta-features stable → drift_count=0
    STAGGER (feature drift)   → meta-features change → drift_count>0

Configurations:
    1. make_sea_sudden_drift        → boundary drift, sudden
    2. make_sea_gradual_drift       → boundary drift, gradual
    3. make_stagger_feature_drift   → feature drift, sudden
    4. make_stagger_gradual_drift   → feature drift, gradual
    5. make_stagger_recurring       → recurring concepts
    6. make_sea_stationary          → no drift (null hypothesis)
    7. make_sea_multi_drift         → multiple boundary drifts
    8. make_stagger_multi_drift     → multiple feature drifts
    9. make_stagger_simultaneous    → boundary + feature drift simultaneously
"""
"""
This is useful for testing whether your meta-features can detect the difference between:
sudden drift  → scores shift abruptly at one window
gradual drift → scores shift slowly over several windows
               → delta_mean stays elevated for longer
               → drift_count may be lower or zero
"""

# SEA has 3 features: f1, f2 (relevant) and f3 (irrelevant).
# STAGGER has 3 categorical features: size, color, shape.


def make_sea_sudden_drift(
    variant_before=0,
    variant_after=1,
    drift_position=5000,
    seed=42
):
    """
    SEA stream with a single SUDDEN concept drift (boundary drift).
    Useful for sanity check: same features remain relevant after drift:
    BEFORE DRIFT: f1 + f2 <= 8 → class 0, else class 1
    AFTER DRIFT: f1 + f2 <= 9 → class 0, else class 1
    The decision boundary shifts but the relevant features remain the same.
    
    Testing: Does ABFS remain stable when feature relevance does NOT change?
    ABFS scores should stay stable: same features remain relevant, no drift signals should fire, same selected features, and therefore, similar meta-features before and after drift.
    """
    stream_a = datasets.synth.SEA(variant=variant_before, seed=seed)
    stream_b = datasets.synth.SEA(variant=variant_after, seed=seed + 1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)

        i = 0
        while True:
            if i < drift_position:
                yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()



def make_sea_gradual_drift(
    variant_before=0,
    variant_after=1,
    drift_position=5000,
    drift_width=1000,
    seed=42
):
    
    """
    SEA stream with a single GRADUAL concept drift (boundary drift).
    Useful for testing whether meta-features capture drift magnitude.
    Useful for sanity check: same features remain relevant after drift:
    BEFORE DRIFT: f1 + f2 <= 8 → class 0, else class 1
    AFTER DRIFT: f1 + f2 <= 9 → class 0, else class 1
    The decision boundary shifts but the relevant features remain the same.
    
    Testing: Does ABFS remain stable when feature relevance does NOT change?
    ABFS scores should stay stable: same features remain relevant, no drift signals should fire, same selected features, and therefore, similar meta-features before and after drift.
    """
    stream_a = datasets.synth.SEA(variant=variant_before, seed=seed)
    stream_b = datasets.synth.SEA(variant=variant_after, seed=seed + 1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)
        rng  = np.random.default_rng(seed)
        i    = 0

        while True:
            if i < drift_position:
                yield next(it_a)
            elif i < drift_position + drift_width:
                # transition zone — mix both concepts probabilistically
                # probability of concept B increases linearly
                p_b = (i - drift_position) / drift_width
                if rng.random() < p_b:
                    yield next(it_b)
                else:
                    yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()


def make_stagger_sudden_drift_01(
    concept_before=0,
    concept_after=1,
    drift_position=5000,
    seed=42
):
    """
    STAGGER stream with SUDDEN feature drift.
    
    concept 0: size=small AND color=red → class 1 (relevant features: size, color)
    concept 1: color=green OR shape=circle → class 1 (relevant features: color, shape)
    concept 2: size=medium OR size=large → class 1 (relevant features: size)
    
    Between concept 0 and concept 1:
    - size becomes IRRELEVANT (was relevant in concept 0)
    - shape becomes RELEVANT (was irrelevant in concept 0)
    - genuine feature drift: F*_t changes
    FEATURES CHANGE, NOT JUST DECISION BOUNDARY → ABFS scores should shift significantly, drift_count should increase, and meta-features should reflect a major change in relevance structure.
    Testing: do ABFS scores change when relevant features change?
    Expected: drift_count > 0, n_changed > 0, entropy changes

    drift_count > 0 -> feature drift detected
    n_changed > 0 -> relevant features changed
    entropy shifts -> relevance structure changed
    delta_mean large -> scores shifted significantly at drift

    The difference between concept A and concept B meta-feature means should be much larger than what you saw for SEA.
    """
    stream_a = datasets.synth.STAGGER(classification_function=concept_before, seed=seed)
    stream_b = datasets.synth.STAGGER(classification_function=concept_after, seed=seed+1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)
        i = 0
        while True:
            if i < drift_position:
                yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()


def make_stagger_gradual_drift(
    concept_before=0,
    concept_after=1,
    drift_position=5000,
    drift_width=1000,
    seed=42
):
    """
    STAGGER stream with GRADUAL feature drift.

    Same concept change as make_stagger_feature_drift_01 but the
    transition happens gradually over drift_width instances.

    Expected ABFS behavior:
    - relevance scores shift progressively during transition
    - drift_count fires more frequently during transition window
    - delta_mean stays elevated for several windows
    - n_changed increases progressively during transition
    - meta-features transition smoothly between concepts

    Role: tests sensitivity to gradual feature drift.
    """
    stream_a = datasets.synth.STAGGER(
        classification_function=concept_before, seed=seed)
    stream_b = datasets.synth.STAGGER(
        classification_function=concept_after, seed=seed + 1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)
        rng  = np.random.default_rng(seed)
        i    = 0

        while True:
            if i < drift_position:
                yield next(it_a)
            elif i < drift_position + drift_width:
                p_b = (i - drift_position) / drift_width
                if rng.random() < p_b:
                    yield next(it_b)
                else:
                    yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()


def make_stagger_recurring(
    concept_a=0,
    concept_b=1,
    cycle_length=3000,
    n_cycles=3,
    seed=42
):
    """
    STAGGER stream with RECURRING concepts.

    Alternates between concept_a and concept_b every cycle_length
    instances for n_cycles full cycles.

    concept 0: size=small AND color=red → class 1
    concept 1: color=green OR shape=circle → class 1

    Expected ABFS behavior:
    - relevance scores oscillate between two patterns
    - drift_count fires at each concept switch
    - meta-features return to earlier patterns when concept recurs
    - delta_mean spikes at each switch and returns to near zero

    Role: tests whether meta-features recognize recurring concepts.
    """
    stream_a = datasets.synth.STAGGER(
        classification_function=concept_a, seed=seed)
    stream_b = datasets.synth.STAGGER(
        classification_function=concept_b, seed=seed + 1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)
        i = 0
        total = cycle_length * n_cycles * 2

        while i < total:
            cycle_pos = i % (cycle_length * 2)
            if cycle_pos < cycle_length:
                yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()


def make_sea_stationary(
    variant=0,
    seed=42
):
    """
    SEA stream with NO drift — completely STATIONARY.

    BEFORE AND AFTER: f1 + f2 <= 8 → class 0, else class 1
    No concept change occurs at any point.

    Expected ABFS behavior:
    - relevance scores stable throughout
    - drift_count = 0 throughout
    - n_changed = 0 throughout
    - delta_mean near zero throughout
    - meta-features consistent across all windows

    Role: null hypothesis check — verifies no false alarms
    when the stream is stationary.
    """
    stream = datasets.synth.SEA(variant=variant, seed=seed)

    def generator():
        it = iter(stream)
        while True:
            yield next(it)

    return generator()


def make_sea_multi_drift(
    n_drifts=4,
    drift_interval=2000,
    seed=42
):
    """
    SEA stream cycling through ALL 4 variants with SUDDEN drifts.
    Variants: 0,1,2,3 — all boundary drift, no feature drift.
    
    Expected: drift_count=0 throughout, meta-features stable
    between drifts, similar across all concepts.
    
    Role: confirms stability under multiple boundary drifts.
    """
    variants = [0, 1, 2, 3]
    streams  = [
        datasets.synth.SEA(variant=v, seed=seed + v)
        for v in variants
    ]

    def generator():
        iterators = [iter(s) for s in streams]
        i = 0
        while True:
            concept_idx = min(i // drift_interval, len(variants) - 1)
            yield next(iterators[concept_idx])
            i += 1

    return generator()


def make_stagger_multi_drift(
    drift_interval=2000,
    seed=42
):
    """
    STAGGER stream cycling through ALL 3 concepts with SUDDEN drifts.
    
    concept 0 → concept 1 → concept 2 → concept 0 (recurring)
    
    concept 0: size=small AND color=red      (relevant: size, color)
    concept 1: color=green OR shape=circle   (relevant: color, shape)
    concept 2: size=medium OR size=large     (relevant: size)
    
    Feature changes at each drift:
    0→1: size irrelevant, shape becomes relevant
    1→2: shape irrelevant, color irrelevant, size relevant again
    2→0: color becomes relevant again, shape irrelevant
    
    Expected:
    - drift_count > 0 at each concept switch
    - meta-features clearly different per concept
    - meta-features return to concept 0 pattern when it recurs
    - delta_mean spikes at each drift, returns to near zero after
    
    Role: tests meta-features across multiple feature drifts
    and concept recurrence.
    """
    concepts = [0, 1, 2, 0]  # concept 0 recurs at the end
    streams  = [
        datasets.synth.STAGGER(
            classification_function=c, seed=seed + i)
        for i, c in enumerate(concepts)
    ]

    def generator():
        iterators = [iter(s) for s in streams]
        i = 0
        while True:
            concept_idx = min(i // drift_interval, len(concepts) - 1)
            yield next(iterators[concept_idx])
            i += 1

    return generator()


def make_stagger_sudden_drift_02(
    concept_before=0,
    concept_after=2,
    drift_position=5000,
    seed=42
):
    """
    STAGGER stream with SUDDEN feature drift.
    
    concept 0: size=small AND color=red → class 1 (relevant features: size, color)
    concept 1: color=green OR shape=circle → class 1 (relevant features: color, shape)
    concept 2: size=medium OR size=large → class 1 (relevant features: size)
    
    Between concept 0 and concept 2:
    - color becomes IRRELEVANT (was relevant in concept 0)
    - genuine feature drift: F*_t changes
    FEATURES CHANGE, NOT JUST DECISION BOUNDARY → ABFS scores should shift significantly, drift_count should increase, and meta-features should reflect a major change in relevance structure.
    Testing: do ABFS scores change when relevant features change?
    Expected: drift_count > 0, n_changed > 0, entropy changes

    drift_count > 0 -> feature drift detected
    n_changed > 0 -> relevant features changed
    entropy shifts -> relevance structure changed
    delta_mean large -> scores shifted significantly at drift

    The difference between concept A and concept B meta-feature means should be much larger than what you saw for SEA.
    """
    stream_a = datasets.synth.STAGGER(
        classification_function=concept_before, seed=seed)
    stream_b = datasets.synth.STAGGER(
        classification_function=concept_after, seed=seed + 1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)
        i = 0
        while True:
            if i < drift_position:
                yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()


def make_stagger_sudden_drift_12(
    concept_before=1,
    concept_after=2,
    drift_position=5000,
    seed=42
):
    """
    STAGGER stream with SUDDEN feature drift.
    
    concept 0: size=small AND color=red → class 1 (relevant features: size, color)
    concept 1: color=green OR shape=circle → class 1 (relevant features: color, shape)
    concept 2: size=medium OR size=large → class 1 (relevant features: size)
    
    Between concept 1 and concept 2:
    - color and shape become IRRELEVANT (were relevant in concept 1)
    - genuine feature drift: F*_t changes
    FEATURES CHANGE, NOT JUST DECISION BOUNDARY → ABFS scores should shift significantly, drift_count should increase, and meta-features should reflect a major change in relevance structure.
    Testing: do ABFS scores change when relevant features change?
    Expected: drift_count > 0, n_changed > 0, entropy changes

    drift_count > 0 -> feature drift detected
    n_changed > 0 -> relevant features changed
    entropy shifts -> relevance structure changed
    delta_mean large -> scores shifted significantly at drift

    The difference between concept A and concept B meta-feature means should be much larger than what you saw for SEA.
    """
    stream_a = datasets.synth.STAGGER(
        classification_function=concept_before, seed=seed)
    stream_b = datasets.synth.STAGGER(
        classification_function=concept_after, seed=seed + 1)

    def generator():
        it_a = iter(stream_a)
        it_b = iter(stream_b)
        i = 0
        while True:
            if i < drift_position:
                yield next(it_a)
            else:
                yield next(it_b)
            i += 1

    return generator()
