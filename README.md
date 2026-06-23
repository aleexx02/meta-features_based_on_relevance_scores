# Meta-features Based on Relevance Scores

ABFS-based stream meta-features for concept classification in non-stationary data streams.

---

## What is being evaluated?

Each experiment evaluates whether meta-features computed per stream window can discriminate the active concept.

- Input: meta-feature vector per window
- Target: concept label (definition depends on stream type)
- Task: multi-class classification at the meta-level

High performance implies that the meta-features encode concept identity effectively.

How the concept label is defined differs by tier:

Every stream `.npy` carries a **target class** in its last column: what ABFS and Komorniczak train their per-window meta-features against (SEA's binary threshold, LED's digit, INSECTS' species, SPAM's spam/legit). This is **not** the **concept label**, which is what the meta-level classifier predicts from the meta-features.

* **Synthetic (Exp 1-4)**: the concept label is the **generative concept id**: exact and known, because we built the stream. For recurring streams (Exp 4) the same id reappears.
* **Real (Exp 5)**: the concept label is **positional**: concept = number of drift boundaries passed so far, because real data tells us *where* drift happens but not *what* each segment is.


## Project Structure
```
meta-features_based_on_relevance_scores/
│
├── abfs/
│   └── abfs_implementation.py
│
├── data/  # local data files - not committed to git                        
│   └── real/
│       ├── annotated_streams/
│       │   ├── INSECTS-abrupt_balanced.npy
│       │   ├── INSECTS-abrupt_imbalanced.npy
│       │   ├── INSECTS-incgradual_balanced.npy
│       │   ├── INSECTS-incgradual_imbalanced.npy
│       │   └── SPAM.npy
│       ├── annotated_streams_gt/
│       │   └── {same filenames}.npy
│       └── annotated_streams_analysis/
│           └── {stream analysis files}.npy
│
├── experiments/
│   ├── experiment_0/
│   │   ├── comparison.py
│   │   └── replication_check_1a.py
│   │
│   ├── experiment_1a/
│   │   └── evaluate_concept_classification_1a.py
│   │
│   ├── experiment_1b/
│   │   ├── evaluate_concept_classification_1b.py
│   │   └── komor_concept_classification_1b.py
│   │
│   ├── experiment_1c/
│   │   ├── analysis_1c.py
│   │   ├── evaluate_concept_classification_1c.py
│   │   └── komor_concept_classification_1c.py
│   │
│   ├── experiment_2/
│   │   ├── analysis_2.py
│   │   └── evaluate_concept_classification_2.py
│   │
│   ├── experiment_3/ # SEA, STAGGER, LED (sequential)
│   │   ├── analysis_3.py
│   │   └── evaluate_concept_classification_3.py
│   │
│   ├── experiment_4/ # recurring concepts (SEA, STAGGER)
│   │   ├── analysis_4.py
│   │   └── evaluate_concept_classification_4.py
│   │
│   ├── experiment_5/ # Real Streams: INSECTS + SPAM
│   │   ├── analysis_5.py
│   │   └── evaluate_concept_classification_5.py
│   │
│   ├── analysis_1a_1b.py
│   ├── classifier_sweep_komor.py
│   └── classifier_sweep_prequential.py
│
├── external/
│   └── komorniczak/
│       ├── results/
│       │   ├── real/ # Exp 5 pymfe cache
│       │   └── synthetic/ # Exp 0 cache
│       │   └── synthetic_sea_stagger_led/ # Exp 3 pymfe cache
│       │   └── synthetic_recurring/ # Exp 4 pymfe cache
│       ├── E1_extract_synthetic.py
│       ├── E2_clf_synthetic.py
│       └── utils.py
│
├── full_pipeline/
│   └── pipeline.py
│
├── metafeatures/
│   └── mf_extraction.py
│
├── results/
│   ├── experiment_0/
│   │   └── figures/
│   ├── experiment_1a/
│   │   └── figures/
│   │       └── analysis/
│   ├── experiment_1b/
│   │   └── figures/
│   │       └── analysis/
│   ├── experiment_1c/
│   │   └── figures/
│   │       └── analysis/
│   ├── experiment_2/
│   │   └── figures/
│   │       └── analysis/
│   ├── experiment_3/
│   │   ├── figures/
│   │       └── analysis/
│   └── experiment_4/
│   │   ├── figures/
│   │       └── analysis/
│   └── experiment_5/
│   │   ├── figures/
│   │       └── analysis/
│   └── sanity_check/
│       └── figures/
│
├── streams/
│   ├── generate_real_streams.py
│   ├── generate_synthetic_streams.py
│   └── generators.py
│
├── .gitignore
├── plot_results.py
├── README.md
├── requirements.txt
└── sanity_check.py
```

---

## Data Files

All data files are gitignored and must be generated locally.

### Setup

**Step 1a: Download the USP DS Repository**

All stream files are sourced from the USP DS Repository
(Souza et al., 2020): https://sites.google.com/view/uspdsrepository

Download the full ZIP and unzip it:
```bash
pip install gdown
gdown "1JERZnbGGToAEz_3LRV7n2Vz79LiDAEY-" -O ~/usp_ds_repository.zip
unzip ~/usp_ds_repository.zip -d ~/usp_ds_repository/
```

**Step 1b: Download the SPAM dataset**
Sourced from Katakis, Tsoumakas, Vlahavas (2010), Knowledge and Information Systems, 22(3), 371-391. Download page:
http://mlkd.csd.auth.gr/concept_drift.html

Click on the "Spam Data" link under Datasets 3.

```bash
mkdir -p ~/spam_data && cd ~/spam_data
wget http://lpis.csd.auth.gr/mlkd/concept_drift/spam_data.rar
unrar x spam_data.rar
```

**Step 2: Generate the real streams (Experiment 5 only):**
```bash
# Experiment 5: INSECTS, SPAM
python streams/generate_real_streams.py
```

**There is no generation step for Experiments 1c, 2, 3 and 4.** Their streams are built on the fly per grid cell inside the evaluate scripts, exactly like Experiment 2's StreamGenerator streams. `generate_synthetic_streams.py` is a builder *library*: it exposes Exp 2, 3 and 4 helpers, and writes nothing to disk. Running it directly just prints the grid layout and smoke-tests the builders.

---

### Two tiers of evaluation

The experiments operate on two types of streams, differing in how concept drift is defined and how reliable the ground truth is.


| Tier | Drift location | Concept identity | Data type | Experiments |
|------|--------------|------------------|----------|
| Synthetic | YES | YES | synthetic (generative id) | 1c, 2, 3, 4 |
| Real annotated | YES (exact INSECTS / approx SPAM) | NO (positional) | real | 5 |

---

### Tier 1 (Exp 1c, 2, 3, 4): concepts and drift fully controlled. Drift
types *sudden* (instantaneous) and *gradual* (sigmoid blend). Experiment 4 additionally has concepts that **recur** (reappear with the same generative id).

Both **where drift occurs** and **what changes** are known exactly.

---

### Tier 2 (Exp 5): real INSECTS + SPAM

Real INSECTS and SPAM data. Concept = positional segment between annotated drift points; concept identity is unknown, so each segment is treated as distinct even if two are statistically similar.

INSECTS drift exact (Souza et al. 2020, Table 2); SPAM approximate (Katakis et al. 2010 via Yu et al. 2018).

Important:
- Drift location is known.
- Concept identity is unknown.

---

## Stream Configurations

Each `.npy` is `(n_instances, n_features + 1)`, last column = **target class** (not the concept label). "Concepts" = number of distinct concept labels; "baseline" = `1 / n_concepts`.

### Experiment 3 - SEA / STAGGER / LED (sequential, chunk_size sweep)

`river.datasets.synth`. **500,000 instances** per stream. The one swept axis is **chunk_size ∈ {100, 200, 500}** (5000 / 2500 / 1000 chunks respectively). Each generator runs its fixed concept order once; 3 drifts placed evenly. **Grid = 3 generators $\times$ 2 drift types $\times$ 3 chunk_sizes = 18 cells.**

| Generator | Features | Target classes | Concept order | Concepts | Baseline |
|---|---|---|---|---|---|
| SEA | 3 (2 relevant) | 2 (binary) | [0,1,2,3] | 4 | 0.250 |
| STAGGER | 3 (size/color/shape) | 2 (binary) | [0,1,2,0] | 3 | 0.333 |
| LED | 24 (7 relevant + 17 noise) | 10 (digit) | [0,1,2,3] | 4 | 0.250 |


### Experiment 4 - recurring SEA / STAGGER (chunk_size $\times$ n_drifts grid)

`river.datasets.synth`. **500,000 instances** per stream. Full grid:
**chunk_size ∈ {100,200,500} $\times$ n_drifts ∈ {1,3,7,15}**, $\times$ 2 generators $\times$ 2 drift types = **64 cells.** Concepts **cycle** through the
generator's set (segment i $\to$ concept i mod n_concepts), so recurrence grows with n_drifts.

| n_drifts | Segments | SEA order | SEA concepts (baseline) | STAGGER order | STAGGER concepts (baseline) |
|---|---|---|---|---|---|
| 1 | 2 | [0,1] | 2 (0.500) | [0,1] | 2 (0.500) |
| 3 | 4 | [0,1,2,3] | 4 (0.250) | [0,1,2,0] | 3 (0.333) |
| 7 | 8 | [0,1,2,3,0,1,2,3] | 4 (0.250) | [0,1,2,0,1,2,0,1] | 3 (0.333) |
| 15 | 16 | 4 cycles of [0,1,2,3] | 4 (0.250) | 16-segment cycle | 3 (0.333) |

Recurrence appears once `n_drifts + 1` exceeds the generator's concept
count: at n_drifts = 1 (and SEA n_drifts = 3) concepts appear once each (no recurrence); from n_drifts = 7 on they genuinely recur. Why this is the only way to get a drift-count axis: SEA has exactly 4 concepts and STAGGER exactly 3, so more drifts must reuse concepts.


### Experiment 5 - real (INSECTS + SPAM)

| Stream | Features | Target classes | Chunks@100 | Concepts | Baseline | Drift |
|---|---|---|---|---|---|---|
| INSECTS-abrupt_balanced | 33 | 6 (species) | 264 | 6 | 0.167 | exact |
| INSECTS-abrupt_imbalanced | 33 | 6 (species) | 1,776 | 6 | 0.167 | exact |
| INSECTS-incgradual_balanced | 33 | 6 (species) | 120 | 2 | 0.500 | exact |—
| INSECTS-incgradual_imbalanced | 33 | 6 (species) | 716 | 2 | 0.500 | exact |
| SPAM | 499 | 2 (spam/legit) | ~46 | 6 | 0.167 | approximate |

### Experiments 1c & 2 — `strlearn` StreamGenerator

Binary-target streams (`make_classification`-backed); drift via concept
switching. Concept label is multi-class even though the target is binary.

| | n_features | n_informative | n_red / n_rep | Target classes | n_drifts | n_concepts | chunk_size |
|---|---|---|---|---|---|---|---|
| Exp 2 sudden | 20 | {3,5,10,15} | 0 / 0 | 2 | 20 | 21 | {100,200,500,1000} |
| Exp 2 gradual | 20 | {3,5,10,15} | 0 / 0 | 2 | 6 | 25 | {100,200,500,1000} |

> Experiment 1c also uses StreamGenerator; its exact `n_drifts` /
> `n_informative` live in `evaluate_concept_classification_1c.py` — fill
> this row from that script.

---

## ABFS Meta-Feature Versions

| Version | Name | Dim | Description |
|---|---|---|---|
| v1.1 | aggstats | 8 | entropy, n_relevant, max_score, std_score, delta_mean, n_changed, drift_count, time_since_drift |
| v2.0 | raw scores | n_features | Normalized relevance score vector — feature identity preserved |
| v2.1 | raw + temporal | n_features + 2 | v2.0 + delta_mean + cosine_sim |

---

## Evaluation Protocol

Prequential (test-then-train) throughout (batch CV in 1a/1b is historical).
Classifiers: River GNB, KNN, HT + sklearn MLP (PAC excluded, BA ≈ 0.095).

**Replications.** Experiments 1c, 2, 3, 4 regenerate each stream/cell from
several seeds and stack into `(n_reps, n_windows, n_clfs)` (`n_reps = 5`).
Experiment 5 has no replication axis — real streams are fixed,
`(n_windows, n_clfs)`. For Exp 3/4 the seed propagates into the river
generators, so the replications are genuinely different realizations;
replication 0 uses `MASTER_SEED`.

**ABFS window size** ties to each cell's chunk_size.

---

## Execution Order

### Experiment 0 — Pipeline Verification
**1.** `external/komorniczak/E1_extract_synthetic.py`
**2.** `external/komorniczak/E2_clf_synthetic.py`
**3.** `experiments/experiment_0/comparison.py`
**4.** `experiments/experiment_0/replication_check_1a.py`

### Experiments 1a & 1b — Batch CV (historical)
**5.** `experiments/experiment_1a/evaluate_concept_classification_1a.py`
**6.** `experiments/analysis_1a_1b.py --exp 1a --sanity --variance --shap --metrics`
**7.** `experiments/experiment_1b/komor_concept_classification_1b.py`
**8.** `experiments/experiment_1b/evaluate_concept_classification_1b.py`
**9.** `experiments/analysis_1a_1b.py --exp 1b --sanity --variance --shap --metrics`

### Experiment 1c — Prequential
**10.** `experiments/experiment_1c/komor_concept_classification_1c.py`
**11.** `experiments/experiment_1c/evaluate_concept_classification_1c.py`
**12.** `experiments/experiment_1c/analysis_1c.py --sanity --performance --shap --metrics --stream_analysis --gap`

### Experiment 2 — Stream Configuration Sensitivity
chunk_size ∈ {100,200,500,1000} × n_informative ∈ {3,5,10,15}, ×2 drift ×5 reps.
**13.** `experiments/experiment_2/evaluate_concept_classification_2.py`
**14.** `experiments/experiment_2/analysis_2.py --sanity --performance --shap --metrics --grid --stream_analysis`

### Experiment 3 — SEA / STAGGER / LED (chunk_size sweep)
**Purpose:** named-generator characterization — how does window size
(chunk_size) affect ABFS vs Komorniczak on three classic generators, at
500k instances?

**15.** `experiments/experiment_3/evaluate_concept_classification_3.py`
Regenerates each cell (generator × drift × chunk_size) per replication
seed; ABFS (3 versions) + Komorniczak (9 measures) inline; caches pymfe to
`external/komorniczak/results/synthetic_sea_stagger_led/`. Output per cell
(shape `(n_reps, n_windows, n_clfs)`):
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_{drift}.npy   (+ _f1_, _kappa_)
preq_komor_{measure}_ba_{gen}_chunk{cs}_{drift}.npy  (+ _f1_, _kappa_)
concept_labels_{gen}_chunk{cs}_{drift}.npy
heatmap_comparison_komorniczak_ABFS_preq_exp3_{gen}_chunk{cs}_{drift}.png
```

**16.** `experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics --stream_analysis --gap --grid`
`--grid` adds BA-vs-chunk_size sensitivity curves (ABFS raw v2.0 vs
Komorniczak best) per generator × drift. Per-cell figures into
`results/experiment_3/figures/analysis/`.

### Experiment 4 — Recurring concepts (chunk_size × n_drifts grid)
**Purpose:** the one experiment where concepts recur; characterize ABFS vs
Komorniczak across window size and drift frequency / recurrence amount.

**17.** `experiments/experiment_4/evaluate_concept_classification_4.py`
Same machinery; grid is generator × drift × chunk_size × n_drifts (64
cells); pymfe cache in `external/komorniczak/results/synthetic_recurring/`.
Output per cell:
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy   (+ _f1_, _kappa_)
preq_komor_{measure}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy  (+ _f1_, _kappa_)
concept_labels_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
heatmap_comparison_komorniczak_ABFS_preq_exp4_{gen}_chunk{cs}_ndrift{nd}_{drift}.png
```
This grid is large (64 cells × 5 reps × 12 feature sets). Consider running
cells as independent OAR jobs; the pymfe cache makes cells resumable.

**18.** `experiments/experiment_4/analysis_4.py --sanity --performance --shap --metrics --stream_analysis --gap --grid`
`--grid` adds chunk_size × n_drifts gap heatmaps (ABFS minus Komorniczak
best) per generator × drift, and BA-vs-n_drifts curves at each chunk_size.

### Experiment 5 — real streams (INSECTS + SPAM)
**19.** `streams/generate_real_streams.py`
**20.** `experiments/experiment_5/evaluate_concept_classification_5.py`
Iterates `REAL_STREAMS`; pymfe cache in `external/komorniczak/results/real/`.
Output per stream (shape `(n_windows, n_clfs)` — no rep axis):
```
preq_abfs_{version}_ba_{stream}.npy   (+ _f1_, _kappa_)
preq_komor_{measure}_ba_{stream}.npy  (+ _f1_, _kappa_)
concept_labels_{stream}.npy
heatmap_comparison_komorniczak_ABFS_preq_exp5_{stream}.png
```
**21.** `experiments/experiment_5/analysis_5.py --sanity --performance --shap --metrics --stream_analysis --gap`
SPAM (499 feat) caps per-feature plots to the top 20 by relevance-score
variance (PCA never capped).

---

## Result File Naming Conventions

### Experiment 2
```
preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy
preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy
  shape: (n_reps, n_windows, n_clfs), n_reps=5
```

### Experiment 3
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_{drift}.npy
preq_komor_{measure}_ba_{gen}_chunk{cs}_{drift}.npy
  shape: (n_reps, n_windows, n_clfs), n_reps=5
  gen   ∈ {sea, stagger, led}
  cs    ∈ {100, 200, 500, 1000}
  drift ∈ {sudden, gradual}
```

### Experiment 4
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
preq_komor_{measure}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
  shape: (n_reps, n_windows, n_clfs), n_reps=5
  gen   ∈ {sea, stagger}
  cs    ∈ {100, 200, 500, 1000}
  nd    ∈ {1, 3, 7, 15}
  drift ∈ {sudden, gradual}
```

### Experiment 5
```
preq_abfs_{version}_ba_{stream}.npy
preq_komor_{measure}_ba_{stream}.npy
  shape: (n_windows, n_clfs) — no replications
  stream ∈ {INSECTS-abrupt_balanced, INSECTS-abrupt_imbalanced,
            INSECTS-incgradual_balanced, INSECTS-incgradual_imbalanced, SPAM}
```

Shared field values:
| Field | Values |
|---|---|
| `version` | `aggstats` \| `raw` \| `raw_temporal` |
| `measure` | `clustering` \| `complexity` \| `concept` \| `general` \| `info-theory` \| `itemset` \| `landmarking` \| `model-based` \| `statistical` |

---

## Key Findings

| Finding | Result |
|---|---|
| Shuffling (1a vs 1b) | <0.002 BA — non-recurring concepts have no temporal structure |
| Raw vs aggstats | v2.0 >> v1.1 on synthetic; v1.1 competitive on high-dim real |
| Temporal features (v2.1) | No improvement — delta_mean, cosine_sim rank last in SHAP |
| Sudden drift (Exp 2) | ABFS competitive at high n_informative (crossover ≈ n_inf=10) |
| Gradual drift (Exp 2) | Komorniczak consistently better — adaptation lag compounds |
| Prequential protocol | Komorniczak structural advantage: no memory → instant adaptation |
| PAC classifier | Fails completely (BA ≈ 0.095) — excluded everywhere |
| n_informative (Exp 2) | Key driver: ABFS rises, Komorniczak flat/falls as n_inf increases |
| chunk_size (Exp 3) | To be determined — window-size sensitivity on named generators |
| Recurrence (Exp 4) | To be determined — does ABFS recognise a recurring concept across the cs × n_drifts grid? |
| Real drift type (Exp 5) | Abrupt/gradual: ABFS competitive; incremental: near baseline |
| Real dimensionality (Exp 5) | v2.0 dominates low-dim; v1.1 more robust high-dim (INSECTS, SPAM) |
| SPAM vs INSECTS (Exp 5) | TBD — first real test on a 499-feature approximately-annotated stream |








### Stream Configurations

Every `.npy` stream is `(n_instances, n_features + 1)` with the last column the target class (not the concept label). "Concepts" below is the number of distinct concept labels the meta-classifier discriminates;
"baseline" is the random-guess accuracy `1 / n_concepts`.

---

### `data/real/annotated_streams/` - Experiment 3 (SEA/STAGGER/LED)

`river.datasets.synth`. 200 chunks $\times$ 200 = 40000 instances each. Sudden drifts at chunks [50, 100, 150]; gradual at [48, 100, 153] with a 5-chunk sigmoid transition. Sequential concepts.

| Stream | Features | Chunks@200 | Concepts | Baseline | Source |
|---|---|---|---|---|---|
| INSECTS-abrupt_balanced | 33 | 264 | 6 | 0.167 | Table 2, Souza et al. (2020) |
| INSECTS-abrupt_imbalanced | 33 | 1,776 | 6 | 0.167 | Table 2, Souza et al. (2020) |
| INSECTS-incgradual_balanced | 33 | 120 | 2 | 0.500 | Table 2, Souza et al. (2020) |
| INSECTS-incgradual_imbalanced | 33 | 716 | 2 | 0.500 | Table 2, Souza et al. (2020) |

Drift chunk indices in `data/real/annotated_streams_gt/` are computed
directly from Table 2 of Souza et al. (2020) by dividing the reported
instance number by chunk_size=200.

### `data/synthetic/streams/` (Experiment ...)

Format: `(n_instances, n_features + 1)`, last column = the original
class label, re-indexed from 0 after sorting instances into
contiguous class blocks.

| Stream | Features | Concepts | Source |
|---|---|---|---|
| XXX | ... | ... | ...  |
| XXX | ... | ... | ...  |

Drift chunk indices in `data/synthetic/streams_gt/` are computed
at generation time from the sorted block boundaries — they are not
estimates of any real, natural drift in the data; they exist because I
constructed them.

---

## Stream Scripts

### `streams/generate_real_streams.py`
Converts all annotated stream CSV files from the USP DS Repository into
`.npy` format. Applies label encoding and min-max normalisation. Saves
the ground truth drift chunk indices. Run once after downloading the
USP ZIP.

### `streams/generate_synthetic_streams.py`
... ADD HERE ...

---

## ABFS Meta-Feature Versions

| Version | Name | Dim | Description |
|---|---|---|---|
| v1.1 | aggstats | 8 | entropy, n_relevant, max_score, std_score, delta_mean, n_changed, drift_count, time_since_drift |
| v2.0 | raw scores | n_features | Normalized relevance score vector — feature identity preserved |
| v2.1 | raw + temporal | n_features + 2 | v2.0 + delta_mean + cosine_sim (window-to-window change) |

---

## Evaluation Protocol

All experiments use **prequential (test-then-train)** evaluation exclusively.
Batch CV (Experiments 1a/1b) is kept for historical comparison only.

**chunk_size = 200** throughout all experiments — consistent with
Experiments 1a–1c and the Experiment 2 baseline. Fixed to isolate the
effect of stream type and meta-feature version from chunk size effects,
which are characterised separately in Experiment 2.

**Classifiers:** River GNB, KNN, HT + sklearn MLP.

---

## Execution Order

### Experiment 0: Pipeline Verification

**Purpose:** before testing anything new, confirm the existing pipeline
(ABFS implementation, Komorniczak feature extraction, classifier sweep)
reproduces a known published result. If this doesn't match, nothing
downstream can be trusted.

**1.** `external/komorniczak/E1_extract_synthetic.py`
→ `external/komorniczak/results/synthetic/`

**2.** `external/komorniczak/E2_clf_synthetic.py`
Compare against Figure 12 of Komorniczak et al.

**3.** `experiments/experiment_0/comparison.py`
→ `results/experiment_0/figures/`

**4.** `experiments/experiment_0/replication_check_1a.py`

---

### Experiments 1a & 1b: Batch CV (Historical)

**Purpose:** first real test of ABFS as a meta-feature — does it
discriminate concepts at all, on the simplest possible synthetic
streams (SEA, STAGGER), using static cross-validation? 1b adds shuffled
vs. unshuffled splits to check whether temporal order matters when the
concepts don't recur — it shouldn't, and confirming that rules out a
class of evaluation artifacts before moving to harder streams.


**5.** `experiments/experiment_1a/evaluate_concept_classification_1a.py`

**6.** `experiments/analysis_1a_1b.py --exp 1a --sanity --variance --shap --metrics`

**7.** `experiments/experiment_1b/komor_concept_classification_1b.py`

**8.** `experiments/experiment_1b/evaluate_concept_classification_1b.py`

**9.** `experiments/analysis_1a_1b.py --exp 1b --sanity --variance --shap --metrics`

---

### Experiment 1c: Prequential Evaluation

**Purpose:** batch CV (1a/1b) lets the classifier see the whole stream
at once, which isn't how a deployed system would actually operate.
Switch to prequential (test-then-train, one chunk at a time) to check
whether ABFS's batch-CV performance holds up under realistic streaming
conditions, and establish prequential as the protocol for everything
after this point.

If you didn't run experiment 0, run E1_extract_synthetic.py script.

**10.** `experiments/experiment_1c/komor_concept_classification_1c.py`

**11.** `experiments/experiment_1c/evaluate_concept_classification_1c.py`

**12.** `experiments/experiment_1c/analysis_1c.py --sanity --performance --shap --metrics`

---

### Experiment 2: Stream Configuration Sensitivity

**Purpose:** 1a-1c used one fixed stream configuration. Here I map out
*when* ABFS wins or loses against Komorniczak as stream parameters
change — chunk size and the number of informative features — across
both sudden and gradual drift, while everything else (ground truth,
classifiers, evaluation protocol) stays controlled and known. This is
the main controlled experiment that characterizes ABFS's strengths and
weaknesses before testing it on real data.

chunk_size ∈ {100, 200, 500, 1000} × n_informative ∈ {3, 5, 10, 15}
4×4 grid × 2 drift types × 5 replications. **Prequential only.**

**13.** `experiments/experiment_2/evaluate_concept_classification_2.py`

Output naming:
```
preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy   (n_reps, n_windows, n_clfs)
preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy  (n_reps, n_windows, n_clfs)
```

**14.** `experiments/experiment_2/analysis_2.py --sanity --performance --shap --metrics --grid`

Figures produced (selective — not all 32 cells):
- Sanity: baseline cell only (chunk_size=200, n_informative=10)
- Performance: baseline cell + representative cells at n_inf=3 and n_inf=15
- SHAP: baseline cell, all 3 ABFS versions × 4 classifiers
- Metrics: baseline cell only
- Grid: one gap heatmap per ABFS version (sudden + gradual, 3 each) and
  two BA vs n_informative sensitivity curves per drift type — at the
  chunk_size where ABFS gains the most over Komorniczak, and where it
  loses the most

Output files stored in: `results/experiment_2/figures/analysis/`

---

### Experiment 3: Real-World Stream Evaluation (Annotated)

**Purpose:** Experiments 1-2 are entirely synthetic — informative, but
they don't tell us whether ABFS works on real feature distributions.
Here I test the same question (does ABFS discriminate concepts better
than Komorniczak?) on real INSECTS data, the only real-world stream
family with a documented, citable drift ground truth. This tests whether the patterns observed under controlled synthetic conditions (Experiments 1–2) transfer to real-world feature distributions and imperfect concept definitions.

Streams:
- INSECTS-abrupt_balanced  
- INSECTS-abrupt_imbalanced  
- INSECTS-incgradual_balanced  
- INSECTS-incgradual_imbalanced  

These include **annotated drift points (ground truth)**.

**15.** `streams/generate_real_streams.py`
Download USP DS Repository ZIP first, then run once. Produces the 4
INSECTS streams with positional concept labels derived from Table 2
of Souza et al. (2020).

**16.** `experiments/experiment_3/evaluate_concept_classification_3.py`
Self-contained: extracts ABFS and Komorniczak features inline, caches
Komorniczak results to `external/komorniczak/results/real/`.

Output per stream:
```
concept_labels_{stream}.npy    (n_windows,): ground truth concept label
preq_abfs_{version}_ba_{stream}.npy   (n_windows, n_clfs)
preq_komor_{measure}_ba_{stream}.npy   (n_windows, n_clfs)
heatmap_comparison_..._{stream}.png   Komorniczak vs ABFS heatmap
```

**17.** `experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics`

Figures produced per stream:
- Sanity: relevance scores over time, meta-features per version (3 plots),
  PCA per version (3 plots)
- Performance: ABFS trajectory (3 versions stacked), Komorniczak trajectory
  (3×3 grid of measure groups)
- SHAP: 2×2 subplot (GNB, KNN, HT, MLP) per ABFS version (3 plots)
- Metrics: F1 heatmap, Kappa heatmap

Output files stored in: `results/experiment_3/figures/analysis/`

---

### Experiment 4: ...

**Purpose:** ...


---

## Result File Naming Conventions

### Experiment 2
```
preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy
preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy
```

| Field | Values |
|---|---|
| `version` | `aggstats` \| `raw` \| `raw_temporal` |
| `measure` | `clustering` \| `complexity` \| `concept` \| `general` \| `info-theory` \| `itemset` \| `landmarking` \| `model-based` \| `statistical` |
| `cs` | `100` \| `200` \| `500` \| `1000` |
| `ni` | `3` \| `5` \| `10` \| `15` |
| `drift` | `sudden` \| `gradual` |
| shape | `(n_reps, n_windows, n_clfs)` with `n_reps=5` |

### Experiment 3
```
preq_abfs_{version}_ba_{stream}.npy
preq_komor_{measure}_ba_{stream}.npy
  shape: (n_windows, n_clfs) — no replications (fixed real stream)
```

---

## Key Findings

| Finding | Result |
|---|---|
| Shuffling (1a vs 1b) | <0.002 BA — non-recurring concepts have no temporal structure |
| Raw vs aggstats | v2.0 >> v1.1 on synthetic; v1.1 competitive on high-dim real streams |
| Temporal features (v2.1) | No improvement — delta_mean and cosine_sim rank last in SHAP |
| Sudden drift (Exp 2) | ABFS competitive at high n_informative (crossover ≈ n_inf=10) |
| Gradual drift (Exp 2) | Komorniczak consistently better — adaptation lag compounds |
| Prequential protocol | Komorniczak structural advantage: no memory → instant adaptation |
| PAC classifier | Fails completely (BA ≈ 0.095) — excluded from all experiments |
| n_informative (Exp 2) | Key driver: ABFS rises, Komorniczak flat/falls as n_inf increases |
| Real streams drift type | Abrupt/gradual: ABFS competitive; incremental: near random baseline |
| Real streams dimensionality | v2.0 dominates low-dim synthetic; v1.1 more robust on high-dim real |
| Balanced vs imbalanced | To be determined from extended Experiment 3 results |