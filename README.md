# Meta-features Based on Relevance Scores

ABFS-based stream meta-features for concept classification in non-stationary data streams.

---

## What is being evaluated?

Each experiment evaluates whether meta-features computed per stream window can discriminate the active concept.

- Input: meta-feature vector per window
- Target: concept label (definition depends on stream type)
- Task: multi-class classification at the meta-level

High performance implies that the meta-features encode concept identity effectively.



## Project Structure
```
meta-features_based_on_relevance_scores/
│
├── abfs/
│   └── abfs_implementation.py
│
├── data/  # local data files - not committed to git                        
│   ├── real/
│   │   ├── annotated_streams/
│   │   │   ├── INSECTS-abrupt_balanced.npy
│   │   │   ├── INSECTS-abrupt_imbalanced.npy
│   │   │   ├── INSECTS-incgradual_balanced.npy
│   │   │   └── INSECTS-incgradual_imbalanced.npy
│   │   ├── annotated_streams_gt/
│   │   │   └── {same filenames}.npy
│   │   └── analysis/
│   │       └── {stream analysis files}.npy
│   ├── semi_synthetic/
│   │   ├── streams/
│   │   │   ├── electricity.npy
│   │   │   └── covtype.npy
│   │   ├── streams_gt/
│   │   │    └── {same filenames}.npy
│   │   └── analysis/
│   │       └── {stream analysis files}.npy
│   └── synthetic/
│       ...
│       # ADD HERE
│
├── experiments/
│   ├── experiment_0/
│   │   ├── comparison.py
│   │   └── replication_check_1a.py
│   ├── experiment_1a/
│   │   └── evaluate_concept_classification_1a.py
│   ├── experiment_1b/
│   │   ├── evaluate_concept_classification_1b.py
│   │   └── komor_concept_classification_1b.py
│   ├── experiment_1c/
│   │   ├── analysis_1c.py
│   │   ├── evaluate_concept_classification_1c.py
│   │   └── komor_concept_classification_1c.py
│   ├── experiment_2/
│   │   ├── analysis_2.py
│   │   └── evaluate_concept_classification_2.py
│   ├── experiment_3/
│   │   ├── analysis_3.py
│   │   └── evaluate_concept_classification_3.py
│   ├── experiment_4/
│   │   ├── analysis_4.py
│   │   └── evaluate_concept_classification_4.py
│   ├── analysis_1a_1b.py
│   ├── classifier_sweep_komor.py
│   └── classifier_sweep_prequential.py
│
├── external/
│   └── komorniczak/
│       ├── results/
│       │   ├── real/
│       │   └── synthetic/
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
│   └── sanity_check/
│       └── figures/
│
├── streams/
│   ├── generate_real_streams.py
│   ├── generate_semi_synthetic_streams.py
│   ├── synthetic_streams_generator.py # ADD IMPLEMENTATION
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

**Step 1 — Download the USP DS Repository:**

All stream files are sourced from the USP DS Repository
(Souza et al., 2020): https://sites.google.com/view/uspdsrepository

Download the full ZIP and unzip it:
```bash
pip install gdown
gdown "1JERZnbGGToAEz_3LRV7n2Vz79LiDAEY-" -O ~/usp_ds_repository.zip
unzip ~/usp_ds_repository.zip -d ~/usp_ds_repository/
```

**Step 2 — Run the stream generation scripts:**
```bash
python streams/generate_real_streams.py             # Experiment 3: INSECTS
python streams/generate_semi_synthetic_streams.py    # Experiment 4: electricity, covtype
```

`generate_real_streams.py` converts the INSECTS CSV files into the
`.npy` format required by the pipeline, using the genuinely documented
drift change points from Table 2 of Souza et al. (2020).

`generate_semi_synthetic_streams.py` builds electricity and covtype
streams with artificially injected drift, since neither dataset has a
published natural drift ground truth (see the Tier 3 explanation
below).

---

### Three tiers of evaluation

The experiments operate on three types of streams, differing in how concept drift is defined and how reliable the ground truth is.

---

### Tier 1 — Synthetic streams (Experiments 1 and 2)

Concepts and drift are fully controlled by the stream generator.

- Concept = underlying feature–label mapping
- Drift = change in this mapping

Two drift types:
- Sudden: instantaneous change → clean segments
- Gradual: sigmoid transition → mixed windows labeled by stage

Both **where drift occurs** and **what changes** are known exactly.

Purpose: controlled evaluation of meta-feature discriminative power

---

### Tier 2 — Real annotated streams (Experiment 3)

Real-world INSECTS data with documented drift locations (Souza et al., 2020).

- Concept = positional segment between annotated drift points
- Drift = boundary between segments

Important:
- Drift location is known.
- Concept identity is unknown.  

Concept labels are **purely positional**: concept = number of drift boundaries crossed. Concept segments are treated as distinct even if their underlying distributions are similar; no recurring concept structure is assumed or inferred.

This means:
- no assumption of recurring concepts
- segments may be statistically similar but still treated as different concepts

Purpose: test robustness on real data with imperfect ground truth

---

### Tier 3 — Semi-synthetic streams (Experiment 4)

Real datasets (electricity, covtype) with **injected drift**.

- Concept = class-dominated block (after sorting by class)
- Drift = transition between class blocks

This guarantees:
- drift locations known. 
- concept identity known.  
- real feature distributions.  

This does NOT claim natural drift — drift is constructed.

Purpose: controlled concept identity on real feature distributions

---

### Summary

| Tier | Drift location | Concept identity | Data type |
|------|--------------|------------------|----------|
| Synthetic | YES | YES | synthetic |
| Real annotated | YES | NO (positional) | real |
| Semi-synthetic | YES | YES (constructed) | real features |

---

These three tiers progressively test:

1. Controlled conditions (synthetic)
2. Real-world uncertainty (INSECTS)
3. Controlled concepts on real distributions (semi-synthetic)


---

### `data/real/annotated_streams/` (Experiment 3)

Format: `(n_instances, n_features + 1)`, last column = positional
concept label (0-indexed) — see the caveat above about what this
label does and does not mean.

| Stream | Features | Chunks@200 | Concepts | Baseline | Source |
|---|---|---|---|---|---|
| INSECTS-abrupt_balanced | 33 | 264 | 6 | 0.167 | Table 2, Souza et al. (2020) |
| INSECTS-abrupt_imbalanced | 33 | 1,776 | 6 | 0.167 | Table 2, Souza et al. (2020) |
| INSECTS-incgradual_balanced | 33 | 120 | 2 | 0.500 | Table 2, Souza et al. (2020) |
| INSECTS-incgradual_imbalanced | 33 | 716 | 2 | 0.500 | Table 2, Souza et al. (2020) |

Drift chunk indices in `data/real/annotated_streams_gt/` are computed
directly from Table 2 of Souza et al. (2020) by dividing the reported
instance number by chunk_size=200.

### `data/semi_synthetic/streams/` (Experiment 4)

Kept in a separate top-level folder from `data/real/`, since these
streams are not real annotated data — they are real feature
distributions with artificially injected drift, which is a different
kind of evidence and shouldn't sit alongside genuinely annotated data.

Format: `(n_instances, n_features + 1)`, last column = the original
class label, re-indexed from 0 after sorting instances into
contiguous class blocks.

| Stream | Features | Concepts | Source |
|---|---|---|---|
| electricity | 8 | 2 | injected (sorted by class) |
| covtype | 54 | 7 | injected (sorted by class, all 7 original classes) |

Drift chunk indices in `data/semi_synthetic/streams_gt/` are computed
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

### `streams/synthetic_streams.py`
Shared utilities for Experiments 1 and 2: concept label assignment
(sudden and gradual drift), ABFS meta-feature extraction (all 3 versions
in a single pass), Komorniczak pymfe extraction.

### `streams/generators.py`
Original synthetic stream helpers.

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

### Experiment 4: Semi-Synthetic Stream Evaluation (Injected Drift)

**Purpose:** Experiment 3 is limited to one stream family (INSECTS).
To check whether the findings generalize to other kinds of real
feature distributions, we use electricity and covtype, but since
neither has a published natural drift locationdrift is injected in a controlled manner (sorting by class into contiguous blocks), so the ground
truth is fully known and the comparison stays fair. This widens the
real-data evidence beyond INSECTS without fabricating a claim about
natural drift we can't verify.

Streams:
- electricity  
- covtype  

These datasets are augmented with:
- injected drift boundaries  
- controlled class/concept changes  

**18.** `streams/generate_semi_synthetic_streams.py`
Produces electricity and covtype streams with artificially injected
drift (instances sorted by class label into contiguous blocks).
covtype uses all 7 original classes.

**19.** `experiments/experiment_4/evaluate_concept_classification_4.py`
Same machinery as Experiment 3 — extracts ABFS and Komorniczak
features inline, evaluates prequentially.

Output per stream:
```
concept_labels_{stream}.npy             (n_windows,)  ← injected concept label
preq_abfs_{version}_ba_{stream}.npy      (n_windows, n_clfs)
preq_komor_{measure}_ba_{stream}.npy     (n_windows, n_clfs)
heatmap_comparison_..._{stream}.png      ← Komorniczak vs ABFS heatmap
```

**20.** `experiments/experiment_4/analysis_4.py --sanity --performance --shap --metrics`

Same figure set as Experiment 3, applied to electricity and covtype.

Output files stored in: `results/experiment_4/figures/analysis/`

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