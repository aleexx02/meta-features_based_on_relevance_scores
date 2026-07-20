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

### A note on numbering

The code directories keep their historical names (`experiment_1a/`, `experiment_1c/`).
The **report** drops the batch-CV experiment (1a) and renumbers, so:

| Report | Code directory | Protocol |
|---|---|---|
| Experiment 0 | `experiment_0/` | Komorniczak replication |
| Experiment 1 | `experiment_1c/` | prequential |
| Experiment 2 | `experiment_2/` | prequential |
| Experiment 3 | `experiment_3/` | prequential |
| Experiment 4 | `experiment_4/` | prequential |
| Experiment 5 | `experiment_5/` | prequential |

`experiment_1a/` is retained in the repository for provenance but is not part of the reported results.


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
│       │   └── {same filenames}.npy      # drift CHUNK indices (ground truth)
│       └── annotated_streams_analysis/
│           └── {stream analysis files}.npy
│
├── experiments/
│   ├── experiment_0/
│   │   ├── comparison.py
│   │   └── replication_check_1a.py
│   │
│   ├── experiment_1a/                    # legacy, not in the report
│   │   ├── analysis_1a.py
│   │   └── evaluate_concept_classification_1a.py
│   │
│   ├── experiment_1c/                    # = report Experiment 1
│   │   ├── analysis_1c.py
│   │   ├── evaluate_concept_classification_1c.py
│   │   ├── komor_concept_classification_1c.py
│   │   └── vanilla_cost_1c.py
│   │
│   ├── experiment_2/
│   │   ├── analysis_2.py
│   │   ├── evaluate_concept_classification_2.py
│   │   └── vanilla_cost_2.py
│   │
│   ├── experiment_3/ # SEA, STAGGER, LED (sequential)
│   │   ├── analysis_3.py
│   │   ├── evaluate_concept_classification_3.py
│   │   └── vanilla_cost_3.py
│   │
│   ├── experiment_4/ # recurring concepts (SEA, STAGGER)
│   │   ├── analysis_4.py
│   │   ├── evaluate_concept_classification_4.py
│   │   └── vanilla_cost_4.py
│   │
│   ├── experiment_5/ # Real Streams: INSECTS + SPAM
│   │   ├── analysis_5.py
│   │   ├── evaluate_concept_classification_5.py
│   │   └── vanilla_cost_5.py
│   │
│   ├── characterize_streams.py           # stream characterisation figures + table
│   ├── feature_ranges.py                 # raw feature scale + sparsity per stream
│   ├── classifier_sweep_komor.py
│   ├── classifier_sweep_prequential.py
│   └── vanilla_and_cost.py               # shared vanilla-baseline + cost engine
│
├── external/
│   └── komorniczak/
│       ├── results/
│       │   ├── real/ # Exp 5 pymfe cache
│       │   ├── synthetic/ # Exp 0 cache
│       │   ├── synthetic_sea_stagger_led/ # Exp 3 pymfe cache
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
│   ├── experiment_{0,1a,1c,2,3,4,5}/
│   │   ├── *.npy                         # evaluation results
│   │   ├── *.csv                         # summary / vanilla / cost / concept-distance
│   │   └── figures/
│   │       ├── analysis/                 # sanity, SHAP, trajectories, gaps, metrics
│   │       └── streams/                  # stream characterisation (fingerprints etc.)
│   ├── concept_separation_all_generators.csv
│   ├── feature_ranges_all_experiments.csv
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

This writes both the stream arrays (`annotated_streams/`) and the drift ground
truth (`annotated_streams_gt/`, drift **chunk** indices). Several downstream
tools read the ground-truth files rather than hardcoding boundaries, so this
step must run before Experiment 5 or `characterize_streams.py`.

**There is no generation step for Experiments 1c, 2, 3 and 4.** Their streams are built on the fly per grid cell inside the evaluate scripts, exactly like Experiment 2's StreamGenerator streams. `generate_synthetic_streams.py` is a builder *library*: it exposes Exp 2, 3 and 4 helpers, and writes nothing to disk. Running it directly just prints the grid layout and smoke-tests the builders.

---

### Two tiers of evaluation

The experiments operate on two types of streams, differing in how concept drift is defined and how reliable the ground truth is.


| Tier | Drift location | Concept identity | Data type | Experiments |
|------|--------------|------------------|----------|----------|
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

### Experiments 1c & 2 - `strlearn` StreamGenerator

Binary-target streams drift via concept switching. Concept label is multi-class even though the target is binary.

Experiment 1c uses StreamGenerator:

| | n_features | n_informative | n_rep | Target classes | n_drifts | n_concepts | chunk_size |
|---|---|---|---|---|---|---|---|
| Exp 1c sudden | 10 | 10 | 5 | 2 | 20 | 21 | 200 |
| Exp 1c gradual | 10 | 10 | 5 | 2 | 6 | 25 | 200 |

Experiment 2 also uses StreamGenerator:

| | n_features | n_informative | n_rep | Target classes | n_drifts | n_concepts | chunk_size |
|---|---|---|---|---|---|---|---|
| Exp 2 sudden | 20 | {3,5,10,15} | 5| 2 | 20 | 21 | {100,200,500,1000} |
| Exp 2 gradual | 20 | {3,5,10,15} | 5 | 2 | 6 | 25 | {100,200,500,1000} |

Under gradual drift the sigmoid transition zones are assigned their own
concept labels, which is why 6 drifts yield 25 concepts rather than 7.


### Experiment 3 - SEA / STAGGER / LED (sequential, chunk_size sweep)

`river.datasets.synth`. **500,000 instances** per stream. The one swept axis is **chunk_size $\in$ {100, 200, 500}** (5000 / 2500 / 1000 chunks respectively). Each generator runs its fixed concept order once; 3 drifts placed evenly. **Grid = 3 generators $\times$ 2 drift types $\times$ 3 chunk_sizes = 18 cells.**

| Generator | Features | Target classes | Concept order | Concepts | Baseline |
|---|---|---|---|---|---|
| SEA | 3 (2 relevant) | 2 (binary) | [0,1,2,3] | 4 | 0.250 |
| STAGGER | 3 (size/color/shape) | 2 (binary) | [0,1,2,0] | 3 | 0.333 |
| LED | 24 (7 relevant + 17 noise) | 10 (digit) | [0,1,2,3] | 4 | 0.250 |


### Experiment 4 - recurring SEA / STAGGER (chunk_size $\times$ n_drifts grid)

`river.datasets.synth`. **500,000 instances** per stream. Full grid:
**chunk_size $\in$ {100,200,500} $\times$ n_drifts $\in$ {1,3,7,15}**, $\times$ 2 generators $\times$ 2 drift types = **48 cells.** Concepts **cycle** through the
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
| INSECTS-incgradual_balanced | 33 | 6 (species) | 120 | 2 | 0.500 | exact |
| INSECTS-incgradual_imbalanced | 33 | 6 (species) | 716 | 2 | 0.500 | exact |
| SPAM | 499 | 2 (spam/legit) | ~93 | 6 | 0.167 | approximate |

Real-stream features are min-max normalised to `[0, 1]` at generation time,
so concept distances measured on them are directly comparable across streams
(unlike the synthetic generators, whose feature scales differ).

---

## ABFS Meta-Feature Versions

| Version | Name | Dim | Description |
|---|---|---|---|
| v1.1 | aggstats | 8 | entropy, n_relevant, max_score, std_score, delta_mean, n_changed, drift_count, time_since_drift |
| v2.0 | raw scores | n_features | Normalized relevance score vector — feature identity preserved |
| v2.1 | raw + temporal | n_features + 2 | v2.0 + delta_mean + cosine_sim |

Note that the aggregate summary has a **fixed** dimension of 8 whatever the
stream, whereas the two raw variants have one component per stream feature: 10
(Exp 1), 20 (Exp 2), 3 (SEA/STAGGER), 24 (LED), 33 (INSECTS), 499 (SPAM). How
the two families scale with dimensionality is itself one of the findings.

---

## Evaluation Protocol

Prequential (test-then-train) throughout (batch CV in the legacy 1a). Classifiers: River GNB, KNN, HT + sklearn MLP.

**Replications.** Experiments 1c, 2, 3, 4 regenerate each stream/cell from several seeds and stack into `(n_reps, n_windows, n_clfs)` (`n_reps = 5`). Experiment 5 has no replication axis: real streams are fixed, `(n_windows, n_clfs)`. For Exp 3/4 the seed propagates into the river generators, so the replications are genuinely different realizations; replication 0 uses `SEED`.

**ABFS window size** ties to each cell's chunk_size.

---

## Baselines, cost, and stream characterisation

Three cross-cutting components sit alongside the per-experiment evaluation.
Each answers a different question, and none replaces another.

### 1. Vanilla baseline and extraction cost — `experiments/vanilla_and_cost.py`

A shared engine used by every experiment through its own
`experiment_N/vanilla_cost_N.py` adapter. It adds two things without re-running
the expensive ABFS / Komorniczak extraction:

- **Vanilla baseline** — the simplest possible meta-features: the per-feature
  mean and standard deviation of each raw window, pushed through *exactly* the
  same prequential sweep, windowing and classifiers as the real meta-features.
  It never sees a class label. It is a diagnostic, not just a floor: if it wins,
  the concept is visible in the raw feature values; if it fails where the
  meta-features succeed, the concept lives in the feature-class relationship.
  Saved as `preq_vanilla_{ba,f1,kappa}_{cell}.npy`.
- **Extraction cost** — wall-clock time per window and peak Python-heap memory
  for ABFS vs Komorniczak extraction, on representative cells spanning the
  feature counts in the streams. Saved to `extraction_cost_exp{N}.csv`.

  **Caveat:** cost is only meaningful for Experiments 3, 4 and 5. The
  `stream-learn` generators (Exp 1c, 2) produce data lazily *inside* the timed
  loop, so generation and extraction cannot be separated there and the numbers
  are not reported.

Each adapter is run with:
```bash
python experiments/experiment_3/vanilla_cost_3.py
# inside: run_experiment(spec, do_vanilla=True, do_cost=True)
```

### 2. Raw feature scale and sparsity — `experiments/feature_ranges.py`

Measures the raw feature values of every stream **configuration**: min, max,
mean, standard deviation and the fraction of non-zero entries. This documents
the scale each generator operates on, which is what makes the concept distances
below interpretable (a distance of 2 means different things on SEA's `[0,10]`
features than on LED's `[0,1]` ones), and it supplies the sparsity figure used
in the SPAM dimensionality-vs-sparsity discussion.

Coverage note: the feature scale depends on the generator and, for
`stream-learn`, on `n_informative`. It does **not** depend on `chunk_size`,
drift type or recurrence, which only regroup or reschedule the same instances.
So Exp 2 is swept over `n_informative` $\in$ {3,5,10,15}, and Exp 3's three
generator rows also characterise Exp 4.

```bash
python experiments/feature_ranges.py     # run on the cluster (needs data/real/)
# -> results/feature_ranges_all_experiments.csv
```

### 3. Concept separation — `--concept_dist_features` + `characterize_streams.py`

These two work as a pair: the analysis flag **measures**, the script
**visualises and consolidates**.

**`--concept_dist_features`** (a flag on `analysis_1c.py`, `analysis_2.py`,
`analysis_3.py`, `analysis_4.py`) groups windows by concept, takes each
concept's mean feature vector, and measures the pairwise Euclidean distance
between those vectors. That distance says how far apart the concepts sit in
feature space — i.e. how much of the concept is readable from the raw data
without ever consulting a label. Distances are computed **per replication and
then averaged**, since for generators whose concepts share identical marginals
(SEA, STAGGER) a single realisation measures only sampling noise.

*Why the fingerprints are not averaged across replications:* each
`stream-learn` seed draws a different Madelon layout, so concept *k* in one
replication is an unrelated cluster to concept *k* in another. Averaging their
means would wash out the structure. The fingerprint CSV therefore records
replication 0; only the distances are averaged.

Experiment 5 needs no such flag — real streams have no generator to rebuild
from, so `characterize_streams.py` computes their concept means directly from
the saved arrays plus the drift ground truth.

**`characterize_streams.py`** reads those CSVs and produces, per cell:

| Output | What it answers |
|---|---|
| `fingerprint_{exp}_{cell}.png` | *Does the data move?* Concepts as rows, features as columns, colour = mean value. Distinct patterning per row means each concept displaces a different feature subset; flat identical rows mean the data does not move and only the labelling rule changes. |
| `distance_{exp}_{cell}.png` | *How far apart do they end up?* Pairwise L2 between concept mean-vectors. Bright = separable from raw values alone; near-zero = concepts sit on top of each other and the difference lives only in P(y\|X). |
| `spread_{exp}_{series}_{sweep}.png` | *How does separation grow with a swept parameter?* Drawn only where a numeric sweep exists (Exp 2's `n_informative`): mean pairwise L2 with a min-max band. |
| `concept_separation_all_generators.csv` | One row per cell across every experiment — feature count, concept count, L2 min/max/mean, plus a "data moves / moves partially / does not move" verdict. This is the cross-generator separation table. |

```bash
python experiments/characterize_streams.py
```
No flags needed: it auto-discovers `concept_feature_means_exp*.csv` under
`results/experiment_*/`, draws **every** cell, and rebuilds the Experiment 5
CSVs from `data/real/` first (skipping that step quietly if the real streams
are not present). Useful options: `--representative` (one cell per experiment
instead of all), `--no-exp5`, `--figdir DIR` (all figures in one place).

Figures are written to each experiment's own
`results/experiment_{N}/figures/streams/`.

**Rerun hygiene:** every CSV above is opened in `'w'` mode and self-overwrites,
so nothing needs deleting. The figures do not self-clean, though, so if the set
of cells changes, clear `results/experiment_*/figures/streams/` first to avoid
stale plots. Never delete the `.npy` results or `figures/analysis/`.

---

## Pipeline overview

The pipeline is organised so that the **baseline (Komorniczak) side is
reproduced and verified first**, and only then are our ABFS meta-features
evaluated against it under the same protocol. Each experiment has an
*evaluate* stage (produces `.npy` results) and an *analysis* stage
(produces figures and tables).

---

## Execution Order

### Experiment 0: Pipeline Verification
Reproduces and checks the original Komorniczak pipeline before building on it. Produces no thesis results; it only validates the setup.

**1.** `external/komorniczak/E1_extract_synthetic.py`
**2.** `external/komorniczak/E2_clf_synthetic.py`
**3.** `experiments/experiment_0/comparison.py`
**4.** `experiments/experiment_0/replication_check_1a.py`

| # | Script | Purpose |
|---|--------|---------|
| 1 | `external/komorniczak/E1_extract_synthetic.py` | Run their original pipeline: generate their synthetic streams and extract the 9 groups of statistical (pymfe) meta-features. Saved to `external/komorniczak/results/synthetic/`. Source for step 7. |
| 2 | `external/komorniczak/E2_clf_synthetic.py` | Run their original classification (their CV protocol) on those meta-features, reproducing their published results. |
| 3 | `experiments/experiment_0/comparison.py` | Compare our output against theirs to confirm the reproduction matches. |
| 4 | `experiments/experiment_0/replication_check_1a.py` | Explicit replication check against their reference numbers (e.g. Figure 12 benchmark). |


### Experiment 1a: Batch CV *(legacy — not in the report)*
Kept for provenance. The reported results begin at the prequential protocol.

**5.** `experiments/experiment_1a/evaluate_concept_classification_1a.py`
**6.** `experiments/experiment_1a/analysis_1a.py --sanity --variance --shap --metrics`


### Experiment 1 (code: `experiment_1c/`): Prequential
Prequential (test-then-train) evaluation, the definitive protocol. Order matters: the Komorniczak side (7) must exist before our side (8), and the analysis (9) needs both.

**7.** `experiments/experiment_1c/komor_concept_classification_1c.py`
**8.** `experiments/experiment_1c/evaluate_concept_classification_1c.py`
**9.** `experiments/experiment_1c/analysis_1c.py --sanity --performance --shap --metrics --stream_analysis --gap --bars --concept_dist --vanilla --summary --concept_dist_features`
**9b.** `experiments/experiment_1c/vanilla_cost_1c.py`

| # | Script | Purpose |
|---|--------|---------|
| 7 | `komor_concept_classification_1c.py` | **Baseline side.** Load Komorniczak's pre-extracted meta-features (from step 1) and evaluate them under our prequential protocol, skipping the first 10 warm-up windows to align with ABFS. Saves `clf_komor_concept_classif_{ba,f1,kappa}_{drift}.npy`. Does *not* generate streams. |
| 8 | `evaluate_concept_classification_1c.py` | **ABFS side.** Extract our 3 meta-feature versions (aggstats / raw / raw+temporal) from the streams and evaluate them with the same prequential protocol and warm-up. Saves `clf_ba_{version}_{drift}.npy`. |
| 9 | `analysis_1c.py` | Combine both sides and produce the material: performance per classifier, relevance-score stream analysis, gap heatmaps (best ABFS − best Komorniczak), and the summary / vanilla / concept-distance CSVs. |
| 9b | `vanilla_cost_1c.py` | Vanilla-baseline trajectories (and the cost timing, which is not reported for this experiment — see the cost caveat above). |

**Note on balanced accuracy.** Both sides report the **final cumulative BA at
the last window, averaged over the 5 replications**. An earlier version of
`evaluate_concept_classification_1c.py` averaged the Komorniczak BA over *all*
windows instead, which inflated it (0.930 vs 0.896 for statistical/HT under
sudden drift) and made the comparison figures disagree with the summary tables.
Any new comparison figure must use the final-window definition.


### Experiment 2: Stream Configuration Sensitivity
chunk_size $\in$ {100,200,500,1000} $\times$ n_informative $\in$ {3,5,10,15}, $\times$ 2 drift $\times$ 5 reps.

**10.** `experiments/experiment_2/evaluate_concept_classification_2.py`
**11.** `experiments/experiment_2/analysis_2.py --sanity --performance --shap --metrics --grid --stream_analysis --vanilla --summary --concept_dist_features`
**11b.** `experiments/experiment_2/vanilla_cost_2.py`

| # | Script | Purpose |
|---|--------|---------|
| 10 | `evaluate_concept_classification_2.py` | Evaluate ABFS (3 versions) and Komorniczak (9 measures) across the chunk_size $\times$ n_informative grid, both drift types, 5 reps. |
| 11 | `analysis_2.py` | `--grid` produces the gap heatmaps (per ABFS version) and the chunk_size / n_informative sensitivity curves. `--concept_dist_features` sweeps `n_informative` and records the `n_informative` column that drives the spread curve. |


### Experiment 3: SEA / STAGGER / LED (chunk_size sweep)
**Purpose:** how does window size (chunk_size) affect ABFS vs Komorniczak on three classic generators, at 500k instances?

**12.** `experiments/experiment_3/evaluate_concept_classification_3.py`
**13.** `experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics --stream_analysis --gap --grid --vanilla --summary --concept_dist_features`
**13b.** `experiments/experiment_3/vanilla_cost_3.py`

| # | Script | Purpose |
|---|--------|---------|
| 12 | `evaluate_concept_classification_3.py` | Regenerates each cell (generator $\times$ drift $\times$ chunk_size) per replication seed; ABFS (3 versions) + Komorniczak (9 measures) inline; caches pymfe to `external/komorniczak/results/synthetic_sea_stagger_led/`. |
| 13 | `analysis_3.py` | `--gap` gap heatmaps per ABFS version (generator $\times$ chunk_size); `--grid` BA-vs-chunk_size curves per classifier. Figures in `results/experiment_3/figures/analysis/`. |

Output per cell (shape `(n_reps, n_windows, n_clfs)`):

```
preq_abfs_{version}_ba_{gen}_chunk{cs}_{drift}.npy   (+ _f1_, _kappa_)
preq_komor_{measure}_ba_{gen}_chunk{cs}_{drift}.npy  (+ _f1_, _kappa_)
concept_labels_{gen}_chunk{cs}_{drift}.npy
heatmap_comparison_komorniczak_ABFS_preq_exp3_{gen}_chunk{cs}_{drift}.png
```


### Experiment 4: Recurring concepts (chunk_size $\times$ n_drifts grid)
**Purpose:** concepts recur; characterize ABFS vs Komorniczak across window size and drift frequency / recurrence amount.

**14.** `experiments/experiment_4/evaluate_concept_classification_4.py`
**15.** `experiments/experiment_4/analysis_4.py --sanity --performance --shap --metrics --stream_analysis --gap --grid --vanilla --summary --concept_dist_features`
**15b.** `experiments/experiment_4/vanilla_cost_4.py`

| # | Script | Purpose |
|---|--------|---------|
| 14 | `evaluate_concept_classification_4.py` | Grid is generator $\times$ drift $\times$ chunk_size $\times$ n_drifts (48 cells); pymfe cache in `external/komorniczak/results/synthetic_recurring/`. Large grid: 48 cells $\times$ 5 reps $\times$ 12 feature sets (3 ABFS versions + 9 Komorniczak measures). |
| 15 | `analysis_4.py` | `--gap` chunk_size $\times$ n_drifts gap heatmaps per ABFS version (per generator $\times$ drift); `--grid` BA-vs-n_drifts curves per classifier at each chunk_size. |


Output per cell (shape `(n_reps, n_windows, n_clfs)`):
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy   (+ _f1_, _kappa_)
preq_komor_{measure}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy  (+ _f1_, _kappa_)
concept_labels_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
heatmap_comparison_komorniczak_ABFS_preq_exp4_{gen}_chunk{cs}_ndrift{nd}_{drift}.png
```


### Experiment 5: real streams (INSECTS + SPAM)
**Purpose:** test on real, externally annotated streams where drift is not engineered.

**16.** `streams/generate_real_streams.py`
**17.** `experiments/experiment_5/evaluate_concept_classification_5.py`
**18.** `experiments/experiment_5/analysis_5.py --sanity --performance --shap --metrics --stream_analysis --gap --bars --vanilla --summary`
**18b.** `experiments/experiment_5/vanilla_cost_5.py`

| # | Script | Purpose |
|---|--------|---------|
| 16 | `generate_real_streams.py` | Build the real streams (INSECTS variants + SPAM) into the format used by the pipeline, plus the drift ground truth. |
| 17 | `evaluate_concept_classification_5.py` | Iterates `REAL_STREAMS`; ABFS (3 versions) + Komorniczak (9 measures); pymfe cache in `external/komorniczak/results/real/`. |
| 18 | `analysis_5.py` | SPAM (499 feat) caps per-feature plots to the top 20 by relevance-score variance (PCA never capped). |
| 18b | `vanilla_cost_5.py` | Vanilla baseline **and** the decisive cost measurement: pass `cost_cells=CELLS` so all five streams are timed, including SPAM at 499 features — the scaling point of the cost comparison. |

Experiment 5 has **no** `--concept_dist_features` flag: `characterize_streams.py`
computes its concept means from the saved arrays and ground truth instead.

Output per stream (shape `(n_windows, n_clfs)` - no rep axis):
```
preq_abfs_{version}_ba_{stream}.npy   (+ _f1_, _kappa_)
preq_komor_{measure}_ba_{stream}.npy  (+ _f1_, _kappa_)
concept_labels_{stream}.npy
heatmap_comparison_komorniczak_ABFS_preq_exp5_{stream}.png
```

### Cross-cutting (after all experiments)

**19.** `python experiments/feature_ranges.py`
**20.** `python experiments/characterize_streams.py`

---

## Notes on fairness
- Both sides of Experiment 1 use the **same prequential protocol** and the **same 10-window warm-up**; only the meta-features differ.
- The legacy Experiment 1a is batch CV; Experiment 1 is prequential: their numbers are **not** directly comparable.
- Komorniczak's meta-features are extracted externally (step 1) and only *re-evaluated* here; ours are extracted inside the pipeline (step 8).
- The vanilla baseline uses the same windowing, protocol and classifiers as both meta-feature families; only the meta-features differ.

---

## Result File Naming Conventions

### Experiment 1c
```
clf_ba_{version}_{drift}.npy                       shape: (n_reps, n_windows, n_clfs)
clf_komor_concept_classif_{ba,f1,kappa}_{drift}.npy   shape: (n_measures, n_reps, n_windows, n_clfs)
preq_vanilla_{ba,f1,kappa}_{drift}.npy             shape: (n_reps, n_windows, n_clfs)
  drift $\in$ {sudden, gradual}
```

### Experiment 2
```
preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy
preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy
preq_vanilla_ba_chunk{cs}_ninf{ni}_{drift}.npy
  shape: (n_reps, n_windows, n_clfs), n_reps=5
  cs    $\in$ {100, 200, 500, 1000}
  ni    $\in$ {3, 5, 10, 15}
  drift $\in$ {sudden, gradual}
```

### Experiment 3
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_{drift}.npy
preq_komor_{measure}_ba_{gen}_chunk{cs}_{drift}.npy
preq_vanilla_ba_{gen}_chunk{cs}_{drift}.npy
  shape: (n_reps, n_windows, n_clfs), n_reps=5
  gen   $\in$ {sea, stagger, led}
  cs    $\in$ {100, 200, 500}
  drift $\in$ {sudden, gradual}
```

### Experiment 4
```
preq_abfs_{version}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
preq_komor_{measure}_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
preq_vanilla_ba_{gen}_chunk{cs}_ndrift{nd}_{drift}.npy
  shape: (n_reps, n_windows, n_clfs), n_reps=5
  gen   $\in$ {sea, stagger}
  cs    $\in$ {100, 200, 500}
  nd    $\in$ {1, 3, 7, 15}
  drift $\in$ {sudden, gradual}
```

### Experiment 5
```
preq_abfs_{version}_ba_{stream}.npy
preq_komor_{measure}_ba_{stream}.npy
preq_vanilla_ba_{stream}.npy
  shape: (n_windows, n_clfs) — no replications
  stream $\in$ {INSECTS-abrupt_balanced, INSECTS-abrupt_imbalanced,
            INSECTS-incgradual_balanced, INSECTS-incgradual_imbalanced, SPAM}
```

Shared field values:
| Field | Values |
|---|---|
| `version` | `aggstats` \| `raw` \| `raw_temporal` |
| `measure` | `clustering` \| `complexity` \| `concept` \| `general` \| `info-theory` \| `itemset` \| `landmarking` \| `model-based` \| `statistical` |


---

## Output CSV files

Per experiment, in `results/experiment_{N}/`:

- **`summary_exp{N}.csv`** — the at-a-glance "who won each cell" table. One row per
  drift type (1c), grid cell (2–4), or stream (5). Columns: cell/drift/stream id,
  feature count, concept count, random baseline, best Komorniczak group + classifier
  and its balanced accuracy, best ABFS version + classifier and its balanced accuracy,
  and the ABFS−Komorniczak gap. Both sides use the final-window cumulative BA,
  averaged over replications.

- **`vanilla_comparison_exp{N}.csv`** — `cell, vanilla_ba, abfs_best_ba, komor_best_ba`.
  Final balanced accuracy of the vanilla baseline (per-window feature mean+std, no labels)
  against the best ABFS variant and best Komorniczak group, best classifier per side.
  Source for the vanilla-baseline tables.

- **`extraction_cost_exp{N}.csv`** — `experiment, tag, method, n_features, n_windows,
  time_s, ms_per_window, peak_mb`. Wall-clock time and peak Python-heap memory for ABFS
  vs Komorniczak extraction, one row per method per timed cell. Meaningful only for
  Exp 3/4/5; for 1c/2 the timing is contaminated by lazy stream generation and is not
  reported.

- **`concept_feature_means_exp{N}.csv`** — `cell, [n_informative,] concept, f0 … fK`.
  Each concept's average value for every raw feature: its fingerprint in feature space.
  Recorded from replication 0 (see the note on why fingerprints are not averaged).
  Files that mix generators of different dimensionality (Exp 3: SEA/STAGGER 3 features,
  LED 24) leave the surplus `f` columns empty on the low-dimensional rows.

- **`concept_distances_exp{N}.csv`** — `cell, [n_informative,] concept_a, concept_b,
  l2, l2_std, n_reps`. Pairwise Euclidean distance between every pair of concept
  mean-vectors, averaged across replications with its spread.

- **`concept_distance_summary_exp{N}.csv`** — `cell, [n_informative,] n_concepts,
  n_reps, l2_min, l2_max, l2_mean, l2_mean_std`. The compact per-cell version quoted
  in the text and consolidated into the cross-generator separation table.

Project-level, in `results/`:

- **`concept_separation_all_generators.csv`** (from `characterize_streams.py`) — one row
  per cell across **all** experiments: experiment, cell, n_features, n_concepts, L2
  min/max/mean, and a "data moves / moves partially / does not move" verdict. This is the
  cross-generator separation table, and the clearest single view of which streams encode
  their concept in the feature values and which only in the feature-class relationship.

- **`feature_ranges_all_experiments.csv`** (from `feature_ranges.py`) —
  `stream, n_features, min, max, mean, std, frac_nonzero` per stream configuration.
  Documents the scale each generator operates on (needed to interpret the concept
  distances) and the sparsity used in the SPAM dimensionality discussion. Stream labels
  deliberately contain no commas, and the file is written fully quoted, so naive
  comma-splitting importers cannot shift the columns.

- **`sparsity_analysis_exp5.csv`** — the SPAM sparsity check: how the relevance-score
  variation spreads across features (how many features account for 80% of it) on SPAM
  vs INSECTS. Confirms SPAM is genuinely high-dimensional rather than sparse-in-disguise.

---

## Figure directories

| Directory | Contents |
|---|---|
| `results/experiment_{N}/figures/analysis/` | Everything produced by `analysis_{N}.py`: relevance scores and meta-features over time, PCA projections, performance trajectories, SHAP importances, F1/Kappa heatmaps, gap heatmaps, sensitivity curves, class and concept distributions. |
| `results/experiment_{N}/figures/streams/` | Everything produced by `characterize_streams.py`: concept fingerprints, pairwise distance matrices, and (Exp 2) the separation-vs-`n_informative` spread curve. |

Most `analysis_{N}.py` figures are guarded by an existence check and are skipped
if already present — delete the specific file to force a redraw. The
`characterize_streams.py` figures always overwrite.

## Key Findings

--- CHECK THIS!!! ---

| Finding | Result |
|---|---|
| Raw vs aggstats | v2.0 >> v1.1 on synthetic; v1.1 competitive on high-dim real |
| Temporal features (v2.1) | No improvement — delta_mean, cosine_sim rank last in SHAP |
| Sudden drift (Exp 2) | ABFS competitive at high n_informative (crossover ≈ n_inf=10) |
| Gradual drift (Exp 2) | Komorniczak consistently better — adaptation lag compounds |
| Prequential protocol | Komorniczak structural advantage: no memory → instant adaptation |
| PAC classifier | Fails completely (BA ≈ 0.095) — excluded everywhere |
| n_informative (Exp 2) | Key driver: ABFS rises, Komorniczak flat/falls as n_inf increases |
| chunk_size (Exp 3) | To be determined — window-size sensitivity on named generators |
| Recurrence (Exp 4) | To be determined — does ABFS recognise a recurring concept across the cs $\times$ n_drifts grid? |
| Real drift type (Exp 5) | Abrupt/gradual: ABFS competitive; incremental: near baseline |
| Real dimensionality (Exp 5) | v2.0 dominates low-dim; v1.1 more robust high-dim (INSECTS, SPAM) |
| SPAM vs INSECTS (Exp 5) | TBD — first real test on a 499-feature approximately-annotated stream |