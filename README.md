# Meta-features Based on Relevance Scores

ABFS-based stream meta-features for concept classification in non-stationary data streams.

## Project Structure
```
meta-features_based_on_relevance_scores/
│
├── abfs/
│   └── abfs_implementation.py
│
├── data/   # local data files - not committed to git
│   ├── real/
│   │   ├──annotated_streams/                  
│   │       ├── INSECTS-abrupt_imbalanced_norm.npy
│   │       ├── INSECTS-gradual_imbalanced_norm.npy
│   │       ├── INSECTS-incremental_imbalanced_norm.npy
│   │       └── poker-lsn-1-2vsAll-pruned.npy
│   │    ├── annotated_streams_gt/
│   │       ├── INSECTS-abrupt_imbalanced_norm.npy
│   │       ├── INSECTS-gradual_imbalanced_norm.npy
│   │       ├── INSECTS-incremental_imbalanced_norm.npy
│   │       └── poker-lsn-1-2vsAll-pruned.npy
│   │   └── unannotated_streams/
│   │       └── elec2/
│   │           ├── elec2_ordered.npz
│   │           ├── elec2_ordered_meta.json
│   │           ├── elec2_inspection.json
│   │           ├── elec2_baseline_profiles.npz
│   │           └── elec2_proxy_labels.npy
│   │ 
│   ├── synthetic/
│   ...
│   # ADD HERE
│   
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
│   ├── experiment_3/
│   │   ├── analysis_3.py
│   │   └── evaluate_concept_classification_3.py
│   │
│   ├── analysis_1a_1b.py
│   ├── classifier_sweep_komor.py
│   └── classifier_sweep_prequential.py
│
├── external/
│   └── komorniczak/
│       ├── results/
│       │   ├── real/
│       │   └── synthetic/
│       ├── E1_extract_real.py
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
│   │   └── figures/
│   │       └── analysis/
│   └── sanity_check/
│       └── figures/
│
├── streams/
│   ├── real_streams_generator.py
│   ├── synthetic_streams_generator.py # ADD IMPLEMENTATION
│   └── ...
│
├── .gitignore
├── plot_results.py
├── README.md
├── requirements.txt
└── sanity_check.py
```

---

## Data Files

All data files are gitignored and must be generated or downloaded locally.
The `data/` folder has the following:

**`data/real/annotated_streams/`** contains the actual stream instances for the four
annotated real-world streams: INSECTS (three variants) and poker-lsn. Each
file is a matrix of shape `(n_instances, n_features + 1)` where the last
column is the class label.

**`data/real/annotated_streams_gt/`** contains the ground truth drift annotations for
those same four streams. Each file is a small array of chunk indices marking
where concept drift occurs, for example, `[125]` for the abrupt INSECTS
stream means drift happens at chunk 125.

**`data/real/unannotated_streams/`** contains the outputs of the proxy label
pipeline for streams that have no ground truth annotations. Currently only
elec2 is included. Inside `elec2/` you have the ordered stream matrix, the
chunk-level classifier performance profiles, the proxy concept labels, and
two JSON files with metadata and dataset statistics.

Ground truth annotations are available for INSECTS and poker-lsn because:
- **INSECTS**: temperature was deliberately manipulated in a laboratory
  setting to induce drift at known time points.
- **poker-lsn**: virtual drift is introduced artificially by sorting
  instances by rank and suit — boundaries are known by construction.


To populate `data/real/annotated_streams/` and `data/real/annotated_streams_gt/` copy the files from the Komorniczak repository:
```bash
# annotated streams (Approach 1)
mkdir -p data/real/annotated_streams data/real/annotated_streams_gt
cp ~/code_komor/data/real_streams_pr/*.npy data/real/annotated_streams/
cp ~/code_komor/data/real_streams_gt/*.npy data/real/annotated_streams_gt/
```

To populate `data/real/unannotated_streams/`:
```bash
# proxy label streams (Approach 2)
python streams/real_streams_generator.py
```


## ... ADD SYNTHETIC PART ...


---

## Stream Scripts

### `streams/real_streams_generator.py`
Generates proxy concept labels for streams without ground truth annotations.
Downloads datasets via River, runs baseline classifiers (GNB, KNN, HT)
chunk by chunk, and clusters the performance profiles with KMeans to assign
proxy concept labels. Outputs go to `data/real_streams_data/{dataset}/`.
Currently runs elec2 only.

### `streams/synthetic_streams_generator.py`
#### ... ADD SYNTHETIC PART ...



---

## ABFS Meta-Feature Versions

| Version | Name | Dim | Description |
|---|---|---|---|
| v1.1 | aggstats | 8 | entropy, n_relevant, max_score, std_score, delta_mean, n_changed, drift_count, time_since_drift |
| v2.0 | raw scores | n_features | Normalized relevance score vector — feature identity preserved |
| v2.1 | raw + temporal | n_features + 2 | v2.0 + delta_mean + cosine_sim (window-to-window change) |

---

## Evaluation Protocol
All experiments use **prequential (test-then-train)** evaluation.
Batch CV (Experiments 1a/1b) is kept for historical comparison only.

**Classifiers:** River GNB, KNN, HT + sklearn MLP with partial_fit.


## Execution Order

### Experiment 0: Pipeline Verification

1. **`external/komorniczak/E1_extract_synthetic.py`**
   Generates synthetic streams and extracts pymfe meta-features for each chunk across 9 measure groups (clustering, complexity, concept, general, info-theory, itemset, landmarking, model-based, statistical). Produces one `.npy` file per measure group in `external/komorniczak/results/synthetic/`. This is the slowest step: run on a cluster if possible.

2. **`external/komorniczak/E2_clf_synthetic.py`**
   Loads the 9 `.npy` files and runs a classifier sweep (GNB, KNN, SVM, DT, MLP) on each measure group across all drift types and replications. Produces `external/komorniczak/results/synthetic/clf.npy`. Compare the output against Figure 12 of Komorniczak to confirm their pipeline runs correctly on our machine.

3. **`experiments/comparison.py`**
   Loads `external/komorniczak/results/synthetic/clf.npy` and compares the balanced accuracy values against Figure 12 of Komorniczak for sudden and gradual drift. Produces a side-by-side heatmap saved to `results/experiment_0/figures/`. If the difference is small, the pipeline is confirmed to run correctly on our machine.

4. **`experiments/replication_check_1a.py`**
   Loads the pre-extracted `.npy` files from `external/komorniczak/results/synthetic/` and runs them through our evaluation protocol (`classifier_sweep_komor.py`). Produces comparison figures saved to `results/experiment_0/figures/` and `.npy` result files in `results/experiment_1a/`. If the results match E2 closely, our evaluation protocol is confirmed equivalent to theirs. The results of this script are used as the Komorniczak baseline in Experiment 1a.



### Experiments 1a & 1b: Batch CV

**5. `experiments/experiment_1a/evaluate_concept_classification_1a.py`**
All 3 ABFS versions under shuffled CV. Heatmap: 9 Komorniczak groups × classifiers | 3 ABFS versions × classifiers.

**6. `experiments/analysis_1a_1b.py --exp 1a --sanity --variance --shap --metrics`**
Sanity, variance, SHAP, F1/Kappa. → `results/experiment_1a/figures/analysis/`

**7. `experiments/experiment_1b/komor_concept_classification_1b.py`**
Komorniczak under no-shuffle CV.

**8. `experiments/experiment_1b/evaluate_concept_classification_1b.py`**
Same as step 5 with shuffling disabled.

**9. `experiments/analysis_1a_1b.py --exp 1b --sanity --variance --shap --metrics`**
Same as step 6 for Experiment 1b. → `results/experiment_1b/figures/analysis/`


### Experiment 1c: Prequential Evaluation

10. **`experiments/experiment_1c/komor_concept_classification_1c.py`**
    Evaluates Komorniczak meta-features under the prequential protocol using `classifier_sweep_prequential.py`. Skips the first 10 windows to align with our ABFS warmup. Produces `.npy` result files in `results/experiment_1c/`. Must be run before step 11.
   
11. **`experiments/experiment_1c/evaluate_concept_classification_1c.py`**
    Generates the same synthetic streams, runs ABFS to extract our meta-feature vectors, and evaluates them using `classifier_sweep_prequential.py`. Produces `.npy` result files in `results/experiment_1c/` and comparison heatmaps against the Komorniczak baseline from step 10.


12. **`experiments/analysis_1c.py --sanity --performance --shap --metrics`**
    Loads the pre-computed `.npy` results from `results/experiment_1c/` and produces: sanity check plots, performance trajectory plots (cumulative BA over time with concept boundaries), SHAP feature importance plots, and F1/Kappa heatmaps. All figures saved to `results/experiment_1c/figures/analysis/`.



### Experiment 2: Stream Configuration Sensitivity

chunk_size ∈ {100, 200, 500, 1000} × n_informative ∈ {3, 5, 10, 15}
4×4 grid × 2 drift types × 5 replications. **Prequential only.**

**13. `experiments/experiment_2/evaluate_concept_classification_2.py`**
For each cell:
- All 3 ABFS versions extracted in one pass
- All 9 Komorniczak measure groups re-extracted via pymfe
- Prequential evaluation, 5 replications

Output files:
```
preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy   (n_reps, n_windows, n_clfs)
preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy  (n_reps, n_windows, n_clfs)
```

**14. `experiments/experiment_2/analysis_2.py --sanity --performance --shap --metrics --grid`**
- `--sanity`      relevance scores, meta-features per window, PCA (rep 0 only)
- `--performance` cumulative BA trajectory per cell
- `--shap`        SHAP — all 4 classifiers (GNB, KNN, HT proxy, MLP), raw v2.0
- `--metrics`     F1 and Kappa heatmaps (all 3 ABFS + 9 Komorniczak)
- `--grid`        gap heatmaps + sensitivity curves across 4×4 grid

Output directory: `results/experiment_2/figures/analysis/`




### Experiment 3: Real-World Stream Evaluation

Two approaches depending on whether ground truth drift annotations are available.

#### Approach 1 - Annotated streams (ground truth labels)

Annotated streams are streams for which ground truth drift boundaries are
known. Concept labels are assigned directly from the ground truth drift annotations, so
the concept classification task has a definitive correct answer and the
comparison between ABFS and Komorniczak is directly interpretable.

Streams: INSECTS (3 variants) + poker-lsn. Concept labels from manually
annotated drift boundaries. Comparison between ABFS and Komorniczak is
directly interpretable.

**15. Copy stream files (see Data Files section above)**

**16. `external/komorniczak/E1_extract_real.py`**
(or 9 parallel scripts `E1_extract_real_{measure}.py`)
Extracts pymfe features for all 9 Komorniczak measure groups $\times$ 4 streams.

Output files: `external/komorniczak/results/real/komor_real_{stream}_{measure}.npy`


**17. `experiments/experiment_3/evaluate_concept_classification_3.py`**
All 3 ABFS versions + all 9 Komorniczak groups, prequential, single pass.

Output files:

`abfs_y_{stream}.npy`  (n_windows,)

`preq_abfs_{version}_ba_{stream}.npy`  (n_windows, n_clfs)

`preq_komor_{measure}_ba_{stream}.npy`  (n_windows, n_clfs)


**18. `experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics`**
- `--sanity`      relevance scores, meta-features (per version), PCA (per version)
- `--performance` BA trajectories (3 ABFS stacked; 9 Komorniczak in 3×3 grid)
- `--shap`        SHAP — all 4 classifiers, per stream per ABFS version
- `--metrics`     F1 and Kappa heatmaps (final window value)

Output directory: `results/experiment_3/figures/analysis/`

#### Approach 2 - Unannotated streams (proxy labels)

Unannotated streams have no ground truth drift boundaries. Concept labels
are derived from baseline classifier (GNB, KNN, HT) performance profiles
clustered with KMeans: chunks where all three classifiers behave similarly
are assigned the same proxy concept label. The comparison between ABFS and
Komorniczak remains valid because both use the same proxy labels. Currently
only elec2 is included.

Stream: elec2 (45,312 instances, 8 features, 152 chunks). Concept labels
derived from baseline classifier performance profiles clustered with KMeans.
Comparison between ABFS and Komorniczak remains valid — both use the same
proxy labels.

**19. `streams/real_streams_generator.py`**
Generates proxy labels for elec2.

Output directory: `data/real_streams_data/elec2/`


**20. `experiments/experiment_3/evaluate_concept_classification_3_proxy.py`**
All 3 ABFS versions + all 9 Komorniczak groups, evaluated against proxy labels.

#### ... IMPLEMENT THIS ...

---

## Result File Naming Conventions

### Experiment 2

| Field | Values |
|---|---|
| `version` | `aggstats` \| `raw` \| `raw_temporal` |
| `measure` | `clustering` \| `complexity` \| `concept` \| `general` \| `info-theory` \| `itemset` \| `landmarking` \| `model-based` \| `statistical` |
| `cs` | `100` \| `200` \| `500` \| `1000` |
| `ni` | `3` \| `5` \| `10` \| `15` |
| `drift` | `sudden` \| `gradual` |
| shape | `(n_reps, n_windows, n_clfs)` with `n_reps=5` |

Naming convention:

`preq_abfs_{version}_ba_chunk{cs}_ninf{ni}_{drift}.npy`

`preq_komor_{measure}_ba_chunk{cs}_ninf{ni}_{drift}.npy`

where:
  * version : aggstats | raw | raw_temporal
  * measure : clustering | complexity | concept | general | info-theory |
            itemset | landmarking | model-based | statistical
  * shape   : (n_reps, n_windows, n_clfs)  n_reps=5


### Experiment 3 - Annotated Streams

| Field | Values |
|---|---|
| `version` | `aggstats` \| `raw` \| `raw_temporal` |
| `measure` | `clustering` \| `complexity` \| `concept` \| `general` \| `info-theory` \| `itemset` \| `landmarking` \| `model-based` \| `statistical` |
| `stream` | `INSECTS-abrupt_imbalanced_norm` \| `INSECTS-gradual_imbalanced_norm` \| `INSECTS-incremental_imbalanced_norm` \| `poker-lsn-1-2vsAll-pruned` |
| shape | `(n_windows, n_clfs)` - no replications (single fixed stream) |

Naming convention:

`preq_abfs_{version}_ba_{stream}.npy`

`preq_komor_{measure}_ba_{stream}.npy`

where:
  * shape : (n_windows, n_clfs)  - no replications


### Experiment 3 - Unannotated Streams (proxy labels)

#### ... ADD HERE ...


---

## Key Findings

| Finding | Result |
|---|---|
| Shuffling (1a vs 1b) | <0.002 BA — non-recurring concepts have no temporal structure |
| Raw vs aggstats | v2.0 >> v1.1 — feature identity matters |
| Temporal features (v2.1) | No improvement — delta_mean and cosine_sim rank last in SHAP |
| CV sudden drift | ABFS competitive at high n_informative (crossover ≈ n_inf=10) |
| CV gradual drift | Komorniczak consistently better — adaptation lag compounds |
| Prequential | Komorniczak structural advantage: no memory → instant adaptation |
| PAC classifier | Fails completely (BA ≈ 0.095) — excluded from all experiments |
| Exp 2 n_informative | Key driver: ABFS rises, Komorniczak flat/falls as n_inf increases |
