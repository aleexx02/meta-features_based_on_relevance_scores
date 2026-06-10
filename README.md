# Meta-features Based on Relevance Scores

ABFS-based stream meta-features for concept classification in non-stationary data streams.

## Project Structure

meta-features_based_on_relevance_scores/
│
├── abfs/
│   └── abfs_implementation.py
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
│   │    # add more experiments here
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
│   │
│   ├── experiment_1a/
│   │   └── figures/
│   │       └── analysis/
│   │
│   ├── experiment_1b/
│   │   └── figures/
│   │       └── analysis/
│   │
│   ├── experiment_1c/
│   │    └── figures/
│   │       └── analysis/
│   │
│   ├── experiment_2/
│   │    └── figures/
│   │       └── analysis/ 
│   │
│   ├── experiment_3/
│   │    └── figures/
│   │       └── analysis/
│   │    ...
│   │
│   └── sanity_check/
│        └── figures/
│
├── streams/
│   └── generators.py
│
├── .gitignore
├── plot_results.py
├── README.md
├── requirements.txt
└── sanity_check.py

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



### Experiments 1a & 1b: Batch CV (Historical)

> **Note:** Shuffled (1a) and unshuffled (1b) CV for historical comparison only.
> Key finding: shuffling makes no difference (<0.002 BA) — non-recurring concepts
> have no temporal structure to leak across folds.

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

Output naming:
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

→ `results/experiment_2/figures/analysis/`



### Experiment 3: Real-World Stream Evaluation (INSECTS)

Streams:
- INSECTS-abrupt_imbalanced_norm      236 chunks, 33 features, 2 concepts
- INSECTS-gradual_imbalanced_norm     236 chunks, 33 features, 6 concepts
- INSECTS-incremental_imbalanced_norm 236 chunks, 33 features, 6 concepts

**15. `external/komorniczak/E1_extract_real.py`** (or 9 parallel scripts per measure)
Extracts pymfe features for all 9 measure groups × 3 streams = 27 .npy files.
Output: `external/komorniczak/results/real/komor_real_{stream}_{measure}.npy`

Submit to cluster:
```bash
mkdir -p external/komorniczak/logs
oarsub -S external/komorniczak/E1_real_{measure}.sh   # one per measure
ls external/komorniczak/results/real/ | wc -l         # verify: should be 27
```

**16. `experiments/experiment_3/evaluate_concept_classification_3.py`**
All 3 ABFS versions + all 9 Komorniczak groups, prequential, single pass per stream.

Output:
```
abfs_y_{stream}.npy                      (n_windows,)
preq_abfs_{version}_ba_{stream}.npy      (n_windows, n_clfs)
preq_komor_{measure}_ba_{stream}.npy     (n_windows, n_clfs)
heatmap_combined_exp3_{stream}.png       left: 9 Komor groups | right: 3 ABFS versions
```

**17. `experiments/experiment_3/analysis_3.py --sanity --performance --shap --metrics`**
- `--sanity`      relevance scores, meta-features per window, PCA per stream
- `--performance` BA trajectories (3 ABFS stacked; 9 Komorniczak in 3×3 grid)
- `--shap`        SHAP — all 4 classifiers, per stream per ABFS version
- `--metrics`     F1 and Kappa heatmaps (final window value)

→ `results/experiment_3/figures/analysis/`