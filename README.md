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
│   │   ├── # add here
│   │   └── evaluate_concept_classification_3.py
│   │
│   ├── analysis_1a_1b.py
│   ├── classifier_sweep_komor.py
│   └── classifier_sweep_prequential.py
│
├── external/
│   └── komorniczak/
│       ├── results/
│       ├── E1_extract_real.py
│       ├── E1_extract_synthetic.py
│       ├── E2_clf_real.py
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



### Experiment 1a: Static Evaluation with Shuffled Cross-Validation

5. **`experiments/experiment_1a/evaluate_concept_classification_1a.py`**
   Generates the same synthetic streams, runs ABFS to extract our meta-feature vectors, and evaluates them using `classifier_sweep_komor.py` with shuffling enabled. Produces `.npy` result files in `results/experiment_1a/` and a comparison heatmap against the Komorniczak baseline from step 4. Any difference in balanced accuracy is due solely to the meta-features.

6. **`experiments/analysis_1a_1b.py --exp 1a --sanity --variance --shap --metrics`**
   Loads the pre-computed `.npy` results from `results/experiment_1a/` and produces: sanity check plots, performance variance plots, SHAP feature importance plots, and F1/Kappa heatmaps. All figures saved to `results/experiment_1a/figures/analysis/`.




### Experiment 1b: Static Evaluation without Shuffling

7. **`experiments/experiment_1b/komor_concept_classification_1b.py`**
   Evaluates Komorniczak meta-features under the no-shuffle protocol using `classifier_sweep_komor.py`. Produces `.npy` result files in `results/experiment_1b/`. Used as the Komorniczak baseline for Experiment 1b.


8. **`experiments/experiment_1b/evaluate_concept_classification_1b.py`**
   Same as step 5 but with shuffling disabled. Produces `.npy` result files in `results/experiment_1b/` and a comparison heatmap against the Komorniczak baseline from step 7.


9. **`experiments/analysis_1a_1b.py --exp 1b --sanity --variance --shap --metrics`**
   Same as step 6 but for Experiment 1b. All figures saved to `results/experiment_1b/figures/analysis/`.



### Experiment 1c: Prequential Evaluation

10. **`experiments/experiment_1c/komor_concept_classification_1c.py`**
    Evaluates Komorniczak meta-features under the prequential protocol using `classifier_sweep_prequential.py`. Skips the first 10 windows to align with our ABFS warmup. Produces `.npy` result files in `results/experiment_1c/`. Must be run before step 11.
   
11. **`experiments/experiment_1c/evaluate_concept_classification_1c.py`**
    Generates the same synthetic streams, runs ABFS to extract our meta-feature vectors, and evaluates them using `classifier_sweep_prequential.py`. Produces `.npy` result files in `results/experiment_1c/` and comparison heatmaps against the Komorniczak baseline from step 10.


12. **`experiments/analysis_1c.py --sanity --performance --shap --metrics`**
    Loads the pre-computed `.npy` results from `results/experiment_1c/` and produces: sanity check plots, performance trajectory plots (cumulative BA over time with concept boundaries), SHAP feature importance plots, and F1/Kappa heatmaps. All figures saved to `results/experiment_1c/figures/analysis/`.



### Experiment 2: Stream Configuration Sensitivity

13. **`experiments/experiment_2/evaluate_concept_classification_2.py`**
   Loops over a $4 \times 4$ grid of stream configurations (chunk\_size ∈ {100, 200, 500, 1000} $\times$ n\_informative ∈ {3, 5, 10, 15}) and evaluates both ABFS raw score meta-features (v2.0) and Komorniczak statistical meta-features under two protocols (shuffled CV and prequential) for both sudden and gradual drift. Komorniczak features are re-extracted using pymfe on the same streams (cannot reuse pre-extracted files from Experiment 1 since chunk\_size and n\_informative vary). Produces 384 `.npy` result files in `results/experiment_2/` and 64 comparison heatmap figures (32 CV + 32 prequential) in `results/experiment_2/figures/`. 

14. **`experiments/experiment_2/generate_missing_heatmaps.py`**
   Generates CV and prequential comparison heatmaps for any grid cells that were computed before the heatmap call was added to the evaluate script. Loads existing `.npy` files and produces missing figures with skip logic. Run once after all evaluate scripts have completed.

15. **`experiments/experiment_2/analysis_2.py --sanity --variance --performance --shap --metrics --grid`**
   Loads all pre-computed `.npy` result files from `results/experiment_2/` and produces per-cell analyses (sanity check plots, performance variance plots, trajectory plots, SHAP importance plots, F1 and Kappa heatmaps) plus two grid-level analyses specific to Experiment 2: gap heatmaps (ABFS minus Komorniczak BA across the $4 \times 4$ grid) and sensitivity curves (BA vs chunk\_size and BA vs n\_informative). Each flag runs the corresponding analysis independently to allow partial reruns. All figures saved to `results/experiment_2/figures/analysis/`. Can be run with `--grid` only to regenerate gap heatmaps and sensitivity curves without rerunning per-cell analyses.


### Experiment 3: Real-World Stream Evaluation (INSECTS)

16. **`external/komorniczak/E1_extract_real.py`**
    Extracts pymfe meta-features for each chunk across 9 measure groups (clustering, complexity, concept, general, info-theory, itemset, landmarking, model-based, statistical), from the three INSECTS streams (abrupt, gradual, incremental) using Komorniczak's NPYParser and ground truth drift annotations. Produces one `.npy` file per stream in `external/komorniczak/results/real/`

17. **`experiments/experiment_3/evaluate_concept_classification_3.py`**
    Extracts our ABFS meta-feature vectors from the same INSECTS streams and evaluates under prequential protocol (Experiment 1c protocol) using `classifier_sweep_prequential.py`. Produces `.npy` result files in `results/experiment_3/` and comparison heatmap figures in
    `results/experiment_3/figures/` against the Komorniczak baseline. Any difference in balanced accuracy is due solely to the meta-features.