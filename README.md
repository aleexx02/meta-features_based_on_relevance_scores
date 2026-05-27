## Execution Order

### Experiment 0: Pipeline Verification

1. **`external/komorniczak/E1_extract_synthetic.py`**
   Generates synthetic streams and extracts pymfe meta-features for each chunk across 9 measure groups (clustering, complexity, concept, general, info-theory, itemset, landmarking, model-based, statistical). Produces one `.npy` file per measure group in `external/komorniczak/results/`. This is the slowest step: run on a cluster if possible.

2. **`external/komorniczak/E2_clf_synthetic.py`**
   Loads the 9 `.npy` files and runs a classifier sweep (GNB, KNN, SVM, DT, MLP) on each measure group across all drift types and replications. Produces `external/komorniczak/results/clf.npy`. Compare the output against Figure 12 of Komorniczak to confirm their pipeline runs correctly on our machine.

3. **`experiments/comparison.py`**
   Loads `external/komorniczak/results/clf.npy` and compares the balanced accuracy values against Figure 12 of Komorniczak for sudden and gradual drift. Produces a side-by-side heatmap saved to `results/experiment_0/`. If the difference is small, the pipeline is confirmed to run correctly on our machine.

4. **`experiments/replication_check_1a.py`**
   Loads the pre-extracted `.npy` files from `external/komorniczak/results/` and runs them through our evaluation protocol (`classifier_sweep_komor.py`). Produces a heatmap of balanced accuracy per measure group and a side-by-side comparison against the E2 output, saved to `results/experiment_0/figures/`. If the results match E2 closely, our evaluation protocol is confirmed equivalent to theirs. The results of this script are used as the Komorniczak baseline in all subsequent experiments.



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


12. **`experiments/analysis_1c.py --sanity --trajectory --shap --metrics`**
    Loads the pre-computed `.npy` results from `results/experiment_1c/` and produces: sanity check plots, performance trajectory plots (cumulative BA over time with concept boundaries), SHAP feature importance plots, and F1/Kappa heatmaps. All figures saved to `results/experiment_1c/figures/analysis/`.