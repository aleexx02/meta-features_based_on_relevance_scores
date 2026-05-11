## Execution Order

1. **`external/komorniczak/E1_extract_synthetic.py`**
   Generates synthetic streams and extracts pymfe meta-features for each chunk across 9 measure groups (clustering, complexity, concept, general, info-theory, itemset, landmarking, model-based, statistical). Produces one `.npy` file per measure group in `external/komorniczak/results/`. This is the slowest step, run on a cluster if possible.

2. **`external/komorniczak/E2_clf_synthetic.py`**
   Loads the 9 `.npy` files and runs a classifier sweep (GNB, KNN, SVM, DT, MLP) on each measure group across all drift types and replications. Produces `external/komorniczak/results/clf.npy`. Compare the output against Figure 12 of Komorniczak et al. (2024) to confirm their pipeline runs correctly.

3. **`external/komorniczak/comparison.py`**
   Loads `external/komorniczak/results/clf.npy` and compares the balanced accuracy values against Figure 12 of Komorniczak et al. (2024) for sudden and gradual drift. Produces a side-by-side heatmap showing the paper values, the E2 output, and the difference between them, saved to `external/komorniczak/results/compare_expected_vs_actual_results_synthetic.png`. If the difference is small, the pipeline is confirmed to run correctly.
   
4. **`experiments/experiment_1/replication_check.py`**
   Loads the same `.npy` files from `external/komorniczak/results/` and runs them through our evaluation protocol (`classifier_sweep_komor.py`). Output should match E2 closely. If it does, our evaluation pipeline is confirmed equivalent to theirs and the subsequent comparison with our ABFS-based meta-features is fully controlled.

5. **`experiments/experiment_1/evaluate_concept_classification.py`**
   Generates the same synthetic streams, runs ABFS to extract our meta-feature vectors, and evaluates them using the same protocol as step 3. Any difference in balanced accuracy compared to step 3 is due solely to the meta-features.
