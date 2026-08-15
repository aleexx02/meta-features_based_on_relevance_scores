
from river import drift as river_drift
from river import tree
import numpy as np
from collections import deque


# ============================================================
#  What this module is (corrected understanding)
# ============================================================
# EMF is OUR per-feature relevance-scoring mechanism. It is INSPIRED BY
# ABFS's idea of using one decision stump per feature to gauge how
# predictive that feature is -- but it is NOT the ABFS algorithm.
#
#   - ABFS (Barddal et al. 2019) is a feature-SELECTION method: a
#     boosted CHAIN of one-feature stumps that outputs a selected
#     feature SUBSET (in / out). It does not output per-feature scores.
#
#   - EMF runs one INDEPENDENT depth-1 Hoeffding stump + one ADWIN per
#     feature, in PARALLEL (no chain, no feature removal, no selection),
#     never resets on drift, and reads out a normalized RELEVANCE SCORE
#     per feature. That score vector is what we turn into meta-features.
#
# The ablation configs below exist ONLY to justify EMF's two design
# choices -- they switch ON the two ABFS-style behaviours (reset,
# weight propagation) so we can show why we left them OFF. They are
# four settings of ONE mechanism, not "ABFS vs EMF".
#
# NOTE on weight propagation: the lambda update rule matches Oza online
# boosting / Barddal's pseudocode, BUT it is applied to independent
# per-feature stumps in fixed index order (not ABFS's selection-ordered
# chain with feature removal). So the weight-prop config tests
# "boosting-style weighting", NOT faithful original ABFS. Describe it
# that way. Our ablation shows it does not help, so EMF omits it.
# ============================================================


class _PerFeatureRelevanceTracker:
    """
    Base mechanism: one depth-1 Hoeffding stump + one ADWIN drift
    detector per feature, run in parallel. The relevance score of
    feature j is its stump's recent accuracy, corrected against the
    majority-class baseline:

        normalized = (raw_accuracy - majority_freq) / (1 - majority_freq)
        clipped to [0, 1]

    A feature that predicts no better than guessing the majority class
    scores ~0; a fully predictive feature scores ~1.

    Two behaviours are fixed by the SUBCLASS (not toggled by callers):
        reset_on_drift      -- rebuild a feature's stump when ADWIN fires
        weight_propagation  -- Oza-style boosting weight across features

    Both default OFF here; the canonical method EMF keeps them OFF.

    Design notes:
      * Feature encoding: numeric features -> float (stump learns a
        numeric split threshold); categorical features -> string
        (River treats them as nominal). Caller lists categorical
        indices via categorical_features (e.g. [0,1,2] for STAGGER).
      * Sliding windows for BOTH accuracy and majority frequency, so
        the two estimates adapt at the same rate and the normalization
        stays consistent after a change.
    """

    # Ablation behaviours -- overridden by subclasses, never by callers.
    reset_on_drift = False
    weight_propagation = False

    def __init__(self, n_features,
                 categorical_features=None,
                 class_window_size=200,
                 accuracy_window_size=200,
                 seed=42):
        """
        Parameters:
            n_features            : total number of stream features
            categorical_features  : list of categorical feature indices
                                    (all others numeric). e.g. [0,1,2]
                                    for STAGGER; None or [] for SEA.
            class_window_size     : sliding window for majority-class
                                    frequency estimation
            accuracy_window_size  : sliding window for per-feature stump
                                    accuracy estimation
        """
        self.n_features           = n_features
        self.categorical_features = set(categorical_features or [])
        self.rng                  = np.random.default_rng(seed)
        self._lam_correct         = np.zeros(n_features)
        self._lam_wrong           = np.zeros(n_features)
        self.detectors            = [river_drift.ADWIN()
                                     for _ in range(n_features)]
        self.stumps               = [
            tree.HoeffdingTreeClassifier(max_depth=1)
            for _ in range(n_features)
        ]
        self.drift_count          = 0
        self.time_since_drift     = 0

        # sliding window for majority class frequency estimation
        self.class_window         = deque(maxlen=class_window_size)

        # sliding window for accuracy estimation per feature
        # both windows use recent history -> normalization consistent
        self.accuracy_windows     = [
            deque(maxlen=accuracy_window_size)
            for _ in range(n_features)
        ]

    def update(self, x_arr, y):
        """
        Update relevance scores with one incoming instance.

        Parameters:
            x_arr : numpy array of feature values
            y     : true class label
        """
        self.time_since_drift += 1
        self.class_window.append(y)

        lam = 1.0
        for j in range(self.n_features):
            if j in self.categorical_features:
                x_single = {j: str(int(x_arr[j]))}
            else:
                x_single = {j: float(x_arr[j])}

            pred = self.stumps[j].predict_one(x_single)
            if pred is None:
                pred = 0
            correct = int(pred == y)
            error   = 1 - correct

            self.accuracy_windows[j].append(correct)
            self.detectors[j].update(error)

            if self.weight_propagation:
                # Boosting weight update EXACTLY as in ABFS
                # (Barddal et al. 2019, Algorithm 1, lines 10-15):
                #   correct: lam_c += lam;  lam *= (lam_c + lam_e) / (2*lam_c)
                #   wrong:   lam_e += lam;  lam *= (lam_c + lam_e) / (2*lam_e)
                # Each stump is trained with k ~ Poisson(lam) copies
                # (Oza online boosting, which ABFS builds on; the paper's
                # "train assuming a weight lam", line 22).
                #
                # This reproduces ABFS's WEIGHT RULE, applied across our
                # PARALLEL per-feature stumps in index order -- NOT ABFS's
                # selection-ordered chain with feature removal. It is the
                # paper's weighting scheme, not the full ABFS algorithm,
                # and is used only as an ablation to show it does not help.
                k = int(self.rng.poisson(lam))
                for _ in range(k):
                    self.stumps[j].learn_one(x_single, y)
                if correct:
                    self._lam_correct[j] += lam
                    denom = 2.0 * self._lam_correct[j]
                else:
                    self._lam_wrong[j] += lam
                    denom = 2.0 * self._lam_wrong[j]
                total = self._lam_correct[j] + self._lam_wrong[j]
                lam *= total / denom if denom > 0 else 1.0
            else:
                self.stumps[j].learn_one(x_single, y)

            if self.detectors[j].drift_detected:
                self.drift_count     += 1
                self.time_since_drift = 0
                if self.reset_on_drift:
                    self.stumps[j] = tree.HoeffdingTreeClassifier(max_depth=1)
                    self.accuracy_windows[j].clear()
                    self.detectors[j] = river_drift.ADWIN()

    def majority_class_frequency(self):
        """
        Frequency of the most common class in the recent sliding window.
        """
        if not self.class_window:
            return 0.5
        counts = {}
        for label in self.class_window:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.values()) / len(self.class_window)

    def relevance_scores(self):
        """
        Normalized relevance score for each feature.

        Raw accuracy is computed from the sliding window -- only recent
        instances contribute to the estimate.
        """
        majority_freq = self.majority_class_frequency()
        denom         = 1.0 - majority_freq + 1e-10

        scores = []
        for j in range(self.n_features):
            window     = self.accuracy_windows[j]
            raw        = sum(window) / len(window) if window else 0.0
            normalized = (raw - majority_freq) / denom
            scores.append(float(np.clip(normalized, 0.0, 1.0)))

        return scores

    def pop_drift_count(self):
        """
        Return drift count since last call and reset to zero.
        Call at the end of each window.
        """
        count            = self.drift_count
        self.drift_count = 0
        return count


# ============================================================
#  The method
# ============================================================

class EMF(_PerFeatureRelevanceTracker):
    """EMF -- our relevance-scoring mechanism (Extended Mechanism for
    Feature-relevance). Independent per-feature stumps, NO reset on
    drift, NO weight propagation. This is the method used in all the
    experiments."""
    reset_on_drift     = False
    weight_propagation = False


# ============================================================
#  Ablation configurations (used ONLY by the sanity-check script)
#  Four settings of the SAME mechanism above. They justify EMF's two
#  design choices by switching the ABFS-style behaviours back ON.
#
#  sanity-check label  ->  class here            (reset , weight-prop)
#  --------------------------------------------------------------------
#  orig                ->  ConfigResetWeightProp  ( on   , on  )
#  noweight            ->  ConfigReset            ( on   , off )
#  noreset             ->  ConfigWeightProp       ( off  , on  )
#  emf                 ->  EMF                    ( off  , off )
# ============================================================

class ConfigResetWeightProp(_PerFeatureRelevanceTracker):
    """Both ABFS-style behaviours ON (sanity-check 'orig'). Reset on
    drift + boosting-style weight propagation. Closest scoring analogue
    to ABFS's choices -- NOT literally original ABFS."""
    reset_on_drift     = True
    weight_propagation = True


class ConfigReset(_PerFeatureRelevanceTracker):
    """Reset ON, weight propagation OFF (sanity-check 'noweight')."""
    reset_on_drift     = True
    weight_propagation = False


class ConfigWeightProp(_PerFeatureRelevanceTracker):
    """Reset OFF, weight propagation ON (sanity-check 'noreset')."""
    reset_on_drift     = False
    weight_propagation = True



ABFS_match = EMF