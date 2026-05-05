
from river import drift as river_drift
from river import tree
import numpy as np
from collections import deque
from scipy.stats import pointbiserialr


class ABFS_mismatch:
    """
    Lightweight incremental feature relevance tracker.

    Maintains one Hoeffding decision stump (max_depth=1) and one
    ADWIN drift detector per feature. The relevance score of
    feature j is the accuracy of its stump normalized by the
    majority class baseline.

    Key design decisions:

    1. Feature type encoding:
       Numeric features are passed as float — the Hoeffding stump
       learns an optimal numeric split threshold.
       Categorical features are passed as string — forces River
       to treat them as nominal variables. The caller specifies
       which features are categorical via categorical_features.

    2. Sliding windows for both accuracy and majority frequency:
       Both the stump accuracy estimate and the majority class
       frequency are computed from sliding windows of recent
       instances. This ensures both estimates adapt at the same
       rate after drift, keeping the normalization consistent.
       Using global cumulative counts would cause the accuracy
       estimate to lag behind the majority frequency estimate
       after drift, artificially inflating or deflating scores.

    3. No stump reset on drift detection:
       Unlike the original ABFS, stumps are not reset when drift
       is detected. Our goal is meta-feature extraction — we want
       stable, continuously evolving relevance estimates.
       Resetting stumps on every ADWIN signal causes score
       instability. ADWIN still counts drift signals but does
       not interrupt stump learning.

    Normalization:
        normalized = (raw_accuracy - majority_freq) /
                     (1 - majority_freq)
        clipped to [0, 1]

    A truly irrelevant feature scores near 0 after normalization.
    A genuinely predictive feature scores above 0.
    """

    def __init__(self, n_features,
                 categorical_features=None,
                 class_window_size=2000,
                 accuracy_window_size=2000):
        """
        Parameters:
            n_features            : total number of stream features
            categorical_features  : list of feature indices that are
                                    categorical. All others numeric.
                                    e.g. [0,1,2] for STAGGER
                                    None or [] for SEA
            class_window_size     : sliding window size for majority
                                    class frequency estimation
            accuracy_window_size  : sliding window size for stump
                                    accuracy estimation per feature
        """
        self.n_features           = n_features
        self.categorical_features = set(categorical_features or [])
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
        # both windows use recent history → normalization consistent
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

        for j in range(self.n_features):
            # encode based on feature type
            if j in self.categorical_features:
                x_single = {j: str(int(x_arr[j]))}
            else:
                x_single = {j: float(x_arr[j])}

            pred = self.stumps[j].predict_one(x_single)
            if pred is None:
                pred = 0

            correct = int(pred == y)
            error   = 1 - correct

            # sliding window accuracy — forgets old instances
            # ensures accuracy adapts at same rate as majority freq
            self.accuracy_windows[j].append(correct)

            self.detectors[j].update(error)
            self.stumps[j].learn_one(x_single, y)

            # ADWIN counts drift signals but does NOT reset stump
            if self.detectors[j].drift_detected:
                self.drift_count      += 1
                self.time_since_drift  = 0

    def majority_class_frequency(self):
        """
        Return the frequency of the most common class in the
        recent sliding window.
        """
        if not self.class_window:
            return 0.5
        counts = {}
        for label in self.class_window:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.values()) / len(self.class_window)

    def relevance_scores(self):
        """
        Return normalized relevance score for each feature.

        raw accuracy is computed from the sliding window —
        only recent instances contribute to the estimate.
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
        Call this at the end of each window.
        """
        count            = self.drift_count
        self.drift_count = 0
        return count
    

class ABFS_match:
    """
    Lightweight incremental feature relevance tracker.

    Maintains one Hoeffding decision stump (max_depth=1) and one
    ADWIN drift detector per feature. The relevance score of
    feature j is the accuracy of its stump normalized by the
    majority class baseline.

    Key design decisions:

    1. Feature type encoding:
       Numeric features are passed as float — the Hoeffding stump
       learns an optimal numeric split threshold.
       Categorical features are passed as string — forces River
       to treat them as nominal variables. The caller specifies
       which features are categorical via categorical_features.

    2. Sliding windows for both accuracy and majority frequency:
       Both the stump accuracy estimate and the majority class
       frequency are computed from sliding windows of recent
       instances. This ensures both estimates adapt at the same
       rate after drift, keeping the normalization consistent.
       Using global cumulative counts would cause the accuracy
       estimate to lag behind the majority frequency estimate
       after drift, artificially inflating or deflating scores.

    3. No stump reset on drift detection:
       Unlike the original ABFS, stumps are not reset when drift
       is detected. Our goal is meta-feature extraction — we want
       stable, continuously evolving relevance estimates.
       Resetting stumps on every ADWIN signal causes score
       instability. ADWIN still counts drift signals but does
       not interrupt stump learning.

    Normalization:
        normalized = (raw_accuracy - majority_freq) /
                     (1 - majority_freq)
        clipped to [0, 1]

    A truly irrelevant feature scores near 0 after normalization.
    A genuinely predictive feature scores above 0.
    """

    def __init__(self, n_features,
                 categorical_features=None,
                 class_window_size=200,
                 accuracy_window_size=200):
        """
        Parameters:
            n_features            : total number of stream features
            categorical_features  : list of feature indices that are
                                    categorical. All others numeric.
                                    e.g. [0,1,2] for STAGGER
                                    None or [] for SEA
            class_window_size     : sliding window size for majority
                                    class frequency estimation
            accuracy_window_size  : sliding window size for stump
                                    accuracy estimation per feature
        """
        self.n_features           = n_features
        self.categorical_features = set(categorical_features or [])
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
        # both windows use recent history → normalization consistent
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

        for j in range(self.n_features):
            # encode based on feature type
            if j in self.categorical_features:
                x_single = {j: str(int(x_arr[j]))}
            else:
                x_single = {j: float(x_arr[j])}

            pred = self.stumps[j].predict_one(x_single)
            if pred is None:
                pred = 0

            correct = int(pred == y)
            error   = 1 - correct

            # sliding window accuracy — forgets old instances
            # ensures accuracy adapts at same rate as majority freq
            self.accuracy_windows[j].append(correct)

            self.detectors[j].update(error)
            self.stumps[j].learn_one(x_single, y)

            # ADWIN counts drift signals but does NOT reset stump
            if self.detectors[j].drift_detected:
                self.drift_count      += 1
                self.time_since_drift  = 0

    def majority_class_frequency(self):
        """
        Return the frequency of the most common class in the
        recent sliding window.
        """
        if not self.class_window:
            return 0.5
        counts = {}
        for label in self.class_window:
            counts[label] = counts.get(label, 0) + 1
        return max(counts.values()) / len(self.class_window)

    def relevance_scores(self):
        """
        Return normalized relevance score for each feature.

        raw accuracy is computed from the sliding window —
        only recent instances contribute to the estimate.
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
        Call this at the end of each window.
        """
        count            = self.drift_count
        self.drift_count = 0
        return count