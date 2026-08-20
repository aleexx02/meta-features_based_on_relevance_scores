# classifier_sweep_prequential.py
# ============================================================
# Prequential (test-then-train) evaluation protocol using
# River incremental classifiers.
#
# This module is the prequential equivalent of
# classifier_sweep_komor.py. It is used by:
#   - evaluate_concept_classification_1.py: to run the sweep
#     on our ReMF meta-features
#   - komor_concept_classification_1.py: to run the sweep on
#     their pre-extracted meta-features
#
# By using the same protocol in both scripts, any difference
# in results can be attributed to the meta-features themselves
# rather than to a difference in the evaluation protocol.
#
# Classifiers:
#   - GNB: River GaussianNB
#   - KNN: River KNNClassifier
#   - HT:  River HoeffdingTreeClassifier
#   - MLP: sklearn MLPClassifier with partial_fit for incremental updates
#          Note: River does not provide an MLPClassifier for classification.
#          sklearn MLP with partial_fit allows direct comparison with the
#          MLP used in experiments 1a/1b.

# Evaluation protocol:
#   For each window t:
#     1. Test: predict concept label using current classifier
#     2. Train: update classifier with true label
#   Cold start: first prediction is random (no prior training).
#   Metrics: cumulative balanced accuracy, macro F1, Cohen's Kappa.
#
# Returns per-window trajectories of shape (n_windows, n_clfs).

# Key difference from classifier_sweep_komor.py:
# Here, windows are processed one by one in temporal order. For each
# window, the classifier first predicts the concept label (test), then
# updates itself with the true label (train). The classifier is never
# retrained from scratch, it accumulates knowledge incrementally.
# Each sample presented to the classifier is one meta-feature vector,
# extracted from a 200-instance chunk of the stream, processed
# sequentially in temporal order.
# This is closer to a real stream deployment scenario where only past
# windows are available at training time.


import numpy as np
from river import naive_bayes, neighbors, tree, metrics
from sklearn.neural_network import MLPClassifier


class SklearnMLPWrapper:
    """
    Wraps sklearn MLPClassifier to expose the same
    predict_one / learn_one interface as River classifiers,
    using partial_fit for incremental updates.
    """
    def __init__(self, classes, random_state=11313):
        self.clf = MLPClassifier(random_state=random_state)
        self.classes = classes
        self.fitted = False

    def predict_one(self, x_dict):
        if not self.fitted:
            return None  # cold start
        x = np.array([x_dict[i] for i in range(len(x_dict))]).reshape(1, -1)
        return int(self.clf.predict(x)[0])

    def learn_one(self, x_dict, y_true):
        x = np.array([x_dict[i] for i in range(len(x_dict))]).reshape(1, -1)
        self.clf.partial_fit(x, [y_true], classes=self.classes)
        self.fitted = True


BASE_CLFS_PREQUENTIAL = [
    ('GNB', lambda classes: naive_bayes.GaussianNB()),
    ('KNN', lambda classes: neighbors.KNNClassifier()),
    ('HT',  lambda classes: tree.HoeffdingTreeClassifier()),
    ('MLP', lambda classes: SklearnMLPWrapper(classes)),
]

def make_prequential_clfs(classes):
    """Instantiate a fresh set of classifiers."""
    return [(name, clf_fn(classes)) for name, clf_fn in BASE_CLFS_PREQUENTIAL]



# ============================================================
#  PREQUENTIAL SWEEP
# ============================================================
def run_prequential_sweep(X, y):
    """
    Run the prequential classifier sweep on a meta-dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_features)
        Meta-feature vectors in temporal order.
    y : np.ndarray, shape (n_windows,)
        Concept labels in temporal order.

    Returns
    -------
    mean_ba : np.ndarray, shape (n_clfs,)
    std_ba  : np.ndarray, shape (n_clfs,)
    traj_ba : np.ndarray, shape (n_windows, n_clfs)
    mean_f1 : np.ndarray, shape (n_clfs,)
    std_f1  : np.ndarray, shape (n_clfs,)
    traj_f1 : np.ndarray, shape (n_windows, n_clfs)
    mean_kappa : np.ndarray, shape (n_clfs,)
    std_kappa  : np.ndarray, shape (n_clfs,)
    traj_kappa : np.ndarray, shape (n_windows, n_clfs)
    """
    X = X.copy().astype(float)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1

    classes   = list(np.unique(y))
    clfs      = make_prequential_clfs(classes)
    n_windows = len(X)
    n_clfs    = len(clfs)

    traj_ba    = np.zeros((n_windows, n_clfs))
    traj_f1    = np.zeros((n_windows, n_clfs))
    traj_kappa = np.zeros((n_windows, n_clfs))

    ba_metrics    = [metrics.BalancedAccuracy() for _ in clfs]
    f1_metrics    = [metrics.MacroF1()          for _ in clfs]
    kappa_metrics = [metrics.CohenKappa()       for _ in clfs]

    for t in range(n_windows):
        x_dict = {i: float(X[t, i]) for i in range(X.shape[1])}
        y_true = int(y[t])

        for clf_id, (name, clf) in enumerate(clfs):
            # test
            y_pred = clf.predict_one(x_dict)
            if y_pred is None:
                y_pred = classes[0]  # cold start

            ba_metrics[clf_id].update(y_true, y_pred)
            f1_metrics[clf_id].update(y_true, y_pred)
            kappa_metrics[clf_id].update(y_true, y_pred)

            traj_ba[t,    clf_id] = ba_metrics[clf_id].get()
            traj_f1[t,    clf_id] = f1_metrics[clf_id].get()
            traj_kappa[t, clf_id] = kappa_metrics[clf_id].get()

            # train
            clf.learn_one(x_dict, y_true)

    mean_ba    = np.mean(traj_ba,    axis=0)
    std_ba     = np.std(traj_ba,     axis=0)
    mean_f1    = np.mean(traj_f1,    axis=0)
    std_f1     = np.std(traj_f1,     axis=0)
    mean_kappa = np.mean(traj_kappa, axis=0)
    std_kappa  = np.std(traj_kappa,  axis=0)

    return (mean_ba, std_ba, traj_ba,mean_f1, std_f1, traj_f1,mean_kappa, std_kappa, traj_kappa)