# ============================================================
# Prequential (test-then-train) evaluation protocol using
# River incremental classifiers.
#
# This module is the prequential equivalent of
# classifier_sweep_komor.py. It is used by:
#   - evaluate_concept_classification_1c.py: to run the sweep
#     on our ABFS-based meta-features
#   - komor_concept_classification_1c.py: to run the sweep on
#     their pre-extracted meta-features
#
# By using the same protocol in both scripts, any difference
# in results can be attributed to the meta-features themselves
# rather than to a difference in the evaluation protocol.
#
# Evaluation protocol:
#   For each window t:
#     1. Test: predict concept label using current classifier
#     2. Train: update classifier with true label
#   Cold start: first prediction is random (no prior training).
#   Metrics: cumulative balanced accuracy, macro F1, Cohen's Kappa.
#
# Returns per-window trajectories of shape (n_windows, n_clfs).
# ============================================================

import numpy as np
from river import naive_bayes, neighbors, linear_model, tree, neural_net, metrics


BASE_CLFS_RIVER = [
    ('GNB', lambda: naive_bayes.GaussianNB()),
    ('KNN', lambda: neighbors.KNNClassifier()),
    ('SVM', lambda: linear_model.SVM()),
    ('HT',  lambda: tree.HoeffdingTreeClassifier()),
    ('MLP', lambda: neural_net.MLPClassifier()),
]


def make_river_clfs():
    """Instantiate a fresh set of River classifiers."""
    return [(name, clf()) for name, clf in BASE_CLFS_RIVER]


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
        Mean cumulative balanced accuracy per classifier.
    std_ba : np.ndarray, shape (n_clfs,)
        Std cumulative balanced accuracy per classifier.
    traj_ba : np.ndarray, shape (n_windows, n_clfs)
        Per-window cumulative balanced accuracy trajectory.
    mean_f1 : np.ndarray, shape (n_clfs,)
        Mean cumulative macro F1 per classifier.
    std_f1 : np.ndarray, shape (n_clfs,)
        Std cumulative macro F1 per classifier.
    traj_f1 : np.ndarray, shape (n_windows, n_clfs)
        Per-window cumulative macro F1 trajectory.
    mean_kappa : np.ndarray, shape (n_clfs,)
        Mean cumulative Cohen's Kappa per classifier.
    std_kappa : np.ndarray, shape (n_clfs,)
        Std cumulative Cohen's Kappa per classifier.
    traj_kappa : np.ndarray, shape (n_windows, n_clfs)
        Per-window cumulative Cohen's Kappa trajectory.
    """
    # preprocessing
    X = X.copy().astype(float)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1

    clfs      = make_river_clfs()
    n_windows = len(X)
    n_clfs    = len(clfs)
    classes   = list(np.unique(y))

    traj_ba    = np.zeros((n_windows, n_clfs))
    traj_f1    = np.zeros((n_windows, n_clfs))
    traj_kappa = np.zeros((n_windows, n_clfs))

    # one cumulative metric object per classifier
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
                y_pred = classes[0]  # cold start: predict first class

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