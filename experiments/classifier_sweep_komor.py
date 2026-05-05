
# Replication of the classifier sweep protocol from:
# Komorniczak et al. (2024)
# https://github.com/w4k2/meta-concept-descriptor
#
# This module replicates the evaluation protocol from
# E2_clf_synthetic.py, implementing the same shuffling,
# repeated stratified k-fold cross-validation, and
# balanced accuracy computation. It is used as a shared
# utility by both:
#   - evaluate_concept_classification.py: runs the sweep
#     on our ABFS-based meta-features
#   - replicate_with_our_pipeline.py: runs the sweep on
#     their pre-extracted meta-features to verify that
#     our evaluation pipeline produces the same results
#
# By using the same protocol in both scripts, any
# difference in results between their meta-features and
# ours can be attributed to the meta-features themselves
# rather than to a difference in the evaluation protocol.

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn import clone

BASE_CLFS = [
    ('GNB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=11313)),
    ('DT',  DecisionTreeClassifier(random_state=11313)),
    ('MLP', MLPClassifier(random_state=11313))
]

def run_classifier_sweep(X, y, n_splits=2, n_repeats=5,
                         cv_random_state=3242,
                         shuffle_seed=1233):
    """
    Run the classifier sweep on a meta-dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_features)
        Meta-feature vectors.
    y : np.ndarray, shape (n_windows,)
        Concept labels.
    n_splits : int
        Number of folds in cross-validation.
    n_repeats : int
        Number of times cross-validation is repeated.
    cv_random_state : int
        Random state for RepeatedStratifiedKFold.
    shuffle_seed : int
        Random seed for shuffling before cross-validation.

    Returns
    -------
    mean_ba : np.ndarray, shape (n_classifiers,)
        Mean balanced accuracy per classifier.
    std_ba : np.ndarray, shape (n_classifiers,)
        Std balanced accuracy per classifier.
    clf_res : np.ndarray, shape (n_folds, n_classifiers)
        Raw balanced accuracy per fold per classifier.
    """
    # preprocessing
    X = X.copy().astype(float)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1

    # shuffle
    np.random.seed(shuffle_seed)
    p    = np.random.permutation(X.shape[0])
    X_s  = X[p]
    y_s  = y[p]

    rskf    = RepeatedStratifiedKFold(n_splits=n_splits,
                                      n_repeats=n_repeats,
                                      random_state=cv_random_state)
    n_folds = n_splits * n_repeats
    clf_res = np.zeros((n_folds, len(BASE_CLFS)))

    for fold_id, (train, test) in enumerate(rskf.split(X_s, y_s)):
        for clf_id, (name, base_clf) in enumerate(BASE_CLFS):
            clf  = clone(base_clf)
            pred = clf.fit(X_s[train], y_s[train]).predict(X_s[test])
            clf_res[fold_id, clf_id] = \
                balanced_accuracy_score(y_s[test], pred)

    mean_ba = np.mean(clf_res, axis=0)
    std_ba  = np.std(clf_res,  axis=0)

    return mean_ba, std_ba, clf_res


def print_results(mean_ba, std_ba, label=''):
    clf_names = [name for name, _ in BASE_CLFS]
    if label:
        print(f"\n{label}")
    print(f"{'Classifier':<10s} {'Mean BA':>10s} {'Std BA':>8s}")
    print('-' * 30)
    for clf_id, name in enumerate(clf_names):
        print(f"{name:<10s} {mean_ba[clf_id]:>10.4f} "
              f"{std_ba[clf_id]:>8.4f}")