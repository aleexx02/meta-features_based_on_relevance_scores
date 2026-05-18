
# Replication of the classifier sweep protocol from Komorniczak.
#
# This module replicates the evaluation protocol from
# E2_clf_synthetic.py, implementing the same shuffling,
# repeated stratified k-fold cross-validation, and
# balanced accuracy computation. It is used as a shared
# utility by both:
#   - evaluate_concept_classification.py: runs the sweep
#     on our ABFS-based meta-features
#   - replication_check.py: runs the sweep on
#     their pre-extracted meta-features to verify that
#     our evaluation pipeline produces the same results
#
# By using the same protocol in both scripts, any
# difference in results between their meta-features and
# ours can be attributed to the meta-features themselves
# rather than to a difference in the evaluation protocol.


# Metrics computed per fold per classifier:
#   Balanced accuracy: measures how well the classifier identifies each concept on average.
#   Computes recall separately for each concept and averages equally across all concepts,
#   so a concept with few windows counts as much as one with many.
#
#   F1: gives equal weight to all concepts regardless of frequency, but also
#   penalises classifiers that are good at finding a concept (high recall) but make
#   many false alarms (low precision). Stricter than balanced accuracy.
#
#   Kappa: measures how much better the classifier is compared to random guessing,
#   corrected for chance. A value of 0 means no better than random, 1 means perfect.
#   Useful when the number of concepts is large and the random baseline is very low.



import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score
from sklearn import clone

BASE_CLFS = [
    ('GNB', GaussianNB()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(random_state=11313)),
    ('DT',  DecisionTreeClassifier(random_state=11313)),
    ('MLP', MLPClassifier(random_state=11313))
]

def run_classifier_sweep(X, y, n_splits=2, n_repeats=5, cv_random_state=3242, shuffle_seed=1233):
    """
    Run the classifier sweep on a meta-dataset.

    Parameters:
    X: np.ndarray, shape (n_windows, n_features)
        Meta-feature vectors.
    y: np.ndarray, shape (n_windows,)
        Concept labels.
    n_splits: int
        Number of folds in cross-validation.
    n_repeats: int
        Number of times cross-validation is repeated.
    cv_random_state: int
        Random state for RepeatedStratifiedKFold.
    shuffle_seed: int
        Random seed for shuffling before cross-validation.

    Returns:
    mean_ba: np.ndarray, shape (n_classifiers,)
        Mean balanced accuracy per classifier.
    std_ba: np.ndarray, shape (n_classifiers,)
        Std balanced accuracy per classifier.
    clf_res_ba: np.ndarray, shape (n_folds, n_classifiers)
        Raw balanced accuracy per fold per classifier.
    mean_f1: np.ndarray, shape (n_classifiers,)
        Mean F1 per classifier.
    std_f1: np.ndarray, shape (n_classifiers,)
        Std F1 per classifier.
    clf_res_f1: np.ndarray, shape (n_folds, n_classifiers)
        Raw F1 per fold per classifier.
    mean_kappa: np.ndarray, shape (n_classifiers,)
        Mean Kappa per classifier.
    std_kappa: np.ndarray, shape (n_classifiers,)
        Std Kappa per classifier.
    clf_res_kappa: np.ndarray, shape (n_folds, n_classifiers)
        Raw Kappa per fold per classifier.
    """
    # preprocessing
    X = X.copy().astype(float)
    X[np.isnan(X)] = 1
    X[np.isinf(X)] = 1

    # shuffle
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
    p = np.random.permutation(X.shape[0])
    X_s = X[p]
    y_s = y[p]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=cv_random_state)
    n_folds = n_splits * n_repeats
    clf_res_ba = np.zeros((n_folds, len(BASE_CLFS)))
    clf_res_f1 = np.zeros((n_folds, len(BASE_CLFS)))
    clf_res_kappa = np.zeros((n_folds, len(BASE_CLFS)))

    for fold_id, (train, test) in enumerate(rskf.split(X_s, y_s)):
        for clf_id, (name, base_clf) in enumerate(BASE_CLFS):
            clf  = clone(base_clf)
            pred = clf.fit(X_s[train], y_s[train]).predict(X_s[test])
            clf_res_ba[fold_id, clf_id] = balanced_accuracy_score(y_s[test], pred)
            clf_res_f1[fold_id, clf_id] = f1_score(y_s[test], pred, average='macro', zero_division=0)
            clf_res_kappa[fold_id, clf_id] = cohen_kappa_score(y_s[test], pred)
    
    
    mean_ba = np.mean(clf_res_ba, axis=0)
    std_ba = np.std(clf_res_ba, axis=0)
    mean_f1 = np.mean(clf_res_f1, axis=0)
    std_f1 = np.std(clf_res_f1, axis=0)
    mean_kappa = np.mean(clf_res_kappa, axis=0)
    std_kappa = np.std(clf_res_kappa, axis=0)

    return mean_ba, std_ba, clf_res_ba, mean_f1, std_f1, clf_res_f1, mean_kappa, std_kappa, clf_res_kappa


# def print_results(mean_ba, std_ba, label=''):
#     clf_names = [name for name, _ in BASE_CLFS]
#     if label:
#         print(f"\n{label}")
#     print(f"{'Classifier':<10s} {'Mean BA':>10s} {'Std BA':>8s}")
#     print('-' * 30)
#     for clf_id, name in enumerate(clf_names):
#         print(f"{name:<10s} {mean_ba[clf_id]:>10.4f} "
#               f"{std_ba[clf_id]:>8.4f}")