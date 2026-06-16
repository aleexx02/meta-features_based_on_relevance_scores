# streams/generate_real_streams.py
# ============================================================
# First-time setup for all real-world streams used in Experiment 3.
#
# Part 1 - Annotated streams (Approach 1):
#   Converts CSV files from the USP DS Repository (Souza et al., 2020)
#   into .npy format with ground truth drift annotations.
#   Output: data/real/annotated_streams/
#           data/real/annotated_streams_gt/
#
# Part 2 - Unannotated streams (Approach 2):
#   Downloads datasets via River and generates proxy concept labels
#   from baseline classifier (GNB, KNN, HT) performance profiles
#   clustered with KMeans.
#   Output: data/real/unannotated_streams/
#
# Requirements:
#   Download USP DS Repository ZIP and unzip to ~/usp_ds_repository/
#           https://sites.google.com/view/uspdsrepository
#
# ============================================================

import os
import json
import collections
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from river import datasets, naive_bayes, neighbors, tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

USP_BASE    = os.path.expanduser('~/usp_ds_repository/USP DS Repository')
USP_OLD     = os.path.join(USP_BASE, 'Old datasets')
USP_INSECTS = os.path.join(USP_BASE, 'INSECTS')

OUT_ANNOTATED    = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams')
OUT_GT           = os.path.join(PROJECT_ROOT, 'data', 'real', 'annotated_streams_gt')
OUT_UNANNOTATED  = os.path.join(PROJECT_ROOT, 'data', 'real', 'unannotated_streams')

os.makedirs(OUT_ANNOTATED,   exist_ok=True)
os.makedirs(OUT_GT,          exist_ok=True)
os.makedirs(OUT_UNANNOTATED, exist_ok=True)


# ============================================================
#  PART 1 — ANNOTATED STREAMS
# ============================================================

# Ground truth drift chunk indices.
# INSECTS boundaries from Souza et al. (2020), verified against
# Komorniczak et al. (2024) real_streams_gt/ annotation files.
# electricity and covtype boundaries from Komorniczak et al. (2024).
# concept label = number of drift points passed up to chunk_idx.

DRIFT_CHUNKS_GT = {
    'INSECTS-abrupt_imbalanced':      np.array([125]),
    'INSECTS-gradual_imbalanced':     np.array([9, 60, 90, 125, 190]),
    'INSECTS-incremental_imbalanced': np.array([9, 35, 60, 180, 220]),
    'INSECTS-abrupt_balanced':        np.array([125]),
    'INSECTS-gradual_balanced':       np.array([9, 60, 90, 125, 190]),
    'INSECTS-incremental_balanced':   np.array([9, 35, 60, 180, 220]),
    'electricity':                    np.array([20, 38, 55, 115, 145]),
    'covtype':                        np.array([57, 121, 131, 155,
                                                205, 260, 295, 350]),
}

# INSECTS species ID → 0-indexed concept label
INSECTS_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 11: 4, 12: 5}

INSECTS_FILES = {
    'INSECTS-abrupt_imbalanced':      'INSECTS abrupt_imbalanced.csv',
    'INSECTS-gradual_imbalanced':     'INSECTS gradual_imbalanced.csv',
    'INSECTS-incremental_imbalanced': 'INSECTS incremental_imbalanced.csv',
    'INSECTS-abrupt_balanced':        'INSECTS abrupt_balanced.csv',
    'INSECTS-gradual_balanced':       'INSECTS gradual_balanced.csv',
    'INSECTS-incremental_balanced':   'INSECTS incremental_balanced.csv',
}


def minmax_normalise(X):
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    rng     = col_max - col_min
    rng[rng == 0] = 1.0
    return (X - col_min) / rng


def load_csv(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return [l.strip() for l in lines if l.strip()]


def already_done_annotated(stream_name):
    sp = os.path.join(OUT_ANNOTATED, f'{stream_name}.npy')
    gp = os.path.join(OUT_GT,        f'{stream_name}.npy')
    if os.path.exists(sp) and os.path.exists(gp):
        d = np.load(sp)
        print(f"  EXISTS : {stream_name}.npy  shape={d.shape}")
        return True
    return False


def save_annotated(stream_name, X, y, normalise=False):
    if normalise:
        X = minmax_normalise(X)
    result = np.hstack([X, y.reshape(-1, 1)])
    np.save(os.path.join(OUT_ANNOTATED, f'{stream_name}.npy'), result)
    counts = dict(collections.Counter(y.tolist()))
    print(f"  SAVED  : {stream_name}.npy  shape={result.shape}  "
          f"chunks@200={result.shape[0]//200}  classes={counts}")


def save_gt(stream_name):
    gt  = DRIFT_CHUNKS_GT[stream_name]
    np.save(os.path.join(OUT_GT, f'{stream_name}.npy'), gt)
    print(f"  GT     : drift_chunks={gt.tolist()}  "
          f"n_concepts={len(gt)+1}  "
          f"random_baseline={1/(len(gt)+1):.3f}")


# ---- INSECTS ----
print("\n" + "=" * 60)
print("PART 1 — Annotated streams")
print("=" * 60)

print("\n  INSECTS variants")
print("  " + "-" * 40)

for stream_name, csv_fname in INSECTS_FILES.items():
    print(f"\n  {stream_name}")
    if already_done_annotated(stream_name):
        continue
    csv_path = os.path.join(USP_INSECTS, csv_fname)
    if not os.path.exists(csv_path):
        print(f"  MISSING: {csv_path}"); continue
    rows = []
    for line in load_csv(csv_path):
        parts    = line.split(',')
        features = [float(x) for x in parts[:-1]]
        label    = INSECTS_LABEL_MAP[int(parts[-1])]
        rows.append(features + [label])
    data = np.array(rows, dtype=float)
    save_annotated(stream_name, data[:, :-1], data[:, -1].astype(int),
                   normalise=True)
    save_gt(stream_name)

# ---- Electricity ----
print(f"\n  electricity")
if not already_done_annotated('electricity'):
    csv_path = os.path.join(USP_OLD, 'Electricity.csv')
    rows = []
    for line in load_csv(csv_path):
        parts    = line.split(',')
        features = [float(x) for x in parts[:-1]]
        label    = 0 if parts[-1].strip() == '-1' else 1
        rows.append(features + [label])
    data = np.array(rows, dtype=float)
    save_annotated('electricity', data[:, :-1], data[:, -1].astype(int),
                   normalise=True)
    save_gt('electricity')

# ---- ForestCoverType ----
print(f"\n  covtype")
if not already_done_annotated('covtype'):
    csv_path = os.path.join(USP_OLD, 'ForestCoverType.csv')
    rows = []
    for line in load_csv(csv_path):
        parts    = line.split(',')
        features = [float(x) for x in parts[:-1]]
        label    = int(parts[-1].strip()) - 1   # remap 1-7 → 0-6
        rows.append(features + [label])
    data = np.array(rows, dtype=float)
    # already normalised in USP version
    save_annotated('covtype', data[:, :-1], data[:, -1].astype(int),
                   normalise=False)
    save_gt('covtype')


# ============================================================
#  PART 2 — UNANNOTATED STREAMS (proxy labels)
# ============================================================

# Only elec2 is included. Other River datasets were excluded:
#   phishing    — only 5 chunks (too short)
#   http        — single class (unclassifiable)
#   insects     — generic variant loads all features as categorical
#   credit_card — 336:1 imbalance (near random baseline)

DATASET_REGISTRY = {
    'elec2': datasets.Elec2(),
}

CHUNK_SIZE_PROXY  = 300
N_PROXY_CONCEPTS  = 4
RANDOM_STATE      = 42


def encode_df(df):
    drop_cols = [c for c in ['target', 't', 'window_id'] if c in df.columns]
    X_df = df.drop(columns=drop_cols).copy()
    encoded, feature_names, cat_idx = [], [], []
    for idx, col in enumerate(X_df.columns):
        s = X_df[col]
        feature_names.append(col)
        if pd.api.types.is_numeric_dtype(s):
            encoded.append(s.astype(float))
        else:
            encoded.append(s.astype('category').cat.codes.astype(float))
            cat_idx.append(idx)
    X = pd.concat(encoded, axis=1).values.astype(float)
    return X, df['target'].values, feature_names, cat_idx


def load_river_dataset(name, max_rows=None):
    rows = []
    for i, (x, y) in enumerate(DATASET_REGISTRY[name]):
        row = dict(x); row['target'] = y; row['t'] = i
        rows.append(row)
        if max_rows and len(rows) >= max_rows:
            break
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in ['target', 't']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if not df[col].mode().empty
                else 'missing')
    return df


def chunk_performance_profiles(X, y, chunk_size):
    clfs    = {'GNB': naive_bayes.GaussianNB(),
               'KNN': neighbors.KNNClassifier(),
               'HT':  tree.HoeffdingTreeClassifier()}
    classes = list(np.unique(y))
    n_chunks = int(np.ceil(len(X) / chunk_size))
    profiles = []
    for chunk_id in range(n_chunks):
        start = chunk_id * chunk_size
        end   = min((chunk_id + 1) * chunk_size, len(X))
        if start >= end:
            break
        stats = []
        for _, clf in clfs.items():
            correct = total = n_prob = 0
            loss_sum = prob_sum = 0.0
            for i in range(start, end):
                x_dict = {j: float(X[i, j]) for j in range(X.shape[1])}
                y_true = y[i]
                y_pred = clf.predict_one(x_dict) or classes[0]
                correct  += int(y_pred == y_true)
                total    += 1
                loss_sum += float(y_pred != y_true)
                if hasattr(clf, 'predict_proba_one'):
                    proba = clf.predict_proba_one(x_dict)
                    if isinstance(proba, dict):
                        prob_sum += float(proba.get(y_true, 0.0))
                        n_prob   += 1
                clf.learn_one(x_dict, y_true)
            stats.extend([correct / max(total, 1),
                           loss_sum / max(total, 1),
                           prob_sum / max(n_prob, 1) if n_prob else np.nan])
        profiles.append(stats)
    return np.asarray(profiles, dtype=float), list(clfs.keys())


def assign_proxy_labels(profiles, k=4, random_state=42):
    A = profiles.copy().astype(float)
    for j in range(A.shape[1]):
        mask = np.isnan(A[:, j])
        A[mask, j] = np.nanmean(A[:, j]) if np.any(~mask) else 0.0
    Xs = StandardScaler().fit_transform(A)
    km = KMeans(n_clusters=max(2, min(k, len(Xs))),
                random_state=random_state, n_init=20)
    raw = km.fit_predict(Xs)
    mapping, next_id, remapped = {}, 0, []
    for l in raw:
        if l not in mapping:
            mapping[l] = next_id; next_id += 1
        remapped.append(mapping[l])
    return np.asarray(remapped, dtype=int)


def already_done_proxy(dataset_name):
    d = os.path.join(OUT_UNANNOTATED, dataset_name)
    files = ['_ordered.npz', '_proxy_labels.npy',
             '_baseline_profiles.npz', '_inspection.json',
             '_ordered_meta.json']
    if all(os.path.exists(os.path.join(d, f'{dataset_name}{f}'))
           for f in files):
        print(f"  EXISTS : {dataset_name} — all proxy files present")
        return True
    return False


print("\n" + "=" * 60)
print("PART 2 — Unannotated streams (proxy labels)")
print("=" * 60)

for dataset_name in DATASET_REGISTRY:
    print(f"\n  {dataset_name}")
    if already_done_proxy(dataset_name):
        continue

    dataset_out = os.path.join(OUT_UNANNOTATED, dataset_name)
    os.makedirs(dataset_out, exist_ok=True)

    df = load_river_dataset(dataset_name)
    df['window_id'] = df.index // CHUNK_SIZE_PROXY
    X, y, feature_names, cat_idx = encode_df(df)

    # ordered stream
    np.savez(os.path.join(dataset_out, f'{dataset_name}_ordered.npz'),
             X=X, y=y)

    # performance profiles + proxy labels
    print(f"  Computing performance profiles...")
    profiles, clf_names = chunk_performance_profiles(X, y, CHUNK_SIZE_PROXY)
    proxy_labels = assign_proxy_labels(profiles, k=N_PROXY_CONCEPTS,
                                       random_state=RANDOM_STATE)

    np.savez(os.path.join(dataset_out, f'{dataset_name}_baseline_profiles.npz'),
             performance_profiles=profiles)
    np.save(os.path.join(dataset_out, f'{dataset_name}_proxy_labels.npy'),
            proxy_labels)

    label_dist = {str(k): int(v)
                  for k, v in zip(*np.unique(proxy_labels,
                                             return_counts=True))}
    print(f"  Proxy labels: {N_PROXY_CONCEPTS} concepts  dist={label_dist}")

    # metadata
    n_chunks = int(np.ceil(len(X) / CHUNK_SIZE_PROXY))
    metadata = {
        'dataset': dataset_name, 'source': 'river',
        'n_instances': int(len(X)), 'n_features': int(X.shape[1]),
        'feature_names': feature_names, 'categorical_indices': cat_idx,
        'chunk_size': int(CHUNK_SIZE_PROXY), 'n_chunks': n_chunks,
        'n_proxy_concepts': int(N_PROXY_CONCEPTS),
        'proxy_label_source':
            'GNB/KNN/HT performance profiles clustered with KMeans',
        'proxy_label_independent_of_meta_features': True,
        'baseline_classifiers': clf_names,
    }
    with open(os.path.join(dataset_out,
                           f'{dataset_name}_ordered_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    y_vals, y_cnts = np.unique(y, return_counts=True)
    inspection = {
        'dataset': dataset_name,
        'n_instances': int(len(X)), 'n_features': int(X.shape[1]),
        'class_distribution': {str(k): int(v)
                               for k, v in zip(y_vals, y_cnts)},
        'imbalance_ratio': float(y_cnts.max() / max(y_cnts.min(), 1)),
        'chunk_size': int(CHUNK_SIZE_PROXY), 'n_chunks': n_chunks,
        'n_proxy_concepts': int(N_PROXY_CONCEPTS),
        'proxy_label_distribution': label_dist,
    }
    with open(os.path.join(dataset_out,
                           f'{dataset_name}_inspection.json'), 'w') as f:
        json.dump(inspection, f, indent=2)

    print(f"  Saved to: {dataset_out}")


# ============================================================
#  VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)

print("\n  Annotated streams:")
all_ok = True
all_annotated = list(INSECTS_FILES.keys()) + ['electricity', 'covtype']
for stream_name in all_annotated:
    sp = os.path.join(OUT_ANNOTATED, f'{stream_name}.npy')
    gp = os.path.join(OUT_GT,        f'{stream_name}.npy')
    if os.path.exists(sp) and os.path.exists(gp):
        d = np.load(sp); g = np.load(gp)
        nc = len(g) + 1; rb = 1.0 / nc
        print(f"  OK  {stream_name}")
        print(f"      shape={d.shape}  chunks@200={d.shape[0]//200}  "
              f"concepts={nc}  random_baseline={rb:.3f}")
    else:
        print(f"  MISSING: {stream_name}")
        all_ok = False

print("\n  Unannotated streams:")
for dataset_name in DATASET_REGISTRY:
    d = os.path.join(OUT_UNANNOTATED, dataset_name)
    pl = os.path.join(d, f'{dataset_name}_proxy_labels.npy')
    if os.path.exists(pl):
        labels = np.load(pl)
        print(f"  OK  {dataset_name}  proxy_labels={labels.shape}  "
              f"concepts={len(np.unique(labels))}")
    else:
        print(f"  MISSING: {dataset_name}")
        all_ok = False

print()
print("ALL OK" if all_ok else "SOME FILES MISSING — check paths above")