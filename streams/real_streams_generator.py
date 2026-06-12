# streams/real_streams_generator.py
# ===============
# Generate ordered real-data streams and chunk-level proxy concept labels.

# This version avoids CSV outputs and keeps only the files needed for the
# pipeline:
# - ordered stream NPZ (X, y)
# - ordered metadata JSON
# - inspection JSON
# - baseline performance profiles NPZ
# - proxy concept labels NPY

# Proxy labels are derived from baseline classifier performance profiles,

# Output files saved to data/real_streams_data/{dataset}/:
#   {dataset}_ordered.npz            X, y matrices
#   {dataset}_ordered_meta.json      pipeline metadata / reproducibility
#   {dataset}_inspection.json        dataset summary statistics
#   {dataset}_baseline_profiles.npz  chunk-level classifier performance
#   {dataset}_proxy_labels.npy       chunk-level proxy concept labels
#
# Usage:
#   python streams/real_streams_generator.py

import os
import json
import warnings

import numpy as np
import pandas as pd
from river import datasets, naive_bayes, neighbors, tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")



# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'real_streams_data')
MAX_ROWS = None # load full dataset
CHUNK_SIZE = 300
N_PROXY_CONCEPTS = 4
RANDOM_STATE = 42
INSPECT_ONLY = False



DATASET_REGISTRY = {
"elec2": datasets.Elec2(), # binary classification stream
}



# ============================================================
#  DATA LOADING AND CLEANING
# ============================================================
 
def load_river_dataset(name, max_rows=None):
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    rows = []
    for i, (x, y) in enumerate(DATASET_REGISTRY[name]):
        row = dict(x)
        row['target'] = y
        row['t']      = i
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    df = pd.DataFrame(rows)
    if 'target' not in df.columns:
        raise ValueError("Dataset did not produce a target column.")
    return df
 
 
def basic_cleaning(df):
    df = df.copy()
    for col in df.columns:
        if col in ['target', 't', 'window_id']:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if not df[col].mode().empty
                else 'missing')
    if df['target'].isna().any():
        df['target'] = df['target'].fillna(
            df['target'].mode().iloc[0] if not df['target'].mode().empty
            else 0)
    return df
 
 
def add_window_id(df, window_size):
    df = df.copy().reset_index(drop=True)
    df['window_id'] = df.index // window_size
    return df
 
 
def encode_for_abfs(df):
    drop_cols = [c for c in ['target', 't', 'window_id'] if c in df.columns]
    X_df = df.drop(columns=drop_cols).copy()
    encoded_cols, feature_names, categorical_indices = [], [], []
    for idx, col in enumerate(X_df.columns):
        s = X_df[col]
        feature_names.append(col)
        if pd.api.types.is_numeric_dtype(s):
            encoded_cols.append(s.astype(float))
        else:
            encoded_cols.append(
                s.astype('category').cat.codes.astype(float))
            categorical_indices.append(idx)
    if not encoded_cols:
        raise ValueError("No feature columns found.")
    X_encoded = pd.concat(encoded_cols, axis=1)
    X_encoded.columns = feature_names
    return X_encoded, df['target'].values, feature_names, categorical_indices
 
 
def df_to_xy(df):
    X_df, y, feature_names, categorical_indices = encode_for_abfs(df)
    return X_df.values.astype(float), y, feature_names, categorical_indices
 
 
def make_ordered_stream(df, chunk_size):
    return add_window_id(df, chunk_size)
 
 
def make_chunked_stream(df, chunk_size):
    df = add_window_id(df, chunk_size)
    return [
        df[df['window_id'] == wid].copy().reset_index(drop=True)
        for wid in sorted(df['window_id'].unique())
    ]
 
 
# ============================================================
#  BASELINE PERFORMANCE PROFILES
# ============================================================
 
def _make_baseline_clfs():
    return {
        'GNB': naive_bayes.GaussianNB(),
        'KNN': neighbors.KNNClassifier(),
        'HT':  tree.HoeffdingTreeClassifier(),
    }
 
 
def _chunk_performance_profiles(X, y, chunk_size):
    clfs    = _make_baseline_clfs()
    classes = list(np.unique(y))
    n_chunks = int(np.ceil(len(X) / chunk_size))
    profiles = []
 
    for chunk_id in range(n_chunks):
        start = chunk_id * chunk_size
        end   = min((chunk_id + 1) * chunk_size, len(X))
        if start >= end:
            break
        chunk_stats = []
        for _, clf in clfs.items():
            correct = total = n_prob = 0
            loss_sum = true_prob_sum = 0.0
            for i in range(start, end):
                x_dict = {j: float(X[i, j]) for j in range(X.shape[1])}
                y_true = y[i]
                y_pred = clf.predict_one(x_dict)
                if y_pred is None:
                    y_pred = classes[0]
                correct  += int(y_pred == y_true)
                total    += 1
                loss_sum += float(y_pred != y_true)
                if hasattr(clf, 'predict_proba_one'):
                    proba = clf.predict_proba_one(x_dict)
                    if isinstance(proba, dict):
                        true_prob_sum += float(proba.get(y_true, 0.0))
                        n_prob += 1
                clf.learn_one(x_dict, y_true)
            chunk_stats.extend([
                correct / max(total, 1),
                loss_sum / max(total, 1),
                true_prob_sum / max(n_prob, 1) if n_prob > 0 else np.nan,
            ])
        profiles.append(chunk_stats)
 
    return np.asarray(profiles, dtype=float), list(clfs.keys())
 
 
# ============================================================
#  PROXY CONCEPT LABELS
# ============================================================
 
def _impute_nan_with_colmean(A):
    A = A.copy().astype(float)
    for j in range(A.shape[1]):
        col  = A[:, j]
        mask = np.isnan(col)
        col[mask] = np.nanmean(col) if np.any(~mask) else 0.0
        A[:, j] = col
    return A
 
 
def _remap_labels_by_first_occurrence(labels):
    mapping, next_id, remapped = {}, 0, []
    for l in labels:
        if l not in mapping:
            mapping[l] = next_id; next_id += 1
        remapped.append(mapping[l])
    return np.asarray(remapped, dtype=int)
 
 
def assign_proxy_concept_labels(performance_profiles,
                                n_proxy_concepts=4, random_state=42):
    if len(performance_profiles) == 0:
        return np.asarray([], dtype=int), None
    X  = _impute_nan_with_colmean(performance_profiles)
    Xs = StandardScaler().fit_transform(X)
    k  = int(max(2, min(n_proxy_concepts, len(Xs))))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    return _remap_labels_by_first_occurrence(km.fit_predict(Xs)), km
 
 
# ============================================================
#  SUMMARY
# ============================================================
 
def summarize_dataset(df, dataset_name, chunk_size):
    X, y, feature_names, categorical_indices = df_to_xy(df)
    y_values, y_counts = np.unique(y, return_counts=True)
    return {
        'dataset':                dataset_name,
        'n_instances':            int(len(df)),
        'n_features':             int(X.shape[1]),
        'n_numeric_features':     int(X.shape[1] - len(categorical_indices)),
        'n_categorical_features': int(len(categorical_indices)),
        'categorical_indices':    categorical_indices,
        'class_distribution':     {str(k): int(v)
                                   for k, v in zip(y_values, y_counts)},
        'imbalance_ratio':        float(y_counts.max() / max(y_counts.min(), 1)),
        'n_missing_values_total': int(df.isna().sum().sum()),
        'chunk_size':             int(chunk_size),
        'n_chunks':               int(np.ceil(len(df) / chunk_size)),
        'feature_names':          feature_names,
    }
 
 
def print_summary(summary):
    print("\n" + "=" * 72)
    print(f"Dataset : {summary['dataset']}")
    print("=" * 72)
    print(f"  Instances            : {summary['n_instances']}")
    print(f"  Features             : {summary['n_features']} "
          f"({summary['n_numeric_features']} numeric, "
          f"{summary['n_categorical_features']} categorical)")
    print(f"  Chunk size           : {summary['chunk_size']}")
    print(f"  Number of chunks     : {summary['n_chunks']}")
    print(f"  Imbalance ratio      : {summary['imbalance_ratio']:.4f}")
    print(f"  Class distribution   : {summary['class_distribution']}")
    print(f"  Missing values       : {summary['n_missing_values_total']}")
    print("=" * 72)
 
 
# ============================================================
#  MAIN PIPELINE — one dataset
# ============================================================
 
def generate_real_stream(dataset_name, output_dir,
                         max_rows=None, chunk_size=300,
                         n_proxy_concepts=4, random_state=42):
    dataset_out = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_out, exist_ok=True)
 
    print(f"\n{'='*72}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*72}")
 
    df         = load_river_dataset(dataset_name, max_rows=max_rows)
    df         = basic_cleaning(df)
    ordered_df = make_ordered_stream(df, chunk_size=chunk_size)
    X, y, feature_names, categorical_indices = df_to_xy(ordered_df)
    chunks     = make_chunked_stream(df, chunk_size=chunk_size)
 
    paths = {
        'ordered_npz':       os.path.join(dataset_out, f'{dataset_name}_ordered.npz'),
        'ordered_meta':      os.path.join(dataset_out, f'{dataset_name}_ordered_meta.json'),
        'inspection':        os.path.join(dataset_out, f'{dataset_name}_inspection.json'),
        'baseline_profiles': os.path.join(dataset_out, f'{dataset_name}_baseline_profiles.npz'),
        'proxy_labels':      os.path.join(dataset_out, f'{dataset_name}_proxy_labels.npy'),
    }
 
    # ordered stream
    np.savez(paths['ordered_npz'], X=X, y=y)
    print(f"  Ordered stream saved : shape=({len(X)}, {X.shape[1]})")
 
    # baseline profiles + proxy labels
    print(f"  Computing baseline performance profiles...")
    performance_profiles, clf_names = _chunk_performance_profiles(
        X, y, chunk_size)
    proxy_labels, _ = assign_proxy_concept_labels(
        performance_profiles,
        n_proxy_concepts=n_proxy_concepts,
        random_state=random_state)
 
    np.savez(paths['baseline_profiles'],
             performance_profiles=performance_profiles)
    np.save(paths['proxy_labels'], proxy_labels)
 
    label_dist = {str(k): int(v)
                  for k, v in zip(*np.unique(proxy_labels,
                                             return_counts=True))}
    print(f"  Proxy labels saved   : {n_proxy_concepts} concepts  "
          f"dist={label_dist}")
 
    # metadata
    metadata = {
        'dataset':            dataset_name,
        'source':             'river',
        'n_instances':        int(len(ordered_df)),
        'n_features':         int(X.shape[1]),
        'feature_names':      feature_names,
        'categorical_indices':categorical_indices,
        'chunk_size':         int(chunk_size),
        'n_chunks':           int(len(chunks)),
        'max_rows':           None if max_rows is None else int(max_rows),
        'n_proxy_concepts':   int(n_proxy_concepts),
        'proxy_label_source':
            'baseline classifier (GNB, KNN, HT) performance profiles '
            'clustered with KMeans',
        'proxy_label_independent_of_meta_features': True,
        'baseline_classifiers': clf_names,
        'output_files': paths,
    }
    with open(paths['ordered_meta'], 'w') as f:
        json.dump(metadata, f, indent=2)
 
    # inspection
    inspection = summarize_dataset(ordered_df, dataset_name, chunk_size)
    inspection['n_proxy_concepts']         = int(n_proxy_concepts)
    inspection['proxy_label_source']       = metadata['proxy_label_source']
    inspection['baseline_classifiers']     = clf_names
    inspection['proxy_label_distribution'] = label_dist
    with open(paths['inspection'], 'w') as f:
        json.dump(inspection, f, indent=2)
 
    print_summary(inspection)
    print(f"  All files saved to: {dataset_out}")
    return paths
 
 
# ============================================================
#  ENTRY POINT — loop over all suitable datasets
# ============================================================
 
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    for dataset_name in DATASET_REGISTRY:
        generate_real_stream(
            dataset_name     = dataset_name,
            output_dir       = OUTPUT_DIR,
            max_rows         = MAX_ROWS,
            chunk_size       = CHUNK_SIZE,
            n_proxy_concepts = N_PROXY_CONCEPTS,
            random_state     = RANDOM_STATE,
        )
 
    print("\nAll datasets processed.")