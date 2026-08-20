# mf_extraction.py

import numpy as np

def extract_metafeatures(wt, wt_prev, drift_count, time_since_drift, threshold=0.05):
    """
    Extract meta-features from ReMF relevance state.
    
    Parameters:
        wt              : current relevance score vector [score_f1, ..., score_fd]
        wt_prev         : relevance score vector from previous window
        drift_count     : number of drift signals fired during this window
        time_since_drift: instances since last drift signal
        threshold       : minimum score to consider a feature relevant
    
    Returns:
        numpy array of 8 meta-features
    """
    wt = np.array(wt, dtype=float)
    wt_prev = np.array(wt_prev, dtype=float) if wt_prev is not None else wt.copy()

    # ── current state ──────────────────────────────────────────
    # entropy: how spread is relevance across features?
    # high entropy → relevance is evenly distributed (many features matter)
    # low entropy  → one feature dominates
    wt_norm = wt / (wt.sum() + 1e-10)
    entropy = float(-np.sum(wt_norm * np.log(wt_norm + 1e-10)))

    # number of features currently above relevance threshold
    n_relevant = int((wt > threshold).sum())

    # maximum relevance score — how dominant is the best feature?
    max_score = float(wt.max())

    # spread of relevance scores across features
    std_score = float(wt.std())

    # ── temporal change ────────────────────────────────────────
    # mean absolute change in relevance scores since last window
    # high value → concept is changing
    # low value  → concept is stable
    delta_mean = float(np.abs(wt - wt_prev).mean())

    # number of features that switched category
    # (relevant → irrelevant or irrelevant → relevant)
    n_changed = int(
        np.sum((wt > threshold) != (wt_prev > threshold))
    )

    # ── drift signals ──────────────────────────────────────────
    # how many stump drift detectors fired during this window
    # high value → many features changed relevance → strong drift
    drift_count = int(drift_count)

    # how many instances since the last drift signal
    # low value  → drift very recent
    # high value → concept has been stable for a while
    time_since_drift = int(time_since_drift)

    return np.array([entropy, n_relevant, max_score, std_score, delta_mean, n_changed, drift_count,
        time_since_drift], dtype=float)


MF_NAMES_AGGSTATS = ["entropy", "n_relevant", "max_score", "std_score", "delta_mean",
    "n_changed", "drift_count", "time_since_drift"]


# This is a simpler version that just returns the raw relevance scores as meta-features.
def extract_metafeatures_raw(wt, normalise=True):
    """
    Use the raw ReMF relevance score vector directly as meta-features.
    
    Parameters:
        wt        : current relevance score vector [score_f1, ..., score_fd]
        normalise : if True, divide by sum so scores are relative (sum to 1)
    
    Returns:
        numpy array of shape (d,)
    """
    wt = np.array(wt, dtype=float)
    if normalise:
        wt = wt / (wt.sum() + 1e-10)
    return wt

MF_NAMES_RAW = [f'r_f{j+1}' for j in range(10)]



# This is a simpler version that just returns the raw relevance scores as meta-features.
def extract_metafeatures_raw_temporal(wt, wt_prev=None, normalise=True):
    """
    Use the raw ReMF relevance score vector directly as meta-features.
    
    Parameters:
        wt: current relevance score vector [score_f1, ..., score_fd]
        wt_prev: previous relevance score vector
        normalise: if True, divide by sum so scores are relative (sum to 1)
    
    Returns:
        numpy array of shape (d,)
    """
    wt = np.array(wt, dtype=float)

    # temporal features — computed on raw scores before normalisation
    if wt_prev is not None:
        wt_prev = np.array(wt_prev, dtype=float)
        delta_mean = float(np.abs(wt - wt_prev).mean())
        norm_t    = np.linalg.norm(wt)
        norm_prev = np.linalg.norm(wt_prev)
        if norm_t > 1e-10 and norm_prev > 1e-10:
            cosine_sim = float(np.dot(wt, wt_prev) / (norm_t * norm_prev))
        else:
            cosine_sim = 1.0
    else:
        delta_mean = 0.0
        cosine_sim = 1.0

    # sum-normalise for the 10 score features
    total = wt.sum()
    wt_norm = wt / (total + 1e-10) if normalise else wt

    return np.concatenate([wt_norm, [delta_mean, cosine_sim]])


MF_NAMES_RAW_TEMPORAL = [f'r_f{j+1}' for j in range(10)] + ['delta_mean', 'cosine_sim']