# characterize_streams.py
# ============================================================================
# Characterising the streams
# ----------------------------------------------------------------------------
# A family-agnostic view of *what changes in the raw data* when a concept
# changes, across every stream used in this work. This is the data-side half of
# the two-perspective description: how far the data itself moves between concepts
# (what a label-free vanilla baseline can see) versus what the meta-features must
# recover.
#
# Three views, produced for every stream/cell:
#
#   1. Fingerprint heatmap   -- per-concept feature means (concept x feature).
#                               Each row is a concept's signature in feature
#                               space; a flat row means the data does not move.
#
#   2. Distance / separability -- pairwise Euclidean distance between the
#                               per-concept mean vectors (matrix + min/max/mean
#                               summary). Large distances => concepts are
#                               separable from raw feature values alone.
#
#   3. Spread curve          -- for streams with a parametric sweep (stream-learn
#                               n_informative), how separation grows with the
#                               parameter.
#
# Expected pattern across this work's generators:
#   stream-learn (1c, 2) : data moves a lot        (L2 ~ 0.8 .. 2.7)  P(X) shifts
#   river SEA / STAGGER  : data barely moves        (L2 ~ 0.002 .. 0.04)
#   river LED            : data moves partially     (L2 ~ 0.42 .. 0.73)
#   real INSECTS / SPAM  : genuine movement
#
# ----------------------------------------------------------------------------
# INPUT
#   The concept_feature_means_exp{N}.csv files each experiment's analysis script
#   writes under its --concept_dist_features flag. Columns:
#       cell, [<sweep cols, e.g. n_informative>], concept, f0, f1, ..., fK
#   The script auto-discovers them under results/experiment_*/ , or takes
#   explicit paths with --csv.
#
# OUTPUT (default: results/generator_characterisation/)
#   fingerprint_<exp>_<cell>.png
#   distance_<exp>_<cell>.png
#   spread_<exp>_<series>.png                (only where a numeric sweep exists)
#   concept_separation_all_generators.csv    (one row per cell -> tab:concept_separation)
#
# REAL STREAMS (Exp 5): there is no generator to introspect, so the concept-mean
# matrix must be built from the data. Use `concept_means_from_windows(...)` in
# your Exp 5 pipeline to write a concept_feature_means_exp5.csv in the same
# format, then this script picks it up like any other. See build_exp5_hook().
# ============================================================================
 
import argparse
import csv
import sys
import glob
import itertools
import os
import re
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
FEATURE_RE = re.compile(r"^f\d+$")
 
 
# ============================================================================
#  CORE -- everything here operates on a concept-mean matrix `cm` of shape
#  (n_concepts, n_features). Nothing below knows or cares which generator it
#  came from, which is what makes the three views generalise.
# ============================================================================
 
def pairwise_l2(cm):
    """Pairwise Euclidean distances between concept mean-vectors.
    Returns (D (n x n symmetric matrix), pairs [(i, j, l2), ...])."""
    n = len(cm)
    D = np.zeros((n, n))
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        d = float(np.linalg.norm(cm[i] - cm[j]))
        D[i, j] = D[j, i] = d
        pairs.append((i, j, d))
    return D, pairs
 
 
def distance_summary(cm):
    """min / max / mean of the pairwise concept distances."""
    _, pairs = pairwise_l2(cm)
    d = np.array([p[2] for p in pairs]) if pairs else np.array([0.0])
    return dict(n_concepts=len(cm),
                l2_min=round(float(d.min()), 4),
                l2_max=round(float(d.max()), 4),
                l2_mean=round(float(d.mean()), 4))
 
 
def movement_verdict(l2_mean):
    """Soft label for how much the raw data moves between concepts. The
    thresholds are illustrative, taken from the measured spread across this
    work's own generators (river SEA/STAGGER ~0.002-0.04, LED ~0.42-0.73,
    stream-learn ~0.8-2.7). They are a reading aid, not a claim."""
    if l2_mean < 0.05:
        return "data does not move (drift in P(y|X) only)"
    if l2_mean < 0.5:
        return "data moves partially"
    return "data moves (drift visible in P(X))"
 
 
# ============================================================================
#  FIGURES
# ============================================================================
 
def plot_fingerprint(cm, concept_ids, title, out_path, centre=True,
                     max_features=60):
    """Per-concept feature means: concept (rows) x feature (cols).
 
    centre=True subtracts each feature's mean across concepts before plotting.
    The resulting heatmap shows how each concept differs from the average
    concept for that feature.

    Colour interpretation:
        red   -> higher than the average concept
        blue  -> lower than the average concept
        white -> approximately average

    This removes absolute feature levels and highlights the concept-specific
    patterns that would otherwise be difficult to see when features are not
    naturally centred (e.g. SEA, STAGGER, LED).
    This is essential for generators whose features are not centred on zero
    (SEA on [0,10], STAGGER on [0,2], LED on [0,1]): plotting raw means there
    puts every cell at the top of a symmetric colour scale and the plot comes
    out a uniform block, hiding the very variation it is meant to show.
    The colourbar range is what carries the magnitude -- a centred SEA plot
    shows structure, but at +/-0.02 against LED's +/-0.3.
    """
    n_c, n_f = cm.shape
    M = cm - cm.mean(axis=0, keepdims=True) if centre else cm
    # Wide streams (SPAM has 499 features) are unreadable at one column per
    # feature. Keep the most concept-discriminative ones -- those whose mean
    # varies most across concepts -- which is the same top-N-by-variance
    # reduction analysis_5.py uses for its per-feature relevance plots.
    kept = None
    if max_features and n_f > max_features:
        kept = np.argsort(M.std(axis=0))[::-1][:max_features]
        kept = np.sort(kept)
        M = M[:, kept]
        n_f = M.shape[1]
    vmax = float(np.abs(M).max()) or 1.0
    fig, ax = plt.subplots(figsize=(max(6, n_f * 0.35), max(3, n_c * 0.3)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("feature")
    ax.set_ylabel("concept")
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(concept_ids, fontsize=7)
    step = max(1, n_f // 40)
    ax.set_xticks(range(0, n_f, step))
    labels = (kept[::step] if kept is not None else range(0, n_f, step))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    if kept is not None:
        ax.set_xlabel(f"feature (top {n_f} of {cm.shape[1]} by variation "
                      f"across concepts)")
    lab = ("difference from average concept" if centre else "mean feature value")
    if centre:
        ax.set_title(
            f"Concept fingerprints in raw feature space\n"
            f"{title}\n"
            f"(feature values relative to the average concept)",
            fontsize=10
        )
    else:
        ax.set_title(
            f"Concept fingerprints in raw feature space\n"
            f"{title}",
            fontsize=10
        )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(lab, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
 
 
def plot_distance_matrix(cm, concept_ids, title, out_path):
    """Heatmap of the pairwise concept-distance matrix.

    Fixes vs the original:
      1. The zero diagonal is masked out (np.nan) so it no longer anchors the
         colour scale at 0. Previously imshow coloured the diagonal zeros, which
         stretched the scale downward and made genuinely different off-diagonal
         distances look similar. Now vmin/vmax are set from the off-diagonal
         values only, so cell colour tracks the actual spread of distances.
      2. Annotation text colour is chosen from the SAME normalised scale that
         drives the cell colour, and the caption states it is a legibility
         choice — so a reader no longer misreads black/white text as encoding
         magnitude. Magnitude is the cell colour / colourbar, full stop.
    """
    D, _ = pairwise_l2(cm)
    n = len(cm)

    # Mask the diagonal so it neither colours nor anchors the scale.
    Dm = D.astype(float).copy()
    np.fill_diagonal(Dm, np.nan)

    # Scale from off-diagonal values only.
    finite = Dm[np.isfinite(Dm)]
    if finite.size == 0:                       # 1 concept, or all-zero distances
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:                       # all pairwise distances equal
            vmin, vmax = vmin - 0.5, vmax + 0.5

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")            # masked diagonal shows grey

    fig, ax = plt.subplots(figsize=(max(4, n * 0.35), max(4, n * 0.35)))
    im = ax.imshow(Dm, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(concept_ids, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(concept_ids, fontsize=6)
    ax.set_title(f"Concept separation in raw feature space\n{title}", fontsize=10)

    if n <= 12 and finite.size > 0:
        span = vmax - vmin
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                norm = (D[i, j] - vmin) / span if span > 0 else 0.5
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if norm < 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("concept separation", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
 
 
def plot_spread_curve(sweep_name, sweep_vals, summaries, title, out_path):
    """Concept separation vs a numeric sweep parameter (e.g. n_informative):
    mean pairwise L2 with a min-max band."""
    order = np.argsort(sweep_vals)
    x = np.array(sweep_vals)[order]
    mean = np.array([summaries[i]["l2_mean"] for i in order])
    lo = np.array([summaries[i]["l2_min"] for i in order])
    hi = np.array([summaries[i]["l2_max"] for i in order])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, lo, hi, alpha=0.2, color="steelblue", label="min-max range")
    ax.plot(x, mean, "o-", color="steelblue", label="mean pairwise L2")
    ax.set_xlabel(sweep_name)
    ax.set_ylabel(" mean concept separation (L2)")
    ax.set_title(f"Concept separation vs {sweep_name}\n{title}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
 
 
# ============================================================================
#  ADAPTERS -- build a concept-mean matrix for each stream family
# ============================================================================
 
def concept_means_from_windows(windows, labels):
    """Build (concept_ids, cm) from raw data -- for real / on-the-fly streams.
 
    windows : list of per-window blocks. Either (n_inst x n_feat) instance
              blocks, or already-reduced per-window mean vectors (n_feat,).
    labels  : concept label per window (positional segment id for real streams).
 
    Returns (concept_ids, cm) where cm[k] is the mean feature vector of the
    windows carrying concept k -- the same quantity the synthetic
    concept_feature_means CSVs store, so the three views apply unchanged.
    """
    labels = np.asarray(labels)
    per_win = np.array([
        np.asarray(w, float).mean(axis=0) if np.ndim(w) == 2 else np.asarray(w, float)
        for w in windows
    ])
    ids = sorted(np.unique(labels).tolist())
    cm = np.array([per_win[labels == c].mean(axis=0) for c in ids])
    return ids, cm
 
 
def load_concept_means_csv(path):
    """Read a concept_feature_means_exp{N}.csv written in either:
    - comma-separated / dot-decimal format, or
    - semicolon-separated / comma-decimal format.
    """
    # Try European spreadsheet format first
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=";", decimal=",")
    except Exception:
        # fallback to the old format
        df = pd.read_csv(path)

    if df.empty:
        return {}

    fcols = sorted((c for c in df.columns if FEATURE_RE.match(c)),
                   key=lambda c: int(c[1:]))

    sweep_cols = [c for c in df.columns if c not in ("cell", "concept") and c not in fcols]

    out = {}
    for cell, g in df.groupby("cell"):
        g = g.sort_values("concept")
        ids = g["concept"].astype(int).tolist()

        cell_fcols = [c for c in fcols if g[c].notna().all()]
        cm = g[cell_fcols].astype(float).to_numpy()

        sweep = {c: g.iloc[0][c] for c in sweep_cols}
        out[cell] = (ids, cm, sweep)

    return out
 
 
# ============================================================================
#  DRIVER
# ============================================================================
 
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
 
 
def _drift_of(cell):
    for k in ("sudden", "gradual", "abrupt", "incgradual"):
        if k in cell:
            return k
    return "all"
 
 
def characterise_csv(path, all_cells=False, fig_dir_override=None,
                     centre_fp=True):
    """Produce the three views for one concept_feature_means CSV and return the
    per-cell separation rows for the consolidated table.
 
    Figures go to the experiment's own results/experiment_X/figures/streams/
    (derived from the CSV's location) unless fig_dir_override is given."""
    exp = re.search(r"(exp\w+)", os.path.basename(path))
    exp = exp.group(1) if exp else os.path.splitext(os.path.basename(path))[0]
    cells = load_concept_means_csv(path)
    if not cells:
        print(f"  [{exp}] empty / unreadable: {path}")
        return []
 
    fig_dir = fig_dir_override or os.path.join(
        os.path.dirname(os.path.abspath(path)), "figures", "streams")
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
 
    # ---- prefer the replication-AVERAGED distances when they exist ---------
    # The fingerprint CSV holds replication 0 only (a cross-rep average of
    # concept means is meaningless -- see the header note), so distances
    # recomputed from it are rep-0 values. concept_distance_summary_expN.csv
    # carries the properly rep-averaged min/max/mean, so the figures and the
    # consolidated table agree with the numbers quoted in the report.
    summary_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                f"concept_distance_summary_{exp}.csv")
    averaged = {}
    if os.path.exists(summary_path):
        try:
            import pandas as pd
            summ_df = pd.read_csv(summary_path, sep=";", decimal=",")
        except Exception:
            summ_df = pd.read_csv(summary_path)

        for _, r in summ_df.iterrows():
            try:
                averaged[str(r["cell"])] = dict(
                    l2_min=float(r["l2_min"]),
                    l2_max=float(r["l2_max"]),
                    l2_mean=float(r["l2_mean"]),
                    l2_mean_std=float(r.get("l2_mean_std", np.nan)),
                    n_reps=int(r.get("n_reps", 1))
                )
            except (KeyError, ValueError, TypeError):
                continue


        if averaged:
            print(f"  [{exp}] using replication-averaged distances from "
                  f"{os.path.basename(summary_path)}")
 
    # ---- fingerprint + distance matrix, per cell (or one representative) ----
    cell_names = sorted(cells)
    to_plot = cell_names if all_cells else [_representative(cell_names)]
    for cell in cell_names:
        ids, cm, sweep = cells[cell]
        summ = distance_summary(cm)
        if cell in averaged:                     # rep-averaged overrides rep-0
            summ.update(averaged[cell])
        rows.append(dict(experiment=exp, cell=cell,
                         n_features=cm.shape[1], **summ,
                         verdict=movement_verdict(summ["l2_mean"]), **sweep))
        if cell in to_plot:
            plot_fingerprint(cm, ids, f"{exp} / {cell}",
                             os.path.join(fig_dir, f"fingerprint_{exp}_{cell}.png"),
                             centre=centre_fp)
            plot_distance_matrix(cm, ids, f"{exp} / {cell}",
                                 os.path.join(fig_dir, f"distance_{exp}_{cell}.png"))
    print(f"  [{exp}] {len(cell_names)} cell(s); "
          f"L2 mean {min(r['l2_mean'] for r in rows):.3f}"
          f"..{max(r['l2_mean'] for r in rows):.3f}  ->  {fig_dir}")
 
    # ---- prefer the replication-AVERAGED distances when available ----------
    # The fingerprint CSV holds replication 0 only, so distances recomputed from
    # it are a single draw. concept_distance_summary_expN.csv holds the values
    # averaged over all replications. On noisy cells the two differ materially
    # (exp2 ninf3_gradual: 0.39 from rep 0 against 0.64 averaged), so the table
    # and the spread curve use the averaged numbers whenever the summary exists.
    summ_path = os.path.join(
        os.path.dirname(os.path.abspath(path)),
        os.path.basename(path).replace("concept_feature_means",
                                       "concept_distance_summary"))
    if os.path.exists(summ_path):
        try:
            import pandas as pd
            summ_df = pd.read_csv(summ_path, sep=";", decimal=",")
        except Exception:
            summ_df = pd.read_csv(summ_path)

        by_cell = {str(r["cell"]): r for _, r in summ_df.iterrows()}

        n_over = 0
        for r in rows:
            src = by_cell.get(r["cell"])
            if src is None:
                continue
            for k in ("l2_min", "l2_max", "l2_mean", "l2_mean_std"):
                if k in src and src[k] not in (None, ""):
                    r[k] = float(src[k])
            if "n_reps" in src and src["n_reps"] not in (None, ""):
                r["n_reps"] = int(src["n_reps"])
            r["verdict"] = movement_verdict(r["l2_mean"])
            n_over += 1

        if n_over:
            print(f"  [{exp}] using replication-averaged distances "
                f"for {n_over} cell(s)")
 
    # ---- spread curve, if a numeric sweep column exists ----
    sweep_cols = [c for c in rows[0] if c not in
                  ("experiment", "cell", "n_features", "n_concepts",
                   "l2_min", "l2_max", "l2_mean", "verdict",
                   "l2_mean_std", "n_reps")]   # stats, not sweep axes
    for sc in sweep_cols:
        if all(_num(r.get(sc)) is not None for r in rows):
            # one curve per drift family so sudden/gradual don't get mixed
            series = {}
            for r in rows:
                series.setdefault(_drift_of(r["cell"]), []).append(r)
            for name, rs in series.items():
                if len({_num(r[sc]) for r in rs}) < 2:
                    continue
                vals = [_num(r[sc]) for r in rs]
                plot_spread_curve(
                    sc, vals, rs, f"{exp} / {name}",
                    os.path.join(fig_dir, f"spread_{exp}_{name}_{sc}.png"))
            print(f"  [{exp}] spread curve over '{sc}'")
 
    return rows
 
 
def _representative(cell_names):
    """Pick one cell for the per-cell figures when --all-cells is off. Prefer a
    'sudden'/'abrupt' baseline-looking cell, else the first."""
    for pref in ("chunk100", "ninf10", "sudden", "abrupt"):
        hits = sorted(c for c in cell_names if pref in c)
        if hits:
            for h in hits:               # prefer a sudden/abrupt variant among them
                if "sudden" in h or "abrupt" in h:
                    return h
            return hits[0]
    return sorted(cell_names)[0]
 
 
def write_consolidated(rows, out_dir):
    """One row per cell across all generators -> feeds tab:concept_separation."""
    if not rows:
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "concept_separation_all_generators.csv")

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}".replace(".", ",")
        return v

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r.get(k, "")) for k in fields])

    print(f"\n  Consolidated separation table -> {path}")

    print(f"\n  {'experiment':10s} {'cell':26s} {'n_feat':>6s} {'n_conc':>6s} "
          f"{'L2 mean':>8s}  verdict")
    for r in sorted(rows, key=lambda r: r["l2_mean"]):
        print(f"  {r['experiment']:10s} {r['cell'][:26]:26s} {r['n_features']:6d} "
              f"{r['n_concepts']:6d} {r['l2_mean']:8.3f}  {r['verdict']}")
 
 
# ============================================================================
#  EXPERIMENT 5 -- real streams
# ----------------------------------------------------------------------------
# Real streams have no generator to introspect, so the concept-mean matrix is
# built from the data itself. Concepts are POSITIONAL, exactly as Experiment 5
# defines them: a window's concept label is the number of annotated drift
# boundaries passed up to that window.
#
# Boundaries are looked up from streams/generate_real_streams.py under several
# likely attribute names; if none is found the script falls back to the table
# below and says so loudly. INSECTS boundaries come from Souza et al. (Table 2)
# and MUST be filled in (or exposed by the module) -- they are stream-specific
# and are not guessed here. SPAM uses the imposed instants from Yu et al.
# ============================================================================
 
# Boundaries are read from the ground-truth files generate_real_streams.py
# already writes: data/real/annotated_streams_gt/{stream}.npy holds the drift
# CHUNK indices (raw instance change points // CHUNK_SIZE). That is the single
# source of truth -- the change points themselves live inside that script's
# __main__ block and are not importable, so nothing is hardcoded here.
#
# Note the saved features are min-max normalised to [0,1], so Exp 5 concept
# distances are on a comparable scale across INSECTS and SPAM.
 
REAL_CHUNK_SIZE = 100          # CHUNK_SIZE in generate_real_streams.py
 
 
def build_exp5(root, chunk_size=REAL_CHUNK_SIZE):
    """Write concept_feature_means / concept_distances / concept_distance_summary
    _exp5.csv from the real streams, in the same format as the synthetic ones."""
    stream_dir = os.path.join(root, "data", "real", "annotated_streams")
    gt_dir = os.path.join(root, "data", "real", "annotated_streams_gt")
 
    # Quietly skip when the real streams are absent (e.g. running locally
    # rather than on the cluster) -- the synthetic experiments still run.
    if not os.path.isdir(stream_dir) or not os.path.isdir(gt_dir):
        print("  [exp5] real streams not found on disk -- skipping "
              "(run generate_real_streams.py on the cluster first).\n")
        return
 
    if root not in sys.path:
        sys.path.insert(0, root)          # repo root, so `streams.` resolves
    try:
        import streams.generate_real_streams as grs
    except ImportError as e:
        print(f"  [exp5] cannot import generate_real_streams ({e}) -- skipping.\n")
        return
    out_dir = os.path.join(root, "results", "experiment_5")
    os.makedirs(out_dir, exist_ok=True)
 
    names = grs.REAL_STREAMS
    names = list(names.keys()) if isinstance(names, dict) else list(names)
 
    print("=" * 74)
    print(f"BUILDING EXP 5 CONCEPT MEANS  (chunk_size={chunk_size})")
    print("=" * 74)
 
    rows_means, rows_dist, rows_summary = [], [], []
    for name in names:
        path = os.path.join(stream_dir, f"{name}.npy")
        if not os.path.exists(path):
            print(f"  MISSING stream: {path}")
            continue
        gt_path = os.path.join(gt_dir, f"{name}.npy")
        if not os.path.exists(gt_path):
            print(f"  !! {name}: missing ground truth {gt_path} -- SKIPPED. "
                  f"Run generate_real_streams.py first.")
            continue
        drift_chunks = sorted(int(c) for c in np.load(gt_path).tolist())
 
        data = np.load(path)
        X = data[:, :-1].astype(float)
        n_win = len(X) // chunk_size
        if n_win == 0:
            print(f"  !! {name}: fewer than one full window -- SKIPPED.")
            continue
 
        win_means, labels = [], []
        for w in range(n_win):
            start = w * chunk_size
            win_means.append(X[start:start + chunk_size].mean(axis=0))
            # positional concept = number of drift CHUNKS passed, matching
            # evaluate_concept_classification_5.py / analysis_5.py
            labels.append(int(np.searchsorted(drift_chunks, w, side="right")))
 
        ids, cm = concept_means_from_windows(win_means, labels)
        summ = distance_summary(cm)
        print(f"  {name:32s} {cm.shape[1]:4d} feat  {len(ids):2d} concepts  "
              f"L2 mean {summ['l2_mean']:.4f}   drift_chunks={drift_chunks}")
 
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                rows_dist.append(dict(cell=name, concept_a=int(ids[i]),
                                      concept_b=int(ids[j]),
                                      l2=round(float(np.linalg.norm(cm[i] - cm[j])), 4)))
        rows_summary.append(dict(cell=name, **summ))
        for c, m in zip(ids, cm):
            rows_means.append(dict(cell=name, concept=int(c),
                                   **{f"f{k}": round(float(v), 4)
                                      for k, v in enumerate(m)}))
 
    if not rows_means:
        print("\n  Nothing built for Exp 5.")
        return
 
    for rows, base in ((rows_means, "concept_feature_means"),
                   (rows_dist, "concept_distances"),
                   (rows_summary, "concept_distance_summary")):
        path = os.path.join(out_dir, f"{base}_exp5.csv")

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}".replace(".", ",")
            return v

        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(fields)
            for r in rows:
                w.writerow([fmt(r.get(k, "")) for k in fields])

        print(f"  -> {path}")
 
 
def main():
    ap = argparse.ArgumentParser(description="Characterise the stream generators.")
    ap.add_argument("--csv", nargs="*", default=None,
                    help="explicit concept_feature_means_*.csv paths "
                         "(default: auto-discover under results/experiment_*/)")
    ap.add_argument("--out", default=None,
                    help="dir for the consolidated CSV (default: results/)")
    ap.add_argument("--figdir", default=None,
                    help="override: put ALL figures in this one dir "
                         "(default: each experiment's figures/streams/)")
    ap.add_argument("--representative", action="store_true",
                    help="draw only ONE representative cell per experiment "
                         "(default: every cell in every CSV)")
    ap.add_argument("--raw-fingerprints", action="store_true",
                    help="plot absolute concept means instead of deviations from "
                         "the per-feature mean across concepts (default: centred, "
                         "which is what makes non-zero-centred generators such as "
                         "SEA and STAGGER legible)")
    ap.add_argument("--skip-exp5", action="store_true",
                    help="do NOT rebuild the Exp 5 concept means from the real "
                         "streams (they are rebuilt automatically by default)")
    ap.add_argument("--chunk-size", type=int, default=REAL_CHUNK_SIZE,
                    help="window size for the Exp 5 build (default 100)")
    args = ap.parse_args()
 
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    table_dir = args.out or os.path.join(root, "results")
 
    # Exp 5 has no --concept_dist_features flag of its own: the real streams
    # are already on disk, so their concept means are built here, automatically,
    # every run. Silently skipped if the streams/ground truth are not present
    # (e.g. running off-cluster).
    if not args.skip_exp5 and not args.csv:
        try:
            build_exp5(root, chunk_size=args.chunk_size)
        except Exception as e:
            print(f"  [exp5] skipped: {type(e).__name__}: {e}\n")
 
    paths = args.csv or sorted(glob.glob(os.path.join(
        root, "results", "experiment_*", "concept_feature_means_exp*.csv")))
    if not paths:
        print("No concept_feature_means_exp*.csv found. Run each experiment's "
              "--concept_dist_features first (Exp 5 is built automatically "
              "from the real streams, if they are on disk).")
        return
 
    dest = args.figdir or "each experiment's results/experiment_X/figures/streams/"
    print(f"Characterising {len(paths)} generator file(s); figures -> {dest}\n")
    all_rows = []
    for p in paths:
        all_rows += characterise_csv(p, all_cells=not args.representative,
                                     fig_dir_override=args.figdir,
                                     centre_fp=not args.raw_fingerprints)
    write_consolidated(all_rows, table_dir)
 
 
if __name__ == "__main__":
    main()