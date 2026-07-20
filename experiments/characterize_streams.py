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
#  Three views, produced for every stream/cell:
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
 
def plot_fingerprint(cm, concept_ids, title, out_path):
    """Per-concept feature means: concept (rows) x feature (cols)."""
    n_c, n_f = cm.shape
    vmax = float(np.abs(cm).max()) or 1.0
    fig, ax = plt.subplots(figsize=(max(6, n_f * 0.35), max(3, n_c * 0.3)))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("feature")
    ax.set_ylabel("concept")
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(concept_ids, fontsize=7)
    step = max(1, n_f // 40)
    ax.set_xticks(range(0, n_f, step))
    ax.set_xticklabels(range(0, n_f, step), fontsize=6, rotation=90)
    ax.set_title(f"Concept fingerprints (per-concept feature means)\n{title}",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("mean feature value", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
 
 
def plot_distance_matrix(cm, concept_ids, title, out_path):
    """Heatmap of the pairwise concept-distance matrix."""
    D, _ = pairwise_l2(cm)
    n = len(cm)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.35), max(4, n * 0.35)))
    im = ax.imshow(D, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(concept_ids, fontsize=6, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(concept_ids, fontsize=6)
    ax.set_title(f"Pairwise concept distance (L2)\n{title}", fontsize=10)
    if n <= 12 and D.max() > 0:
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if D[i, j] < D.max() * 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("L2 distance", fontsize=8)
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
    ax.set_ylabel("concept separation (L2)")
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
    """Read a concept_feature_means_exp{N}.csv.
 
    Returns dict: cell -> (concept_ids, cm, sweep) where sweep is a dict of any
    non-{cell, concept, f*} columns for that cell (e.g. {'n_informative': '15'}).
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    fcols = sorted((c for c in rows[0] if FEATURE_RE.match(c)),
                   key=lambda c: int(c[1:]))
    sweep_cols = [c for c in rows[0] if c not in ("cell", "concept") and c not in fcols]
    grouped = {}
    for r in rows:
        grouped.setdefault(r["cell"], []).append(r)
    out = {}
    for cell, rs in grouped.items():
        rs = sorted(rs, key=lambda r: int(r["concept"]))
        ids = [int(r["concept"]) for r in rs]
        # feature columns actually populated for THIS cell -- generators within
        # one file can differ in feature count (SEA/STAGGER 3 vs LED 24), which
        # leaves the surplus f-columns empty on the low-dimensional cells.
        cell_fcols = [c for c in fcols
                      if all(str(r.get(c, "")).strip() != "" for r in rs)]
        cm = np.array([[float(r[c]) for c in cell_fcols] for r in rs])
        out[cell] = (ids, cm, {c: rs[0][c] for c in sweep_cols})
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
 
 
def characterise_csv(path, all_cells=False, fig_dir_override=None):
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
 
    # ---- fingerprint + distance matrix, per cell (or one representative) ----
    cell_names = sorted(cells)
    to_plot = cell_names if all_cells else [_representative(cell_names)]
    for cell in cell_names:
        ids, cm, sweep = cells[cell]
        summ = distance_summary(cm)
        rows.append(dict(experiment=exp, cell=cell,
                         n_features=cm.shape[1], **summ,
                         verdict=movement_verdict(summ["l2_mean"]), **sweep))
        if cell in to_plot:
            plot_fingerprint(cm, ids, f"{exp} / {cell}",
                             os.path.join(fig_dir, f"fingerprint_{exp}_{cell}.png"))
            plot_distance_matrix(cm, ids, f"{exp} / {cell}",
                                 os.path.join(fig_dir, f"distance_{exp}_{cell}.png"))
    print(f"  [{exp}] {len(cell_names)} cell(s); "
          f"L2 mean {min(r['l2_mean'] for r in rows):.3f}"
          f"..{max(r['l2_mean'] for r in rows):.3f}  ->  {fig_dir}")
 
    # ---- spread curve, if a numeric sweep column exists ----
    sweep_cols = [c for c in rows[0] if c not in
                  ("experiment", "cell", "n_features", "n_concepts",
                   "l2_min", "l2_max", "l2_mean", "verdict")]
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
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Consolidated separation table -> {path}")
    # compact printout
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
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  -> {path}")
    print()
 
 
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
                                     fig_dir_override=args.figdir)
    write_consolidated(all_rows, table_dir)
 
 
if __name__ == "__main__":
    main()