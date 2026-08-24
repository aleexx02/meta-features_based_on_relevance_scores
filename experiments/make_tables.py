"""
make_tables.py -- turn a summary_expN.csv (the combined per-cell summary) into
ready-to-paste LaTeX tables for the report/slides.

Usage (run from anywhere; pass the path to the CSV):
    python make_tables.py experiment_3/summary_exp3.csv
        -> writes experiment_3/tables_exp3.tex  (next to the CSV, by default)
    python make_tables.py experiment_3/summary_exp3.csv --out somewhere/foo.tex

Loop over all experiments from the experiments/ folder:
    for d in experiment_*; do python make_tables.py "$d"/summary_*.csv; done

What it emits, adapting to the file automatically:
  * PER-CLASSIFIER table (rows = GNB/KNN/HT/MLP): each method's balanced
    accuracy per classifier, with ReMF's best representation and Komorniczak's
    best group annotated. Row winner across the four methods is bolded.
    One such table per cell (per stream / per sweep point).
  * COST table (if the file has cost columns): ReMF / Komorniczak / vanilla
    time and peak memory, one row per distinct feature count.

Reads the European format the summary uses (';' delimiter, comma decimals).
Nothing here re-runs experiments; it only formats the CSV you already have.
"""
import csv, sys, argparse, os
from collections import OrderedDict

def parse_num(s):
    if s is None: return None
    s = s.strip().replace(',', '.')
    if s == '' or s.lower() == 'nan': return None
    try: return float(s)
    except ValueError: return None

def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))

def tex_escape(s):
    return (str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&'))

CLFS = ['GNB', 'KNN', 'HT', 'MLP']
METHODS = ['ReMF', 'Komorniczak', 'vanilla', 'random']

def cell_label(row):
    bits = []
    for k in ('stream', 'n_informative', 'n_drifts', 'drift_type', 'chunk_size'):
        if k in row and row[k] not in ('', None):
            bits.append(f"{row[k]}" if k == 'stream' else f"{k.replace('_','')}={row[k]}")
    return ', '.join(bits) if bits else 'cell'

def per_clf_table(row):
    rb = parse_num(row.get('random_baseline'))
    lines = []
    lines.append(r'\begin{table}[htbp]\centering')
    lines.append(rf'\caption{{Per-classifier balanced accuracy --- {tex_escape(cell_label(row))}. '
                 r'Best method per row in \textbf{bold}; ReMF representation / Komorniczak group in parentheses.}')
    lines.append(r'\begin{tabular}{lllll}')
    lines.append(r'\toprule')
    lines.append(r'Classifier & ReMF & Komorniczak & vanilla & random \\')
    lines.append(r'\midrule')
    for clf in CLFS:
        remf = parse_num(row.get(f'ReMF {clf} BA'))
        remf_rep = row.get(f'ReMF {clf} representation', '') or ''
        kom = parse_num(row.get(f'Komorniczak {clf} BA'))
        kom_grp = row.get(f'Komorniczak {clf} group', '') or ''
        van = parse_num(row.get(f'vanilla {clf} BA'))
        vals = {'ReMF': remf, 'Komorniczak': kom, 'vanilla': van, 'random': rb}
        present = {m: v for m, v in vals.items() if v is not None}
        winner = max(present, key=present.get) if present else None
        def fmt(m, v, ann=None):
            if v is None: return '--'
            s = f'{v:.3f}'
            if ann: s += f' ({tex_escape(ann)})'
            return rf'\textbf{{{s}}}' if m == winner else s
        lines.append(' & '.join([
            clf,
            fmt('ReMF', remf, remf_rep),
            fmt('Komorniczak', kom, kom_grp),
            fmt('vanilla', van),
            fmt('random', rb),
        ]) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

def _bold_min(vals, fmt='{:.1f}'):
    nums = [(i, v) for i, v in enumerate(vals) if v is not None]
    best_i = min(nums, key=lambda t: t[1])[0] if nums else None
    out = []
    for i, v in enumerate(vals):
        if v is None:
            out.append('--')
        else:
            s = fmt.format(v)
            out.append(rf'\textbf{{{s}}}' if i == best_i else s)
    return out

def cost_table(rows):
    if not any('ReMF time (ms/win)' in r for r in rows):
        return None
    has_vanilla = any('vanilla time (ms/win)' in r for r in rows)
    seen = OrderedDict()
    for r in rows:
        nf = r.get('n_features')
        rt = parse_num(r.get('ReMF time (ms/win)'))
        rm = parse_num(r.get('ReMF peak MB (rel.)'))
        kt = parse_num(r.get('Komorniczak time (ms/win)'))
        km = parse_num(r.get('Komorniczak peak MB (rel.)'))
        vt = parse_num(r.get('vanilla time (ms/win)')) if has_vanilla else None
        vm = parse_num(r.get('vanilla peak MB (rel.)')) if has_vanilla else None
        if nf in seen or all(x is None for x in (rt, rm, kt, km, vt, vm)):
            continue
        if rt is None and kt is None:
            continue
        seen[nf] = (rt, rm, kt, km, vt, vm)
    if not seen:
        return None
    ncol = 'lllllll' if has_vanilla else 'lllll'
    lines = []
    lines.append(r'\begin{table}[htbp]\centering')
    lines.append(r'\caption{Extraction cost by feature count: time per window and peak '
                 r'memory (relative, tracemalloc peak Python heap --- not process RSS). '
                 r'Lower is better; cheapest per row in \textbf{bold}. Vanilla is mean+std only.}')
    lines.append(r'\begin{tabular}{' + ncol + r'}')
    lines.append(r'\toprule')
    if has_vanilla:
        lines.append(r'\# features & ReMF time & Komorniczak time & vanilla time '
                     r'& ReMF mem & Komorniczak mem & vanilla mem \\')
    else:
        lines.append(r'\# features & ReMF time (ms/win) & Komorniczak time '
                     r'& ReMF mem (rel.) & Komorniczak mem \\')
    lines.append(r'\midrule')
    for nf, (rt, rm, kt, km, vt, vm) in seen.items():
        if has_vanilla:
            t_remf, t_kom, t_van = _bold_min([rt, kt, vt])
            m_remf, m_kom, m_van = _bold_min([rm, km, vm])
            lines.append(f'{tex_escape(nf)} & {t_remf} & {t_kom} & {t_van} '
                         f'& {m_remf} & {m_kom} & {m_van} ' + r'\\')
        else:
            t_remf, t_kom = _bold_min([rt, kt])
            m_remf, m_kom = _bold_min([rm, km])
            lines.append(f'{tex_escape(nf)} & {t_remf} & {t_kom} & {m_remf} & {m_kom} ' + r'\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rows = read_rows(args.csv)
    if not rows:
        print('No rows in', args.csv); return
    # default output: sits NEXT TO the input CSV, named tables_<stem-without-summary_>.tex
    if args.out:
        out = args.out
    else:
        d = os.path.dirname(os.path.abspath(args.csv))
        stem = os.path.splitext(os.path.basename(args.csv))[0].replace('summary_', '')
        out = os.path.join(d, f'tables_{stem}.tex')
    chunks = []
    chunks.append('% Auto-generated from ' + os.path.basename(args.csv) + ' by make_tables.py')
    chunks.append('% Requires \\usepackage{booktabs} in your preamble.\n')
    ct = cost_table(rows)
    if ct: chunks.append(ct + '\n')
    for r in rows:
        chunks.append(per_clf_table(r) + '\n')
    with open(out, 'w') as f:
        f.write('\n'.join(chunks))
    print(f'Wrote {out}: {len(rows)} per-classifier table(s)' + (' + 1 cost table' if ct else ''))

if __name__ == '__main__':
    main()