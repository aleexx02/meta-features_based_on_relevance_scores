#!/usr/bin/env python3
"""
make_tables.py -- turn a summary_expN.csv (the combined per-cell summary) into
ready-to-paste LaTeX tables for the report/slides.

Usage:
    python make_tables.py summary_exp3.csv                 # writes tables_exp3.tex
    python make_tables.py summary_exp3.csv --out out.tex   # custom output name

What it emits, adapting to the file automatically:
  * PER-CLASSIFIER table (rows = GNB/KNN/HT/MLP): each method's balanced
    accuracy per classifier, with ReMF's best representation and Komorniczak's
    best group annotated. Row winner across the four methods is bolded.
    One such table per cell (per stream / per sweep point).
  * COST table (if the file has cost columns): ReMF vs Komorniczak time and
    peak memory, one row per distinct feature count.

Reads the European format the summary uses (';' delimiter, comma decimals).
Nothing here re-runs experiments; it only formats the CSV you already have.
"""
import csv, sys, argparse, os
from collections import OrderedDict

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_DIR_EXP1  = os.path.join(PROJECT_ROOT, 'results', 'experiment_1')
RESULTS_DIR_EXP2  = os.path.join(PROJECT_ROOT, 'results', 'experiment_2')
RESULTS_DIR_EXP3  = os.path.join(PROJECT_ROOT, 'results', 'experiment_3')
RESULTS_DIR_EXP4  = os.path.join(PROJECT_ROOT, 'results', 'experiment_4')
RESULTS_DIR_EXP5  = os.path.join(PROJECT_ROOT, 'results', 'experiment_5')

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

# classifier order as stored in the summary
CLFS = ['GNB', 'KNN', 'HT', 'MLP']
METHODS = ['ReMF', 'Komorniczak', 'vanilla', 'random']

def cell_label(row):
    """Human label for a cell, from whichever metadata the file carries."""
    bits = []
    for k in ('stream', 'n_informative', 'n_drifts', 'drift_type', 'chunk_size'):
        if k in row and row[k] not in ('', None):
            bits.append(f"{row[k]}" if k == 'stream' else f"{k.replace('_','')}={row[k]}")
    return ', '.join(bits) if bits else 'cell'

def per_clf_table(row):
    """One per-classifier LaTeX table for a single cell/row."""
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

def cost_table(rows):
    """One cost table across distinct feature counts, if cost columns exist."""
    if not any('ReMF time (ms/win)' in r for r in rows):
        return None
    seen = OrderedDict()
    for r in rows:
        nf = r.get('n_features')
        rt = parse_num(r.get('ReMF time (ms/win)'))
        rm = parse_num(r.get('ReMF peak MB (rel.)'))
        kt = parse_num(r.get('Komorniczak time (ms/win)'))
        km = parse_num(r.get('Komorniczak peak MB (rel.)'))
        if nf in seen or all(x is None for x in (rt, rm, kt, km)):
            continue
        if rt is None and kt is None:
            continue
        seen[nf] = (rt, rm, kt, km)
    if not seen:
        return None
    lines = []
    lines.append(r'\begin{table}[htbp]\centering')
    lines.append(r'\caption{Extraction cost by feature count: time per window and peak '
                 r'memory (relative, tracemalloc peak Python heap --- not process RSS). '
                 r'Lower is better; winner per row in \textbf{bold}.}')
    lines.append(r'\begin{tabular}{lllll}')
    lines.append(r'\toprule')
    lines.append(r'\# features & ReMF time (ms/win) & Komorniczak time & ReMF mem (rel.) & Komorniczak mem \\')
    lines.append(r'\midrule')
    for nf, (rt, rm, kt, km) in seen.items():
        def pick(a, b, va, vb):
            # bold the smaller (better) of the two
            fa = '--' if va is None else f'{va:.1f}'
            fb = '--' if vb is None else f'{vb:.1f}'
            if va is not None and vb is not None:
                if va < vb: fa = rf'\textbf{{{fa}}}'
                elif vb < va: fb = rf'\textbf{{{fb}}}'
            return fa, fb
        t_remf, t_kom = pick('t', 't', rt, kt)
        m_remf, m_kom = pick('m', 'm', rm, km)
        lines.append(f'{tex_escape(nf)} & {t_remf} & {t_kom} & {m_remf} & {m_kom} ' + r'\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)



def process_summary(csv_path, out_path):
    rows = read_rows(csv_path)

    if not rows:
        print(f'No rows in {csv_path}')
        return

    chunks = []
    chunks.append('% Auto-generated from ' + os.path.basename(csv_path))
    chunks.append('% Requires \\usepackage{booktabs}.\n')

    ct = cost_table(rows)
    if ct:
        chunks.append(ct + '\n')

    for r in rows:
        chunks.append(per_clf_table(r) + '\n')

    with open(out_path, 'w') as f:
        f.write('\n'.join(chunks))

    print(f'Wrote {out_path}')


def main():

    exp_dirs = [
        RESULTS_DIR_EXP1,
        RESULTS_DIR_EXP2,
        RESULTS_DIR_EXP3,
        RESULTS_DIR_EXP4,
        RESULTS_DIR_EXP5,
    ]

    for exp_id, exp_dir in enumerate(exp_dirs, start=1):

        csv_file = os.path.join(
            exp_dir,
            f"summary_exp{exp_id}.csv"
        )

        if not os.path.exists(csv_file):
            print(f"Missing: {csv_file}")
            continue

        out_file = os.path.join(
            exp_dir,
            f"tables_exp{exp_id}.tex"
        )

        process_summary(csv_file, out_file)


if __name__ == '__main__':
    main()