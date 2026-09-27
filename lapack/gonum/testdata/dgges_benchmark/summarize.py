#!/usr/bin/env python3
"""Summarize paired DGGES samples without hiding individual case regressions."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
args = parser.parse_args()
root = args.results
samples = json.loads((root/'metadata.json').read_text())['samples']
pattern = re.compile(r'^BenchmarkDggesNetlib/(\S+)\s+\d+\s+([\d.]+) ns/op\s+(\d+) B/op\s+(\d+) allocs/op$')
data = {}
for revision in ['baseline', 'current']:
    for sample in range(1, samples+1):
        text = (root/f'{revision}-{sample}.txt').read_text()
        if not text.rstrip().endswith('PASS'):
            raise RuntimeError(f'incomplete run: {revision}-{sample}')
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                name, ns, size, allocs = match.groups()
                key, backend = name.rsplit('/', 1)
                data.setdefault((key, revision, backend), []).append((float(ns), int(size), int(allocs)))

rows = []
rng = random.Random(1729)
for key in sorted({key for key, _, _ in data}):
    entries = [data[key, 'baseline', 'Go'], data[key, 'current', 'Go'], data[key, 'current', 'Netlib']]
    if any(len(v) != samples for v in entries):
        raise RuntimeError(f'missing samples for {key}')
    old, new, ref = entries
    medians = [statistics.median(v[0] for v in group) for group in entries]
    ratios = [b[0]/a[0] for a, b in zip(old, new)]
    # Paired percentile bootstrap of the median ratio. Six short samples give
    # exploratory uncertainty estimates, not a guarantee against machine drift.
    boot = sorted(statistics.median(rng.choices(ratios, k=samples)) for _ in range(10000))
    rows.append([key, *medians, statistics.median(ratios), boot[249], boot[9749],
                 min(ratios), max(ratios), medians[1]/medians[2],
                 statistics.median(v[1] for v in old), statistics.median(v[1] for v in new),
                 statistics.median(v[2] for v in old), statistics.median(v[2] for v in new)])
with (root/'summary.csv').open('w') as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(['case', 'baseline_ns', 'current_ns', 'netlib_ns', 'paired_ratio',
                     'bootstrap_low', 'bootstrap_high', 'paired_min', 'paired_max',
                     'current_over_netlib', 'baseline_bytes', 'current_bytes',
                     'baseline_allocs', 'current_allocs'])
    writer.writerows(rows)
for sorting in ['false', 'true']:
    selected = [row for row in rows if f'sort={sorting}' in row[0]]
    ratio = math.exp(statistics.mean(math.log(row[4]) for row in selected))
    print(f'sort={sorting}: {len(selected)} cases; geometric mean paired ratio {ratio:.4f}')
for row in rows:
    if row[0].startswith(('n=100/', 'n=200/')) and '/vectors=right/sort=true' in row[0]:
        print(row[0], f'baseline={row[1]/1e6:.3f}ms current={row[2]/1e6:.3f}ms Netlib={row[3]/1e6:.3f}ms',
              f'ratio={row[4]:.3f} CI=[{row[5]:.3f},{row[6]:.3f}]', f'allocs={row[12]:g}->{row[13]:g}')
