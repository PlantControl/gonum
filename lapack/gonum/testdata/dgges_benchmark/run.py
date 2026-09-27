#!/usr/bin/env python3
"""Build identical DGGES harnesses and alternate baseline/current benchmark runs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', default='a78cf83eff160df0e423b08af0c7642ade964150')
parser.add_argument('--samples', type=int, default=6)
parser.add_argument('--benchtime', default='100ms')
parser.add_argument('--case', action='append', help='exact case such as n=10/dense/vectors=none/sort=true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.samples < 2:
    parser.error("at least two samples are required")
root = Path(__file__).resolve().parents[4]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, GOMAXPROCS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
harness = ['lapack/gonum/dgges_netlib_test.go', 'lapack/gonum/bench_test.go',
           'lapack/testlapack/dgges_bench.go', 'lapack/gonum/netlib_helpers_test.go',
           'lapack/gonum/internal/netlib/differential.go']

def capture(command):
    return subprocess.check_output(command, cwd=root, env=env, text=True).strip()

metadata = {'base': capture(['git', 'rev-parse', args.base]),
            'head': capture(['git', 'rev-parse', 'HEAD']),
            'go': capture(['go', 'version']), 'samples': args.samples,
            'benchtime': args.benchtime, 'cases': args.case,
            'threads': {k: env[k] for k in ['GOMAXPROCS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
            'cpu': capture(['sysctl', '-n', 'machdep.cpu.brand_string']),
            'blas': capture(['otool', '-L', '/opt/homebrew/opt/lapack/lib/libblas.dylib']),
            'lapack': capture(['otool', '-L', '/opt/homebrew/opt/lapack/lib/liblapack.dylib'])}
metadata['lapack_path'] = str(Path('/opt/homebrew/opt/lapack').resolve())
metadata['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
    [root/name for name in harness] +
    [root/'lapack/gonum'/name for name in ['dgges.go', 'dtgex2.go', 'dtgexc.go', 'dtgsen.go', 'dtgsy2.go']] +
    [Path('/opt/homebrew/opt/lapack/lib')/name for name in ['liblapack.dylib', 'liblapacke.dylib', 'libblas.dylib']]}
(out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
(out/'implementation.patch').write_text(capture(['git', 'diff', '--',
    'lapack/gonum/dgges.go', 'lapack/gonum/dtgex2.go', 'lapack/gonum/dtgexc.go',
    'lapack/gonum/dtgsen.go', 'lapack/gonum/dtgsy2.go'])+'\n')
with tempfile.TemporaryDirectory(prefix='dgges-compare-') as temp:
    temp = Path(temp)
    baseline = temp/'baseline'
    baseline.mkdir()
    archive = subprocess.check_output(['git', 'archive', args.base], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(baseline, filter='data')
    for name in harness:
        shutil.copy2(root/name, baseline/name)
    binaries = {}
    for name, source in [('baseline', baseline), ('current', root)]:
        binary = temp/(name+'.test')
        subprocess.run(['go', 'test', '-tags', 'netlib', '-c', '-o', str(binary), './lapack/gonum'], cwd=source, env=env, check=True)
        binaries[name] = binary
    # Verify the measured fixtures before collecting timings. The log also
    # records LAPACKE_ilaver, which can differ from the package version.
    for name, binary in binaries.items():
        with (out/f'{name}-oracle.txt').open('w') as log:
            subprocess.run([str(binary), '-test.run=^TestDggesNetlibControlPencils$', '-test.v'],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for sample in range(args.samples):
        order = ['baseline', 'current'] if sample % 2 == 0 else ['current', 'baseline']
        for name in order:
            patterns = ['^BenchmarkDggesNetlib$']
            if args.case:
                patterns = ['^BenchmarkDggesNetlib$/'+'/'.join('^'+part+'$' for part in case.split('/')) for case in args.case]
            print(f'Sample {sample+1}/{args.samples}: {name}', flush=True)
            with (out/f'{name}-{sample+1}.txt').open('w') as log:
                for pattern in patterns:
                    if name == 'baseline':
                        pattern += '/Go$' if args.case else '/.*/.*/.*/.*/Go$'
                    command = [str(binaries[name]), '-test.run=^$', '-test.bench='+pattern,
                               '-test.benchmem', '-test.benchtime='+args.benchtime,
                               '-test.cpu=1', '-test.count=1', '-test.timeout=30m']
                    subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
