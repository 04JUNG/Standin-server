"""Reproducible offline two-case fitting sanity; never runs a paid provider."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--library', required=True)
    p.add_argument('--body', required=True)
    p.add_argument('--blender', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cases', nargs='+', default=['001', '074'])
    a = p.parse_args()
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    (root / 'data').symlink_to(Path(a.data).resolve(), target_is_directory=True)

    def command(args):
        proc = subprocess.run(args, cwd=repo, capture_output=True, text=True)
        with (root / 'execution.log').open('a') as f:
            f.write(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError('pilot step failed; see execution.log')

    def module(name, *args):
        command([sys.executable, '-m', 'experiments.refine_v5.' + name, *map(str, args)])

    def native(mode, cases, out):
        request = root / (mode + '.json')
        request.write_text(json.dumps(dict(mode=mode, repo=str(repo), body=a.body, out=str(out), cases=cases)))
        command([a.blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(repo / 'experiments/refine_v5/native_pilot.py'), '--', str(request)])
    module('reconstruct_seeds', '--data', a.data, '--library', a.library, '--out', root / 'seeds', '--cases', *a.cases)
    seeds = json.loads((root / 'seeds/seeds.json').read_text())
    native('prepare', [r for r in seeds if r['status'] == 'ready'], root / 'base')
    module('train', '--data', a.data, '--base', root / 'base/records.json', '--out', root / 'train')
    module('predict', '--checkpoint', root / 'train/checkpoint.npz', '--inputs', root / 'base/records.json', '--output', root / 'inference.json')
    native('apply', json.loads((root / 'inference.json').read_text()), root / 'learned')
    module('evaluate_pilot', '--root', root)
    manifest = dict(scope='training_reconstruction_four_arm_directions', requested_cases=a.cases, paid_provider_calls=0, full_pose_complete=False, mesh_checked=False, sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo / 'experiments/refine_v5').glob('*.py')})
    (root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(root / 'learned-metrics.json')
if __name__ == '__main__':
    main()
