"""Summarize completed paired from-scratch screening runs copied to one root."""
import argparse
import csv
import json
from pathlib import Path


def mean(rows, select=lambda sigma: True):
    values = [r['mse'] for r in rows if select(r['measurement_sigma'])]
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    records = []
    for mode in ('variance', 'sigma', 'adanorm'):
        directory = args.root / mode
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['status'] != 'completed':
            raise ValueError(f'{mode} has not completed')
        final_step = metadata['args']['steps']
        final = json.loads((directory / f'val_{final_step:06d}.json').read_text())['results']
        summary[mode] = dict(curve=metadata['results'], best_step=metadata['best_step'],
            best_mse=metadata['results'][str(metadata['best_step'])], final_step=final_step,
            final_mse=mean(final), low_mse=mean(final, lambda s: s <= .03),
            high_mse=mean(final, lambda s: s >= .05),
            per_sigma={str(s): mean(final, lambda v: v == s) for s in (0., .01, .03, .05, .07, .09)},
            elapsed_seconds=metadata['elapsed_seconds'])
        ablation = directory / f'val_{final_step:06d}_reported_zero.json'
        if ablation.exists():
            rows = json.loads(ablation.read_text())['results']
            summary[mode]['reported_zero_mse'] = mean(rows)
        for path in sorted(directory.glob('val_*.json')):
            result = json.loads(path.read_text())
            records.extend(dict(mode=mode, step=result['checkpoint_step'],
                condition='reported_zero' if 'reported_zero' in path.name else 'correct', **r)
                for r in result['results'])
        (args.output / f'{mode}_metadata.json').write_text(json.dumps(metadata, indent=2))
    base = summary['variance']
    for mode, row in summary.items():
        row['relative_to_variance_percent'] = {key: 100 * (row[key] / base[key] - 1)
                                                for key in ('final_mse', 'low_mse', 'high_mse')}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output / 'conditions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, row in summary.items():
        axes[0].plot([int(s) for s in row['curve']], list(row['curve'].values()), 'o-', label=mode)
        axes[1].plot([float(s) for s in row['per_sigma']], list(row['per_sigma'].values()), 'o-', label=mode)
    axes[0].set(xlabel='Training steps from random initialization', ylabel='Validation MSE')
    axes[1].set(xlabel='Observation sigma', ylabel='MSE at step 4000')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'screening.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
